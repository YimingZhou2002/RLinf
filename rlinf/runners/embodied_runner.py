# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer
from rlinf.utils.integrated_profiler import IntegratedProfiler
from rlinf.utils.fine_grained_profiler import (
    enable_profiling,
    disable_profiling,
    is_profiling_enabled,
    TimelineAggregator,
)
from rlinf.utils.pipeline_visualizer import PipelineVisualizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )
        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()
        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    def run(self):
        start_step = self.global_step
        start_time = time.time()

        # Initialize integrated profiler
        enable_timeline_profile = self.cfg.runner.get("enable_timeline_profile", False)
        profile_output_dir = os.path.join(self.cfg.runner.logger.log_path, "profile")
        profiler = IntegratedProfiler.get_instance()
        if enable_timeline_profile:
            profiler._enabled = True
            profiler._output_dir = profile_output_dir

        # Initialize fine-grained profiler for pipeline bubble chart
        enable_fine_grained_profile = self.cfg.runner.get(
            "enable_fine_grained_profile", False
        )
        fine_grained_profile_interval = self.cfg.runner.get(
            "fine_grained_profile_interval", 10
        )  # Profile every N steps
        fine_grained_profile_output_dir = os.path.join(
            self.cfg.runner.logger.log_path, "fine_grained_profile"
        )
        if enable_fine_grained_profile:
            os.makedirs(fine_grained_profile_output_dir, exist_ok=True)
            enable_profiling()
            self.logger.info(
                f"Fine-grained profiling enabled. Output dir: {fine_grained_profile_output_dir}"
            )

        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            # Start profiling for this step
            if enable_timeline_profile:
                profiler.start_step()

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        if enable_timeline_profile:
                            with profiler.record("sync_weights", "runner"):
                                self.update_rollout_weights()
                        else:
                            self.update_rollout_weights()

                with self.timer("generate_rollouts"):
                    if enable_timeline_profile:
                        # env_interact includes the wait time
                        with profiler.record("env_interact", "env"):
                            env_handle: Handle = self.env.interact(
                                input_channel=self.env_channel,
                                rollout_channel=self.rollout_channel,
                                reward_channel=self.reward_channel,
                                actor_channel=self.actor_channel,
                            )
                        # rollout_generate includes the wait time
                        with profiler.record("rollout_generate", "rollout"):
                            rollout_handle: Handle = self.rollout.generate(
                                input_channel=self.rollout_channel,
                                output_channel=self.env_channel,
                            )
                        if self.reward is not None:
                            with profiler.record("reward_compute", "reward"):
                                reward_handle: Handle = self.reward.compute_rewards(
                                    input_channel=self.reward_channel,
                                    output_channel=self.env_channel,
                                )
                        # recv_trajectories - this waits for data transfer
                        with profiler.record("recv_trajectories", "runner"):
                            self.actor.recv_rollout_trajectories(
                                input_channel=self.actor_channel
                            ).wait()
                        # Wait for env and rollout to complete
                        with profiler.record("env_wait", "env"):
                            env_results = env_handle.wait()
                        with profiler.record("rollout_wait", "rollout"):
                            rollout_results = rollout_handle.wait()
                        if self.reward is not None:
                            with profiler.record("reward_wait", "reward"):
                                reward_handle.wait()
                    else:
                        env_handle: Handle = self.env.interact(
                            input_channel=self.env_channel,
                            rollout_channel=self.rollout_channel,
                            reward_channel=self.reward_channel,
                            actor_channel=self.actor_channel,
                        )
                        rollout_handle: Handle = self.rollout.generate(
                            input_channel=self.rollout_channel,
                            output_channel=self.env_channel,
                        )
                        if self.reward is not None:
                            reward_handle: Handle = self.reward.compute_rewards(
                                input_channel=self.reward_channel,
                                output_channel=self.env_channel,
                            )
                        self.actor.recv_rollout_trajectories(
                            input_channel=self.actor_channel
                        ).wait()
                        env_results = env_handle.wait()
                        rollout_results = rollout_handle.wait()
                        if self.reward is not None:
                            reward_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    if enable_timeline_profile:
                        with profiler.record("compute_advantages", "runner"):
                            actor_rollout_metrics = (
                                self.actor.compute_advantages_and_returns().wait()
                            )
                    else:
                        actor_rollout_metrics = (
                            self.actor.compute_advantages_and_returns().wait()
                        )

                # actor training.
                if enable_timeline_profile:
                    with profiler.record("actor_training_submit", "runner"):
                        actor_training_handle: Handle = self.actor.run_training()
                else:
                    actor_training_handle: Handle = self.actor.run_training()
                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                if enable_timeline_profile:
                    with profiler.record("actor_training_wait", "actor"):
                        actor_training_metrics = actor_training_handle.wait()
                else:
                    actor_training_metrics = actor_training_handle.wait()
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1

                # End profiling and save for this step
                if enable_timeline_profile:
                    profiler.end_step()
                    profiler.save_all(prefix=f"step_{_step}")
                    profiler.clear()

                # Collect fine-grained profiling intervals and visualize
                # Get profile intervals through separate method calls (not from return values)
                if enable_fine_grained_profile and _step % fine_grained_profile_interval == 0:
                    self._collect_and_visualize_fine_grained_profile(
                        _step,
                        fine_grained_profile_output_dir,
                    )

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
                return_per_rank=True
            )
            rollout_time_metrics, rollout_time_metrics_per_rank = (
                rollout_handle.consume_durations(return_per_rank=True)
            )
            actor_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/env/{k}": v for k, v in env_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
            )
            if self.reward is not None:
                reward_time_metrics, reward_time_metrics_per_rank = (
                    reward_handle.consume_durations(return_per_rank=True)
                )
                time_metrics.update(
                    {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
                )

            env_results = env_handle.wait()
            env_results_list = [
                results for results in env_results if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            ranked_env_results = [
                {"rank": rank, "env": rank_metrics}
                for rank, rank_metrics in enumerate(env_results)
                if rank_metrics is not None
            ]
            _, env_metrics_per_rank = self._process_ranked_eval_results(
                ranked_env_results, metric_field="env"
            )

            rollout_metrics = {
                f"rollout/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_rollout_metrics
                ).items()
            }
            training_metrics = {
                f"train/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_training_metrics
                ).items()
            }

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            self._log_ranked_metrics(
                metrics_list=actor_rollout_metrics,
                step=_step,
                prefix="rollout",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_training_metrics,
                step=_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
            )
            if self.reward is not None:
                self._log_ranked_metrics(
                    metrics_list=reward_time_metrics_per_rank,
                    step=_step,
                    prefix="time/reward",
                    worker_group_name=self.reward.worker_group_name,
                )

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    def _collect_and_visualize_fine_grained_profile(
        self,
        step: int,
        output_dir: str,
    ):
        """Collect fine-grained profiling intervals from workers via separate calls and visualize."""
        try:
            aggregator = TimelineAggregator()

            # Get intervals from env workers through separate method call
            import ray
            # WorkerGroup has _workers attribute which is list[WorkerRank]
            # Each WorkerRank has 'worker' (Ray actor) and 'rank' attributes
            if hasattr(self.env, "_workers") and self.env._workers:
                for worker_rank in self.env._workers:
                    try:
                        worker = worker_rank.worker  # Ray actor
                        rank = worker_rank.rank
                        # Ray remote call returns ObjectRef, need to get the actual value
                        intervals_ref = worker.get_profile_intervals.remote()
                        intervals = ray.get(intervals_ref)
                        if intervals:
                            aggregator.add_worker_intervals(f"env_rank{rank}", intervals)
                    except Exception as e:
                        self.logger.warning(f"Failed to get env worker {worker_rank.rank} intervals: {e}")

            # Get intervals from rollout workers through separate method call
            if hasattr(self.rollout, "_workers") and self.rollout._workers:
                for worker_rank in self.rollout._workers:
                    try:
                        worker = worker_rank.worker  # Ray actor
                        rank = worker_rank.rank
                        # Ray remote call returns ObjectRef, need to get the actual value
                        intervals_ref = worker.get_profile_intervals.remote()
                        intervals = ray.get(intervals_ref)
                        if intervals:
                            aggregator.add_worker_intervals(f"rollout_rank{rank}", intervals)
                    except Exception as e:
                        self.logger.warning(f"Failed to get rollout worker {worker_rank.rank} intervals: {e}")

            # Check if we have any intervals
            if not aggregator._intervals_by_worker:
                self.logger.warning("No intervals collected from workers.")
                return

            # Normalize timeline (align timestamps)
            aggregator.normalize_timeline()

            # Generate visualizations
            visualizer = PipelineVisualizer(figsize=(16, 10), dpi=100)

            # Save pipeline bubble chart
            bubble_chart_path = os.path.join(
                output_dir, f"pipeline_bubble_chart_step_{step}.png"
            )
            visualizer.plot(
                aggregator,
                save_path=bubble_chart_path,
                title=f"Pipeline Execution Timeline - Step {step}",
            )

            # Save efficiency analysis
            efficiency_path = os.path.join(
                output_dir, f"pipeline_efficiency_step_{step}.png"
            )
            visualizer.plot_pipeline_efficiency(
                aggregator, save_path=efficiency_path
            )

            # Save Chrome trace format for interactive viewing
            chrome_trace_path = os.path.join(
                output_dir, f"timeline_step_{step}.json"
            )
            aggregator.save_chrome_trace(chrome_trace_path)

            self.logger.info(
                f"Fine-grained profile saved for step {step}: "
                f"{bubble_chart_path}, {efficiency_path}, {chrome_trace_path}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to collect fine-grained profile: {e}")

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
