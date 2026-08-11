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

"""Benchmark worker subclasses that sweep one dimension on the live workers.

Both subclasses reuse the *real* atomic ops (``predict`` / ``train_micro_batch``)
so the measured compute and VRAM are production-faithful. Only the batch axis of
the fed data is varied; every other dimension comes from the captured default
run. Nothing here hardcodes a shape or an absolute sweep value.

- ``BenchRolloutWorker.bench_predict_sweep``   : vary the ``predict`` batch size.
- ``BenchEmbodiedActor.bench_micro_batch_sweep``: pin the global batch, vary the
  micro batch size (so a micro size must divide the per-rank global batch).
- ``BenchEnvWorker.bench_interact_sweep``       : vary the env parallelism
  (num_envs). The simulator bakes num_envs in at construction, so each sweep
  point rebuilds a fresh single env; the env being CPU-heavy, this sweep also
  profiles host RAM / CPU% / GPU util alongside wall-time and HBM.
"""

from __future__ import annotations

import gc
import os
from typing import Any

import torch

from rlinf.envs import get_env_cls
from rlinf.tools.benchmark.fake_messages import (
    build_actor_pool,
    infer_batch_size,
    load_rollout_obs,
    load_trajectories,
    resize_batch,
)
from rlinf.tools.benchmark.measure import progress_iter, timed_vram
from rlinf.tools.benchmark.sweeps import make_sizes, parse_multipliers
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

_ROLLOUT_SEED = "env_to_rollout_obs.pt"
_ACTOR_SEED = "env_to_actor_trajectory.pt"
_ENV_SEED = "rollout_policy_output.pt"


def _nvml_index() -> int | None:
    """Best-effort physical NVML index for the current CUDA device.

    Ray sets ``CUDA_VISIBLE_DEVICES`` per worker, so the local torch ordinal maps
    into that list to recover the physical GPU NVML reads. Falls back to the
    local ordinal. GPU util is a secondary metric here (env is CPU-heavy), so a
    slight index mismatch under non-default CUDA_DEVICE_ORDER is not critical.
    """
    if not torch.cuda.is_available():
        return None
    local = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        parts = [p for p in visible.split(",") if p != ""]
        try:
            return int(parts[local])
        except (IndexError, ValueError):
            return local
    return local


class BenchRolloutWorker(MultiStepRolloutWorker):
    """Sweep the batch size fed to a single ``rollout.predict``."""

    def bench_predict_sweep(
        self,
        *,
        seed_dir: str,
        multipliers: str | None = None,
        warmup: int = 2,
        repeats: int = 5,
        cap: int | None = None,
    ) -> dict[str, Any]:
        seed_obs = load_rollout_obs(os.path.join(seed_dir, _ROLLOUT_SEED))
        default_b = infer_batch_size(seed_obs)
        sizes = make_sizes(default_b, parse_multipliers(multipliers), cap=cap)

        # With rollout.enable_offload, init_worker leaves the HF model on CPU
        # (offload_model). The real generate() calls reload_model() before every
        # forward; we must onload here too, else predict runs on CPU -> zero GPU
        # utilization and each call takes minutes. We move the weights on
        # directly (not via reload_model) so no batch-locked CUDA graph is
        # captured -- the whole point here is to vary the batch size.
        rollout_offloaded = getattr(self, "enable_offload", False)
        if rollout_offloaded:
            self.hf_model.to(self.device)
            if getattr(self, "rlt_feature_model", None) is not None:
                self.rlt_feature_model.to(self.device)
            if getattr(self, "expert_model", None) is not None:
                self.expert_model.to(self.device)

        records: list[dict[str, Any]] = []
        nvml_index = _nvml_index()
        bar = progress_iter(
            sizes,
            desc=f"rollout.predict sweep [rank{self._rank}]",
            enabled=(self._rank == 0),
        )
        for b in bar:
            obs = resize_batch(seed_obs, b)
            try:
                rec = timed_vram(
                    lambda o=obs: self.predict(o, "train"),
                    warmup=warmup,
                    repeats=repeats,
                    sample_gpu=True,
                    device_index=nvml_index,
                )
                rec["oom"] = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                records.append(
                    {"batch_size": b, "op": "rollout.predict", "oom": True}
                )
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix_str(f"b={b} OOM (stop)")
                break
            rec.update(
                batch_size=b,
                op="rollout.predict",
                is_default=(b == default_b),
            )
            records.append(rec)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix_str(
                    f"b={b} {rec['ms_mean']:.1f}ms {rec['peak_alloc_MB']:.0f}MB "
                    f"gpu{rec.get('gpu_util_mean', 0):.0f}%"
                )
            torch.cuda.empty_cache()

        # Restore the post-init_worker state (weights back on CPU).
        if rollout_offloaded:
            self.offload_model()

        return {"rank": self._rank, "default_batch": default_b, "records": records}

    def _bench_compute_adv(self, rollout_batch):  # pragma: no cover - unused hook
        raise NotImplementedError


class BenchEmbodiedActor(EmbodiedFSDPActor):
    """Sweep the actor micro batch size with the global batch pinned."""

    def bench_micro_batch_sweep(
        self,
        *,
        seed_dir: str,
        multipliers: str | None = None,
        warmup: int = 1,
        repeats: int = 3,
    ) -> dict[str, Any]:
        trajs = load_trajectories(os.path.join(seed_dir, _ACTOR_SEED))

        # After init_worker the FSDP weights/optimizer are offloaded to CPU when
        # enable_offload is set (see EmbodiedFSDPActor.init_worker). The real
        # run_training onloads them before touching the model, so we must too --
        # otherwise the first forward hits "FSDP-managed module ... on cpu".
        weights_were_offloaded = getattr(self, "is_weight_offloaded", False)
        optim_was_offloaded = getattr(self, "is_optimizer_offloaded", False)
        if weights_were_offloaded:
            self.load_param_and_grad(self.device)
        if optim_was_offloaded:
            self.load_optimizer(self.device)
        self.model.train()

        def _compute_adv(rollout_batch):
            # Hand the batch to the live actor and let it add advantages/returns.
            self.rollout_batch = rollout_batch
            self.compute_advantages_and_returns()
            return self.rollout_batch

        pool = build_actor_pool(trajs, _compute_adv)

        # Global batch is pinned: only the micro batch size may change, so a
        # micro size must evenly divide the per-rank global batch.
        global_per_rank = self.cfg.actor.global_batch_size // self._world_size
        pool = resize_batch(pool, global_per_rank)

        default_micro = self.cfg.actor.micro_batch_size
        micro_sizes = make_sizes(
            default_micro,
            parse_multipliers(multipliers),
            divisor_of=global_per_rank,
        )

        records: list[dict[str, Any]] = []
        nvml_index = _nvml_index()
        bar = progress_iter(
            micro_sizes,
            desc=f"actor.micro_batch sweep [rank{self._rank}]",
            enabled=(self._rank == 0),
        )
        for m in bar:
            num_micro = global_per_rank // m
            micro_batches = split_dict_to_chunk(pool, num_micro)
            self.gradient_accumulation = num_micro

            def _run_global_batch(mbs=micro_batches, cnt=num_micro):
                self.optimizer.zero_grad()
                for idx, mb in enumerate(mbs):
                    self.train_micro_batch(
                        micro_batch=mb,
                        metrics={},
                        is_last=(idx + 1) == cnt,
                    )

            try:
                rec = timed_vram(
                    _run_global_batch,
                    warmup=warmup,
                    repeats=repeats,
                    sample_gpu=True,
                    device_index=nvml_index,
                )
                rec["oom"] = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                records.append(
                    {
                        "micro_batch_size": m,
                        "op": "actor.train_micro_batch",
                        "oom": True,
                    }
                )
                self.optimizer.zero_grad()
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix_str(f"micro={m} OOM")
                continue
            rec.update(
                micro_batch_size=m,
                num_micro_batches=num_micro,
                global_batch_per_rank=global_per_rank,
                op="actor.train_micro_batch",
                is_default=(m == default_micro),
            )
            records.append(rec)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix_str(
                    f"micro={m} x{num_micro} {rec['ms_mean']:.1f}ms "
                    f"{rec['peak_alloc_MB']:.0f}MB gpu{rec.get('gpu_util_mean', 0):.0f}%"
                )
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

        # Restore the post-init_worker state (weights/optimizer back on CPU).
        if weights_were_offloaded and not self.is_weight_offloaded:
            self.offload_param_and_grad()
        if optim_was_offloaded and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "global_batch_per_rank": global_per_rank,
            "default_micro": default_micro,
            "records": records,
        }


class BenchEnvWorker(EnvWorker):
    """Sweep the env parallelism (num_envs) fed to a single ``env_interact_step``.

    The simulator allocates exactly ``num_envs`` parallel sims at construction,
    so -- unlike the rollout/actor sweeps -- each point rebuilds a fresh single
    env at the swept size, resets it, times the atomic op, then frees it. Since
    the env is CPU-heavy, the measurement also profiles host RAM, process CPU%
    and GPU util in addition to wall-time and HBM.
    """

    def _build_single_env(self, env_cls, env_cfg, num_envs: int):
        """Build one env instance at ``num_envs`` (a single pipeline stage)."""
        saved_stage_num = self.stage_num
        self.stage_num = 1  # build only one stage's env
        try:
            envs = self._setup_env_and_wrappers(
                env_cls=env_cls,
                env_cfg=env_cfg,
                num_envs_per_stage=num_envs,
            )
        finally:
            self.stage_num = saved_stage_num
        return envs[0]

    @staticmethod
    def _close_env(env) -> None:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def bench_interact_sweep(
        self,
        *,
        seed_dir: str,
        multipliers: str | None = None,
        warmup: int = 2,
        repeats: int = 5,
        cap: int | None = None,
    ) -> dict[str, Any]:
        # Seed = the captured PolicyOutput (rollout -> env). Its .actions tensor is
        # [num_envs, chunk, action_dim]; dim 0 is the env-parallelism axis we vary.
        policy_output = torch.load(
            os.path.join(seed_dir, _ENV_SEED), map_location="cpu", weights_only=False
        )
        seed_actions = (
            policy_output.actions
            if hasattr(policy_output, "actions")
            else policy_output["actions"]
        )

        default_n = int(self.train_num_envs_per_stage)
        sizes = make_sizes(default_n, parse_multipliers(multipliers), cap=cap)

        env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        env_cfg = self.cfg.env.train
        nvml_index = _nvml_index()
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )

        original_env_list = self.env_list
        records: list[dict[str, Any]] = []
        bar = progress_iter(
            sizes,
            desc=f"env.interact sweep [rank{self._rank}]",
            enabled=(self._rank == 0),
        )
        for n in bar:
            temp_env = self._build_single_env(env_cls, env_cfg, n)
            actions_n = resize_batch(seed_actions, n).to(device)
            try:
                temp_env.reset()
                self.env_list = [temp_env]
                rec = timed_vram(
                    lambda a=actions_n: self.env_interact_step(a, 0),
                    warmup=warmup,
                    repeats=repeats,
                    sample_host=True,
                    sample_gpu=True,
                    device_index=nvml_index,
                )
                rec["oom"] = False
            except torch.cuda.OutOfMemoryError:
                records.append(
                    {"batch_size": n, "op": "env.interact_step", "oom": True}
                )
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix_str(f"n={n} OOM (stop)")
                self._close_env(temp_env)
                self.env_list = original_env_list
                del temp_env, actions_n
                gc.collect()
                torch.cuda.empty_cache()
                break
            finally:
                self.env_list = original_env_list

            self._close_env(temp_env)
            del temp_env, actions_n
            gc.collect()
            torch.cuda.empty_cache()

            rec.update(
                batch_size=n,
                op="env.interact_step",
                is_default=(n == default_n),
            )
            records.append(rec)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix_str(
                    f"n={n} {rec['ms_mean']:.1f}ms "
                    f"cpu{rec.get('cpu_percent_mean', 0):.0f}% "
                    f"rss{rec.get('host_rss_peak_MB', 0):.0f}MB "
                    f"gpu{rec.get('gpu_util_mean', 0):.0f}%"
                )

        return {"rank": self._rank, "default_batch": default_n, "records": records}
