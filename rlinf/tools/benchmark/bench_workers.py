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
import json
import os
from typing import Any

import torch

from rlinf.envs import get_env_cls
from rlinf.envs.utils import get_env_attr
from rlinf.tools.benchmark.fake_messages import (
    build_actor_pool,
    infer_batch_size,
    load_rollout_obs,
    load_trajectories,
    resize_batch,
)
from rlinf.tools.benchmark.measure import progress_iter, timed_once, timed_vram
from rlinf.tools.benchmark.message_capture import (
    CaptureEnvWorker,
    CaptureRolloutWorker,
)
from rlinf.tools.benchmark.sweeps import make_sizes, parse_multipliers
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.metric_utils import append_to_dict
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

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


class BenchRolloutWorker(CaptureRolloutWorker):
    """Sweep the batch size fed to a single ``rollout.predict``.

    Subclasses ``CaptureRolloutWorker`` so that, when ``RLINF_BENCH_CAPTURE_DIR``
    is armed for the inline capture round, ``_build_policy_output`` dumps the
    rollout->env message; the sweep itself is unaffected (the dump fires at most
    once, during the real generate round that precedes the sweep).
    """

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
        nvml_index = _nvml_index()

        # With rollout.enable_offload, init_worker leaves the HF model on CPU
        # (offload_model). The real generate() calls reload_model() before every
        # forward; we must onload here too, else predict runs on CPU -> zero GPU
        # utilization and each call takes minutes. We move the weights on
        # directly (not via reload_model) so no batch-locked CUDA graph is
        # captured -- the whole point here is to vary the batch size. We also
        # time this onload (and the closing offload) so the sweep reports the
        # cost of getting the rollout weights on/off the GPU.
        rollout_offloaded = getattr(self, "enable_offload", False)
        onload_stats: dict[str, Any] | None = None
        if rollout_offloaded:

            def _onload_rollout():
                self.hf_model.to(self.device)
                if getattr(self, "rlt_feature_model", None) is not None:
                    self.rlt_feature_model.to(self.device)
                if getattr(self, "expert_model", None) is not None:
                    self.expert_model.to(self.device)

            onload_stats = timed_once(
                _onload_rollout, sample_gpu=True, device_index=nvml_index
            )

        records: list[dict[str, Any]] = []
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

        # Restore the post-init_worker state (weights back on CPU), timing it.
        offload_stats: dict[str, Any] | None = None
        if rollout_offloaded:
            offload_stats = timed_once(
                self.offload_model, sample_gpu=True, device_index=nvml_index
            )

        return {
            "rank": self._rank,
            "default_batch": default_b,
            "records": records,
            "onload": onload_stats,
            "offload": offload_stats,
        }

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
        nvml_index = _nvml_index()

        # After init_worker the FSDP weights/optimizer are offloaded to CPU when
        # enable_offload is set (see EmbodiedFSDPActor.init_worker). The real
        # run_training onloads them before touching the model, so we must too --
        # otherwise the first forward hits "FSDP-managed module ... on cpu". We
        # time this onload (and the closing offload) to report the cost of
        # moving the actor weights/optimizer on/off the GPU.
        weights_were_offloaded = getattr(self, "is_weight_offloaded", False)
        optim_was_offloaded = getattr(self, "is_optimizer_offloaded", False)
        onload_stats: dict[str, Any] | None = None
        if weights_were_offloaded or optim_was_offloaded:

            def _onload_actor():
                if weights_were_offloaded:
                    self.load_param_and_grad(self.device)
                if optim_was_offloaded:
                    self.load_optimizer(self.device)

            onload_stats = timed_once(
                _onload_actor, sample_gpu=True, device_index=nvml_index
            )
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

        # Restore the post-init_worker state (weights/optimizer back on CPU),
        # timing the offload back to host.
        offload_stats: dict[str, Any] | None = None
        if (weights_were_offloaded and not self.is_weight_offloaded) or (
            optim_was_offloaded and not self.is_optimizer_offloaded
        ):

            def _offload_actor():
                if weights_were_offloaded and not self.is_weight_offloaded:
                    self.offload_param_and_grad()
                if optim_was_offloaded and not self.is_optimizer_offloaded:
                    self.offload_optimizer()

            offload_stats = timed_once(
                _offload_actor, sample_gpu=True, device_index=nvml_index
            )

        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "global_batch_per_rank": global_per_rank,
            "default_micro": default_micro,
            "records": records,
            "onload": onload_stats,
            "offload": offload_stats,
        }


class BenchEnvWorker(CaptureEnvWorker):
    """Sweep the env parallelism (num_envs) fed to a single ``env_interact_step``.

    The simulator allocates exactly ``num_envs`` parallel sims at construction,
    so -- unlike the rollout/actor sweeps -- each point rebuilds a fresh single
    env at the swept size, resets it, times the atomic op, then frees it. Since
    the env is CPU-heavy, the measurement also profiles host RAM, process CPU%
    and GPU util in addition to wall-time and HBM.

    Subclasses ``CaptureEnvWorker`` so the inline capture round (when armed via
    ``RLINF_BENCH_CAPTURE_DIR``) dumps the env->rollout obs and env->actor
    trajectory; the sweep's direct ``env_interact_step`` calls are no-ops for
    capture (each message dumps at most once, during the real round beforehand).
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
    def _dispose_env(env) -> None:
        """Tear an env down the RLinf-native way, deterministically.

        RoboTwin / ManiSkill-offload envs expose ``offload`` -- which routes
        through the real ``VectorEnv.close`` (RoboTwin) or the offload
        subprocess kill (ManiSkill) -- as their teardown. ``RoboTwinEnv`` has no
        ``close`` (it is ``gym.Env.close``, a no-op), so calling ``close`` would
        leave the SAPIEN engines *and* the ``VectorEnv``'s never-shut-down
        ``ThreadPoolExecutor`` to be finalized later inside a ``gc.collect()`` on
        a pool worker thread whose Python thread-state is already gone -- the
        "pybind11 inc_ref() while GIL not held" / "PyGILState_Release" crash. So
        we (1) prefer ``offload`` over ``close`` (mirrors what env_worker does in
        production) and (2) join the SAPIEN thread pool while we still hold the
        GIL, before dropping the last reference, so no SAPIEN C++ destructor ever
        runs on a pool thread after teardown.
        """
        offload = get_env_attr(env, "offload")
        if callable(offload):
            try:
                offload(clear_cache=True)
            except TypeError:
                offload()  # envs whose offload() takes no clear_cache arg
            except Exception:
                pass
        else:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        # RoboTwin's VectorEnv.close() empties the scenes but never shuts its
        # ThreadPoolExecutor down; join it here (GIL held) so its worker threads
        # are gone before gc finalizes any remaining SAPIEN objects.
        venv = get_env_attr(env, "venv")
        pool = getattr(venv, "env_thread_pool", None) if venv is not None else None
        if pool is not None:
            try:
                pool.shutdown(wait=True)
            except Exception:
                pass

    def _leak_failed_env(self, temp_env, original_env_list) -> None:
        """Contain a failed sweep point without crashing the worker.

        A half-built RoboTwin env cannot be finalized safely: its native SAPIEN
        thread pool sits in a reference cycle, and collecting it runs a C++
        destructor on a pool thread with no Python thread-state -- the fatal
        "PyGILState_Release: ... no thread-state for this thread" abort that
        kills the worker (and, being a group call, aborts the whole job). So
        rather than dispose it we (1) disable gc so no automatic pass ever
        collects the cycle, (2) pin a strong reference to whatever got built so
        refcounting cannot free it either, and (3) restore the persistent
        env_list reference. The env leaks deliberately: the sweep is the
        worker's last task, so the process exits (partial records already
        returned to the driver) before anything would finalize it.
        """
        gc.disable()
        if temp_env is not None:
            leaked = getattr(self, "_bench_leaked_envs", None)
            if leaked is None:
                leaked = self._bench_leaked_envs = []
            leaked.append(temp_env)
        self.env_list = original_env_list

    def bench_interact_sweep(
        self,
        *,
        seed_dir: str,
        multipliers: str | None = None,
        warmup: int = 2,
        repeats: int = 5,
        cap: int | None = None,
        partial_path: str | None = None,
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

        # Each sweep point builds a fresh num_envs-wide RoboTwin/SAPIEN sim whose
        # renderers leak Vulkan/OIDN TLS pthread-keys that are never reclaimed
        # in-process (see _leak_failed_env). Building points ABOVE the production
        # size accumulates those keys until CPython itself can't allocate a
        # thread-state key -> an UNCATCHABLE native SIGABRT ("Couldn't create
        # autoTSSkey mapping") that no try/except can contain. Warn if the sweep
        # is unbounded and reaches into that regime; the caller can cap it with
        # RLINF_BENCH_ENV_MAX.
        if cap is None and any(s > default_n for s in sizes):
            self.log_warning(
                f"env.interact sweep will build num_envs points above the "
                f"production size ({default_n}): {[s for s in sizes if s > default_n]}. "
                "On single-GPU RoboTwin this risks an uncatchable native "
                "SAPIEN/OIDN TLS-key SIGABRT; set RLINF_BENCH_ENV_MAX to cap it."
            )

        def _flush_partial(record: dict[str, Any]) -> None:
            """Append one measured record to the recovery file (rank 0 only).

            Bounding avoids the native crash but cannot *contain* it, so we stream
            each row to disk as it is produced: if a later build still aborts
            natively, the points already measured survive here. Best-effort --
            never let a logging error break the sweep.
            """
            if partial_path is None or self._rank != 0:
                return
            try:
                with open(partial_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass

        env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        env_cfg = self.cfg.env.train
        nvml_index = _nvml_index()
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )

        original_env_list = self.env_list
        # The env built at init_worker (and used by the inline capture round)
        # still owns a live SAPIEN thread pool / scenes. Tear it down
        # deterministically BEFORE building any swept env, so only one set of
        # SAPIEN engines+threads is ever alive in-process at a time -- coexisting
        # pools (plus RoboTwin re-forcing the "spawn" start method on every
        # build) are what make its pybind11 teardown race the GIL. The sweep is
        # the worker's last task, so these envs are never needed again.
        for persistent_env in original_env_list:
            self._dispose_env(persistent_env)
        gc.collect()
        torch.cuda.empty_cache()

        records: list[dict[str, Any]] = []
        bar = progress_iter(
            sizes,
            desc=f"env.interact sweep [rank{self._rank}]",
            enabled=(self._rank == 0),
        )
        for n in bar:
            # One sweep point must never take down the worker. A build / reset /
            # step can fail beyond a torch OOM -- e.g. SAPIEN render-buffer
            # exhaustion ("RuntimeError: cannot create buffer") when num_envs on
            # a single GPU gets too large. Catching the Python exception is NOT
            # enough on its own: the half-built RoboTwin env's native SAPIEN
            # thread pool sits in a reference cycle, so letting gc finalize it
            # runs a C++ destructor on a pool thread with no Python thread-state
            # -- the fatal "PyGILState_Release" crash. Therefore the teardown
            # that reclaims memory between points runs ONLY on the success path,
            # where the env is fully built and _dispose_env can join its thread
            # pool GIL-safely. On ANY failure we hand off to _leak_failed_env
            # (disable gc, pin the partial env, skip dispose/gc) and break --
            # returning the points already measured instead of crashing.
            temp_env = None
            stop = False
            try:
                # Bootstrap = build the num_envs-wide simulator + reset it. This
                # is the env's "onload": the SAPIEN engines/scenes/render buffers
                # get allocated here, and it is the CPU/host-RAM-heavy part that
                # scales with num_envs. Time it as one unit. Assign temp_env via
                # nonlocal so a partial env is still captured if reset() raises.
                def _bootstrap(nn=n):
                    nonlocal temp_env
                    temp_env = self._build_single_env(env_cls, env_cfg, nn)
                    temp_env.reset()

                bootstrap_stats = timed_once(
                    _bootstrap,
                    sample_host=True,
                    sample_gpu=True,
                    device_index=nvml_index,
                )
                actions_n = resize_batch(seed_actions, n).to(device)
                self.env_list = [temp_env]
                rec = timed_vram(
                    lambda a=actions_n: self.env_interact_step(a, 0),
                    warmup=warmup,
                    repeats=repeats,
                    sample_host=True,
                    sample_gpu=True,
                    device_index=nvml_index,
                )
                rec.update(
                    batch_size=n,
                    op="env.interact_step",
                    is_default=(n == default_n),
                    oom=False,
                    bootstrap_ms=bootstrap_stats["ms"],
                    bootstrap_host_rss_peak_MB=bootstrap_stats.get(
                        "host_rss_peak_MB"
                    ),
                    bootstrap_peak_alloc_MB=bootstrap_stats.get("peak_alloc_MB"),
                )
                # Success: the env is fully built, so tear it down the GIL-safe
                # way and reclaim host/GPU memory before the next (larger) point.
                # The dispose is the env's "offload"; time it too.
                self.env_list = original_env_list
                offload_stats = timed_once(
                    lambda e=temp_env: self._dispose_env(e),
                    sample_host=True,
                    sample_gpu=True,
                    device_index=nvml_index,
                )
                rec["offload_ms"] = offload_stats["ms"]
                temp_env = None
                records.append(rec)
                _flush_partial(rec)
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix_str(
                        f"n={n} step{rec['ms_mean']:.0f}ms "
                        f"boot{rec['bootstrap_ms']:.0f}ms "
                        f"off{rec['offload_ms']:.0f}ms "
                        f"rss{rec.get('host_rss_peak_MB', 0):.0f}MB "
                        f"gpu{rec.get('gpu_util_mean', 0):.0f}%"
                    )
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                oom_rec = {"batch_size": n, "op": "env.interact_step", "oom": True}
                records.append(oom_rec)
                _flush_partial(oom_rec)
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix_str(f"n={n} OOM (stop)")
                self._leak_failed_env(temp_env, original_env_list)
                stop = True
            except Exception as exc:
                # Non-OOM failure (SAPIEN buffer/render, planner, driver, ...).
                # Keep the harness alive and remember why this point failed.
                self.log_warning(
                    f"env.interact sweep point num_envs={n} failed "
                    f"({type(exc).__name__}: {exc}); recording and stopping "
                    "the env sweep."
                )
                err_rec = {
                    "batch_size": n,
                    "op": "env.interact_step",
                    "oom": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                records.append(err_rec)
                _flush_partial(err_rec)
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix_str(f"n={n} ERR (stop)")
                self._leak_failed_env(temp_env, original_env_list)
                stop = True

            if stop:
                break

        return {"rank": self._rank, "default_batch": default_n, "records": records}


class BenchNFTEmbodiedActor(EmbodiedNFTFSDPPolicy):
    """Sweep the actor micro batch size for the NFT (diffusion) policy.

    Mirrors ``BenchEmbodiedActor`` but inherits from ``EmbodiedNFTFSDPPolicy``
    so that ``init_worker`` runs the NFT-specific initialization (including
    ``init_rollout_model``) and the worker loads the diffusion model pipeline
    (VAE, text encoder, transformer) correctly.
    """

    def _nft_train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        """NFT-specific train_micro_batch override.

        The base ``EmbodiedFSDPActor.train_micro_batch`` calls
        ``self.model(forward_inputs=forward_inputs, ...)`` with the default
        ``ForwardType.DEFAULT``, which raises ``NotImplementedError`` for the
        Wan2.2 model (``"Wan2.2 flow-grpo is unavailable now. Use
        rl_mode='nft'."``).  Instead, this method uses the NFT-specific
        ``nft_forward_and_loss`` path, matching ``EmbodiedNFTFSDPPolicy.run_training``.
        """
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        loss, metrics_data = self.nft_forward_and_loss(micro_batch)
        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()
        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def bench_micro_batch_sweep(
        self,
        *,
        seed_dir: str,
        multipliers: str | None = None,
        warmup: int = 1,
        repeats: int = 3,
    ) -> dict[str, Any]:
        trajs = load_trajectories(os.path.join(seed_dir, _ACTOR_SEED))
        nvml_index = _nvml_index()

        weights_were_offloaded = getattr(self, "is_weight_offloaded", False)
        optim_was_offloaded = getattr(self, "is_optimizer_offloaded", False)
        onload_stats: dict[str, Any] | None = None
        if weights_were_offloaded or optim_was_offloaded:

            def _onload_actor():
                if weights_were_offloaded:
                    self.load_param_and_grad(self.device)
                if optim_was_offloaded:
                    self.load_optimizer(self.device)

            onload_stats = timed_once(
                _onload_actor, sample_gpu=True, device_index=nvml_index
            )
        self.model.train()

        def _compute_adv(rollout_batch):
            # NFT diffusion models do not have a ``loss_mask`` in the seed
            # trajectory (it is normally built during the rollout generation loop
            # from the ``dones`` tensor). Without it ``compute_grpo_video_advantages``
            # fails with ``TypeError: unsupported operand type * 'Tensor' and
            # 'NoneType'``. Create a unit mask so every entry is valid.
            if rollout_batch.get("loss_mask") is None:
                rollout_batch["loss_mask"] = torch.ones_like(
                    rollout_batch["rewards"], dtype=torch.bool
                )
            self.rollout_batch = rollout_batch
            self.compute_advantages_and_returns()
            return self.rollout_batch

        pool = build_actor_pool(trajs, _compute_adv)

        global_per_rank = self.cfg.actor.global_batch_size // self._world_size
        pool = resize_batch(pool, global_per_rank)

        # NFT models need ``_precompute_nft_training_inputs`` to populate
        # ``nft_xcur``, ``nft_step_index``, and ``nft_v`` in the forward_inputs
        # before the training loop.  This mirrors what
        # ``EmbodiedNFTFSDPPolicy.run_training`` does.
        self.rollout_batch = pool
        self._precompute_nft_training_inputs()
        pool = self.rollout_batch

        default_micro = self.cfg.actor.micro_batch_size
        micro_sizes = make_sizes(
            default_micro,
            parse_multipliers(multipliers),
            divisor_of=global_per_rank,
        )

        records: list[dict[str, Any]] = []
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
                    self._nft_train_micro_batch(
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

        offload_stats: dict[str, Any] | None = None
        if (weights_were_offloaded and not self.is_weight_offloaded) or (
            optim_was_offloaded and not self.is_optimizer_offloaded
        ):

            def _offload_actor():
                if weights_were_offloaded and not self.is_weight_offloaded:
                    self.offload_param_and_grad()
                if optim_was_offloaded and not self.is_optimizer_offloaded:
                    self.offload_optimizer()

            offload_stats = timed_once(
                _offload_actor, sample_gpu=True, device_index=nvml_index
            )

        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "global_batch_per_rank": global_per_rank,
            "default_micro": default_micro,
            "records": records,
            "onload": onload_stats,
            "offload": offload_stats,
        }
