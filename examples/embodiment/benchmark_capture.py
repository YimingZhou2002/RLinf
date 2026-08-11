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

"""Entry point that runs ONE embodied rollout step to capture the intermediate
messages exchanged between the env / rollout / actor components.

It mirrors ``train_embodied_agent.py`` but swaps the env and rollout workers for
capture subclasses (see ``rlinf.tools.benchmark.message_capture``) and forces a
single, save/eval-free step. Set ``RLINF_BENCH_CAPTURE_DIR`` to the directory
where the captured ``*.schema.txt`` / ``*.schema.json`` / ``*.pt`` files should
be written, e.g.::

    RLINF_BENCH_CAPTURE_DIR=/tmp/bench_msgs \
        bash examples/embodiment/run_capture.sh maniskill_ppo_openvla
"""

import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.tools.benchmark.message_capture import (
    CaptureEnvWorker,
    CaptureRolloutWorker,
    capture_enabled,
)
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvla"
)
def main(cfg) -> None:
    if not capture_enabled():
        raise RuntimeError(
            "RLINF_BENCH_CAPTURE_DIR is not set. Point it at an output directory "
            "so captured messages can be written, e.g. "
            "RLINF_BENCH_CAPTURE_DIR=/tmp/bench_msgs."
        )

    cfg = validate_cfg(cfg)

    # Force a single, side-effect-free capture step.
    with open_dict(cfg):
        cfg.runner.max_steps = 1
        cfg.runner.val_check_interval = -1
        cfg.runner.save_interval = -1
        cfg.runner.only_eval = False

    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    print(
        f"[bench-capture] capturing messages to {os.environ['RLINF_BENCH_CAPTURE_DIR']}",
        flush=True,
    )

    if cfg.algorithm.loss_type != "actor_critic":
        raise NotImplementedError(
            "benchmark_capture currently targets the default PPO actor "
            f"(loss_type=actor_critic); got loss_type={cfg.algorithm.loss_type}. "
            "The actor is not hooked for capture, so other actor variants only "
            "need to exist as the trajectory consumer -- extend here if needed."
        )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Actor is not captured; it just needs to consume the trajectories so the
    # rollout loop runs to completion. Use the real (unmodified) actor.
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=component_placement.get_strategy("actor"),
    )

    rollout_group = CaptureRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )

    env_group = CaptureEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=None,
    )

    runner.init_workers()
    runner.run()

    print(
        f"[bench-capture] done. Inspect {os.environ['RLINF_BENCH_CAPTURE_DIR']}/*.schema.txt",
        flush=True,
    )


if __name__ == "__main__":
    main()
