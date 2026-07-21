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

"""Build an offline (Ray-free) Cluster for auto-placement dry runs.

The auto-placement search only reads a handful of scalars off the cluster
(``num_accelerators``) and off the component placement (``world_size`` /
``dp_size`` / component list), all of which are derivable from the config and a
single "total GPU count". A real :class:`~rlinf.scheduler.Cluster`, however,
calls ``ray.init()`` and probes live nodes for hardware, so the tool cannot run
without a live Ray cluster and real GPUs.

:func:`build_offline_cluster` fabricates a homogeneous cluster (``num_nodes``
nodes, each with ``gpus_per_node`` accelerators) and bypasses ``Cluster.__init__``
entirely, so the real ``ComponentPlacement`` parsing can run offline and reuse
the official rank-resolution logic.
"""

from rlinf.scheduler import Cluster
from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo
from rlinf.scheduler.hardware import Accelerator, HardwareInfo, HardwareResource


def build_offline_cluster(
    num_nodes: int,
    gpus_per_node: int,
    accelerator_model: str = "H100",
) -> Cluster:
    """Construct a Cluster without initializing Ray, for dry-run placement search.

    The returned cluster is homogeneous: every node has the same number of
    accelerators of the same model. This matches the auto-placement algorithm's
    assumption that GPUs form a single flat pool (no node/topology awareness).

    Limitations (dry-run only):
    - Heterogeneous ``node_groups`` are not modeled; a config placing a component
      on a non-accelerator hardware group (e.g. ``node_group: robot``) is not
      supported.
    - ``gpus_per_node`` must match the intended hardware; a wrong value yields a
      wrong total GPU count.

    Args:
        num_nodes: Number of nodes in the fabricated cluster.
        gpus_per_node: Number of accelerators per node.
        accelerator_model: Accelerator model string (only affects reporting).

    Returns:
        Cluster: A Cluster instance whose ``num_accelerators`` and
        ``get_node_group`` work offline, ready to feed to ``ComponentPlacement``.
    """
    # Bypass Cluster.__init__ (which calls ray.init) via the singleton __new__.
    cluster = Cluster.__new__(Cluster)

    nodes: list[NodeInfo] = []
    for node_rank in range(num_nodes):
        hardware_resources = [
            HardwareResource(
                type=Accelerator.HW_TYPE,
                infos=[
                    HardwareInfo(type=Accelerator.HW_TYPE, model=accelerator_model)
                    for _ in range(gpus_per_node)
                ],
            )
        ]
        nodes.append(
            NodeInfo(
                node_labels=[NodeGroupInfo.DEFAULT_GROUP_LABEL],
                node_rank=node_rank,
                ray_id="",
                node_ip="",
                num_cpus=0,
                python_interpreter_path="",
                default_env_vars={},
                env_vars={},
                hardware_resources=hardware_resources,
            )
        )

    # NodeGroupInfo.__post_init__ auto-sets hardware_type to the default
    # ("Accelerator"), so hardware_resource_count counts GPUs, not nodes.
    node_group = NodeGroupInfo(label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes)

    cluster._num_nodes = num_nodes
    cluster._nodes = nodes
    cluster._node_groups = [node_group]
    cluster._distributed_log_collector = None
    cluster._has_initialized = True

    return cluster
