# ===- fuse_ops.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Construct op fusion pattern.
#
# ===---------------------------------------------------------------------------

from colorama import Fore

from .. import Graph
from ..operation import *
from .. import DeviceType

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = []


def set_subgraph(
    subgraph: list,
    graph: Graph, 
    subgraph_name: str, 
    device_type: DeviceType
):
    subgraph_name = subgraph_name + "-" + device_type.value
    graph.group_map_device[subgraph_name] = device_type
    graph.op_groups[subgraph_name] = subgraph


def simply_fuse(graph: Graph):
    """
    Function to fuse all operations into one graph. Set the device type to CPU.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def heter_fuse_lenet(graph: Graph):
    """
    Function to fuse operations on heter device for Lenet.
    Set the device type to Heter.

    Args:
    - graph (Graph): graph (Graph): The input graph to be simplified.
    
    Returns:
    - None: Modifies the input graph in place.
    """
    group = []
    device = DeviceType.HETER
    for i, op in graph.body:
        if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp) or i == 25:
            continue
        group.append(op)
    new_op_groups = [graph._body[25]]
    set_subgraph(group, graph, "subgraph0", DeviceType.CPU)
    set_subgraph(new_op_groups, graph, "subgraph1", DeviceType.GPU)


# TODO: Evaluate

def pim_fuse(
    graph: Graph,
    model_name: str
):
    """
    Function to fuse operations for PIM Accelerator.
    Set the device type to PIMAcc.

    Args:
    - graph (Graph): graph (Graph): The input graph to be simplified.
    
    Returns:
    - None: Modifies the input graph in place.
    """
    
    def pim_basic_fuse(
        graph: Graph
    ):
        subgraph_idx = -1
        subgraph_host_idxs, subgraph_device_idxs = [], []
        group_host, group_device = [], []
        device = DeviceType.CPU
        for i, op in enumerate(graph.body):
            # skip PlaceholderOp and OutputOp
            if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
                continue
            # offload to PIM Acc
            if isinstance(op, MatmulOp) \
            or isinstance(op, AddMMOp) \
            or isinstance(op, BatchMatmulOp) \
            or isinstance(op, Conv2dOp) \
            or isinstance(op, ReluOp) \
            or isinstance(op, MaxPool2dOp) \
            or isinstance(op, TransposeOp):
                # TODO: offload according to cost function
                device = DeviceType.PIM
                subgraph_idx = subgraph_idx + 1
                group_device_tmp = [op]
                group_device.append(group_device_tmp)
                subgraph_device_idxs.append(subgraph_idx)
                continue
            # offload to CPU
            if device == DeviceType.CPU and len(group_host) != 0:
                group_host[-1].append(op)
            else:
                device = DeviceType.CPU
                subgraph_idx = subgraph_idx + 1
                group_host_tmp = [op]
                group_host.append(group_host_tmp)
                subgraph_host_idxs.append(subgraph_idx)
        return group_host, subgraph_host_idxs, group_device, subgraph_device_idxs


    def pim_greedy_fuse(
        graph: Graph
    ):
        subgraph_idx = -1
        subgraph_host_idxs, subgraph_device_idxs = [], []
        group_host, group_device = [], []
        device = DeviceType.CPU
        for i, op in enumerate(graph.body):
            # skip PlaceholderOp and OutputOp
            if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
                continue
            # offload to PIM Acc
            if isinstance(op, MatmulOp) \
            or isinstance(op, AddMMOp) \
            or isinstance(op, BatchMatmulOp) \
            or isinstance(op, Conv2dOp) \
            or isinstance(op, ReluOp) \
            or isinstance(op, MaxPool2dOp) \
            or isinstance(op, TransposeOp):
                # TODO: offload according to cost function
                if device == DeviceType.PIM and len(group_device) != 0:
                    group_device[-1].append(op)
                else:
                    device = DeviceType.PIM
                    subgraph_idx = subgraph_idx + 1
                    group_device_tmp = [op]
                    group_device.append(group_device_tmp)
                    subgraph_device_idxs.append(subgraph_idx)
                continue
            # offload to CPU
            if device == DeviceType.CPU and len(group_host) != 0:
                group_host[-1].append(op)
            else:
                device = DeviceType.CPU
                subgraph_idx = subgraph_idx + 1
                group_host_tmp = [op]
                group_host.append(group_host_tmp)
                subgraph_host_idxs.append(subgraph_idx)
        return group_host, subgraph_host_idxs, group_device, subgraph_device_idxs
    
    fuse_func = pim_greedy_fuse

    subgraph_prefix = "subgraph-"
    host, acc = DeviceType.CPU, DeviceType.PIM
    group_host, subgraph_host_idxs, group_device, subgraph_device_idxs = fuse_func(graph)
    # subgraph in host
    print(Fore.GREEN + "Subgraphs offloaded to CPU." + Fore.RESET)
    for i, subgraph in enumerate(group_host):
        dict = {subgraph_host_idxs[i] : subgraph}
        print(dict)
        set_subgraph(subgraph, graph,
                     subgraph_prefix + str(subgraph_host_idxs[i]),
                     host)
    # subgraph in acc
    print(Fore.GREEN + 'Subgraphs offloaded to PIM Acc.' + Fore.RESET)
    for i, subgraph in enumerate(group_device):
        dict = {subgraph_device_idxs[i] : subgraph}
        print(dict)
        set_subgraph(subgraph, graph,
                     subgraph_prefix + str(subgraph_device_idxs[i]),
                     acc)
    print(Fore.GREEN + 'There are ' + str(len(group_host)) + ' subgraphs offloaded to CPU.' + Fore.RESET)
    print(Fore.GREEN + 'There are ' + str(len(group_device)) + ' subgraphs offloaded to PIM Acc.' + Fore.RESET)
   
    # summary of memcpy and compute latency info
    from .evaluate import evaluate_graph
    evaluate_graph(graph, model_name)
