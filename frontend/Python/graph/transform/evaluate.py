from colorama import Fore
from typing import Dict, List
import logging
import math

from .. import Graph
from ..operation import *
from .. import DeviceType

logging.basicConfig(
    level=logging.INFO
)


def dtype2bytes(dtype: TensorDType):
    factor = 4
    if dtype is TensorDType.Float16:
        factor = 2
    elif dtype is TensorDType.Float32:
        factor = 4
    elif dtype is TensorDType.Float64:
        factor = 8
    elif dtype is TensorDType.Int32:
        factor = 4
    elif dtype is TensorDType.Int64:
        factor = 8
    elif dtype is TensorDType.Bool:
        factor = 1
    else:
        print("node_dtype: " + str(dtype) + " is not supported.")
    return factor


def calc_memcpy(
    graph: Graph,
    device_type: DeviceType = DeviceType.PIM
):    
    global acc
    # this function can be used by all Accelerators, we mainly focus on PIM Acc.
    if device_type is DeviceType.PIM:
        acc = PIMAcc(DeviceType.PIM)
    # We can only pay attention to subgraphs offloaded to Acc.
    # 1. transfer input args from Host to Acc.
    # 2. transfer output from Acc to Host.
    global input_bytes, output_bytes, memcpy_h2d, memcpy_d2h
    input_bytes, output_bytes = 0, 0
    memcpy_h2d, memcpy_d2h = 0., 0.
    
    subgraphs_inputs, subgraphs_outputs = {}, {}
    # Identify inputs for each subgraph
    for subgraph_name in graph.op_groups.keys():
        subgraphs_inputs[subgraph_name] = []
        for op in graph.op_groups[subgraph_name]:
            for parent in op._parents:
                if (
                    graph.node_table[parent]
                    not in graph.op_groups[subgraph_name]
                ):
                    subgraphs_inputs[subgraph_name].append(parent)
    # Identify output nodes of the entire graph
    output_node = []
    for node in graph.body:
        if isinstance(node, OutputOp):
            for arg in node.args:
                output_node.append(arg)
    # Identify outputs for each subgraph
    for subgraph_name in graph.op_groups.keys():
        subgraphs_outputs[subgraph_name] = []
        for op in graph.op_groups[subgraph_name]:
            for key in subgraphs_inputs.keys():
                if op.name in subgraphs_inputs[key]:
                    subgraphs_outputs[subgraph_name].append(op.name)
            if (op.name in output_node) and (
                op.name not in subgraphs_outputs[subgraph_name]
            ):
                subgraphs_outputs[subgraph_name].append(op.name)
    # Calculate latency
    input_databytes, output_databytes = [], []
    for subgraph_name in graph.op_groups.keys():
        device_type = subgraph_name.split('-')[2]
        if device_type != "pim":
            continue
        # print(subgraph_name)
        subgraph_inputs, subgraph_output = subgraphs_inputs[subgraph_name], subgraphs_outputs[subgraph_name]
        # print("input tensor info")
        input_data, output_data = 0, 0
        for inp in subgraph_inputs:
            node = graph.node_table[inp]
            node_shape = node.tensor_meta["shape"]
            node_dtype = node.tensor_meta["dtype"]
            # print(node.tensor_meta)
            factor = dtype2bytes(node_dtype)
            data_bytes = node_shape.numel() * factor
            input_data = input_data + data_bytes
        input_databytes.append(input_data)
        # print("output tensor info")
        for output in subgraph_output:
            node = graph.node_table[output]
            # print(node.tensor_meta)
            node_shape = node._tensor_meta["shape"]
            node_dtype = node._tensor_meta["dtype"]
            factor = dtype2bytes(node_dtype)
            data_bytes = node_shape.numel() * factor
            output_data = output_data + data_bytes
        output_databytes.append(output_data)
    memcpy_h2d_list, memcpy_d2h_list = [], []
    for input_data in input_databytes:
        input_bytes = input_bytes + input_data
        memcpy_latency = acc.memcpy(input_data * 8, 0)
        memcpy_h2d = memcpy_h2d + memcpy_latency
        memcpy_h2d_list.append(memcpy_latency)
    for output_data in output_databytes:
        output_bytes = output_bytes + output_data
        memcpy_latency = acc.memcpy(output_data * 8, 1)
        memcpy_d2h = memcpy_d2h + memcpy_latency
        memcpy_d2h_list.append(memcpy_latency)
    
    print(Fore.GREEN + "    - Total Bytes transferred between Host and Acc: {} bytes, total memcpy latency is {} us.".format(input_bytes + output_bytes, memcpy_h2d + memcpy_d2h) + Fore.RESET)
    print(Fore.GREEN + "        - Bytes transferred from Host to Acc: {} bytes, latency is {} us.".format(input_bytes, memcpy_h2d) + Fore.RESET)
    print(Fore.GREEN + "        - Bytes transferred from Acc to Host: {} bytes, latency is {} us.".format(output_bytes, memcpy_d2h) + Fore.RESET)
    return memcpy_h2d_list, memcpy_d2h_list


def calc_computation(
    graph: Graph,
    device_type: DeviceType = DeviceType.PIM
):
    global acc
    # this function can be used by all Accelerators, we mainly focus on PIM Acc.
    if device_type is DeviceType.PIM:
        acc = PIMAcc(DeviceType.PIM)
    compute_latency = 0.
    compute_latencys = []
    for subgraph_name in graph.op_groups.keys():
        subgraph_compute_latency = acc._piminfo['kern_launch_latency']
        device_type = subgraph_name.split('-')[2]
        if device_type != "pim":
            continue
        for op in graph.op_groups[subgraph_name]:
            # get op input
            inputs = []
            for arg in op._arguments:
                if type(arg) is str:
                    inputs.append(list(graph.node_table[arg]._tensor_meta['shape']))
            latency = acc.evaluate(op, inputs)
            subgraph_compute_latency = subgraph_compute_latency + latency
        compute_latencys.append(subgraph_compute_latency)
    for latency in compute_latencys:
        compute_latency = compute_latency + latency
    print(Fore.GREEN + "    - Total compute latency of subgraphs offloaded to Acc is {} us".format(compute_latency) + Fore.RESET)
    print(Fore.GREEN + "    - Total xbar write time is {}/{}".format(acc.write_times, acc._piminfo['endurance']) + Fore.RESET)
    return compute_latencys


def evaluate_graph(
    graph: Graph
):
    memcpy_h2d_list, memcpy_d2h_list = calc_memcpy(graph)
    compute_latency_list = calc_computation(graph)
    assert len(memcpy_h2d_list) == len(compute_latency_list), "Number of subgraphs need to be same."
    
    # W/O double buffer
    latency = 0.
    for i in range(len(memcpy_h2d_list)):
        latency = latency + memcpy_h2d_list[i] + compute_latency_list[i] + memcpy_d2h_list[i]
    print(Fore.GREEN + "W/O double buffer, total latency of subgraphs offloaded to Acc is {} us".format(latency) + Fore.RESET)
    
    # With double buffer
    # memcpy -> memcpy -> memcpy
    #       compute   ->  compute  -> compute
    latency = memcpy_h2d_list[0] + memcpy_d2h_list[0] + compute_latency_list[-1]
    for i in range(1, len(memcpy_h2d_list)):
        latency = latency + max(memcpy_h2d_list[i], compute_latency_list[i - 1]) + memcpy_d2h_list[i]
    print(Fore.GREEN + "With double buffer, total latency of subgraphs offloaded to Acc is {} us".format(latency) + Fore.RESET)


class Evaluater:
    """
    Evaluater is a tool for evaluating performance.
    
    Attributes:
    - _device: str
        The hardware for Accelerating some ops.
    - _host: str
        The hardware for general computing.
    """
    
    def __init__(
        self,
        graph: Graph,
        device: DeviceType,
        host: DeviceType,
        has_double_buffer: bool = False,
        hub: Dict = None,
    ) -> None:
        self._graph = graph
        self._hub = hub
        self._device = device
        self._host = host
    
    def evaluate(
        self,
        subgraph: List,
    ) -> DeviceType:
        # 1. special situation.
        if self._host == self._device:
            return self._host
        if self._hub.get(subgraph) is not None:
            return self._hub[subgraph]
        # 2. init hardware.
        device, host = None, None
        # init host hardware.
        if self._host is DeviceType.CPU:
            host = CPU(self._host)
        elif self._host is DeviceType.GPU:
            host = GPU(self._host)
        else:
            pass
        # init device hardware.
        if self._device is DeviceType.PIM:
            device = PIMAcc(self._device)
        elif self._device is DeviceType.GPU:
            device = GPU(self._device)
        else:
            pass
        assert (host is not None) and (device is not None), "host or device is not supported."
        logging.info("host is %s, device is %s" % (self._host.value, self._device.value))
        # 3. evaluate operation on device and host, compare the result.
        global host_latency, device_latency
        host_latency, device_latency = 0., 0.
        subgraph_input, subgraph_output = 0, 0
        for i, op in enumerate(subgraph):
            inputs = []
            for arg in op._arguments:
                if type(arg) is str:
                    inputs.append(self._graph.node_table[arg]._tensor_meta['shape'])
            # TODO: record subraph input and output
            host_op_latency = host.evaluate(op, inputs)
            device_op_latency = device.evaluate(op, inputs)
            logging.debug("Operation %s, host latency %lf us, device latency %lf us" % 
                          (op._name, host_op_latency, device_latency))
            host_latency = host_latency + host_op_latency
            device_latency = device_latency + device_op_latency
        # 4. memcpy between host and device.
        #   a. if this subgraph contains more than one operator, it benefits from not writing back to host, otherwise rwite to buffer.
        #   b. if we offload this subgraph not to  host, we need consider memcpy latency between host and device.
        #   c. we can use double buffer to overlap the data transfer latency and compute latency
        memcpy_latency = device.memcpy(subgraph_input) + device.memcpy(subgraph_output)
        logging.debug("host latency: %lf us, device latency: %lf us + memcpy latency: %lf us, device total latency" %\
                        (host_latency, device_latency, memcpy_latency, device_latency + memcpy_latency))
        device_latency = device_latency + memcpy_latency
        return self._device if device_latency < host_latency else self._host


class Hardware:
    """
    Hardware is the platform for computing.
    
    Attributes:
    - _device: str
        The hardware for computing.
    """
    def __init__(
        self,
        device: DeviceType
    ) -> None:
        self._device = device
        self._relu_latency = 0.
        self._relu_energy = 0.52            # mW
        self._max_pooling_latency = 0.
        self._max_pooling_energy = 0.4      # mW
        self._transpose_latency = 0.
        self._transpose_energy = None


class PIMAcc(Hardware):
    def __init__(
        self,
        device: DeviceType = DeviceType.PIM,
        piminfo: Dict = None
    ) -> None:
        super().__init__(device)
        self.write_times = 0
        # Set PIM Acc hardware info.
        if piminfo is not None:
            self._piminfo = piminfo
        else:
            self._piminfo = {
                # default settings
                'tile_size': 4 * 4,           # nums
                'tile_rows': 4,               # nums
                'tile_cols': 4,               # nums
                'ima_size' : 8,               # nums
                'ima_rows' : 2,               # nums
                'ima_cols' : 4,               # nums
                'xbar_size': 128 * 128,       # nums
                'xbar_rows': 128,             # nums
                'xbar_cols': 128,             # nums
                'precision': 2,               # 2-bit per cell
                'compute_latency': 1.8,       # us/32-bit
                'write_latency': 1.0,        # us/32-bit
                'compute_energy': 200.0,     # fJ/8-bit
                'read_energy': 200.0,        # fJ/8-bit
                'write_energy': 200000.0,    # fJ/8-bit
                'endurance': 3.2 * 1e7,      # times
                'circuit_energy': 3.9 * 1e6, # fJ @ 1.2GHz
                'input_buffer_energy': 5400, # fJ/byte @ 1.5KB
                'output_buffer_energy': 5400,# fJ/byte @ 1.5KB
                'gevm_energy': 40.0 * 1e3,   # fJ/GEVM for weighted sum
                'alu_energy': 2.11 * 1e3,    # fJ/ALU Operation
                'control_energy': 0.78 * 1e6,# fJ
                # other settings
                'kern_launch_latency': 4,    # us
                'ddr5_read_latency': 0.08,   # us/64 bytes
                'ddr5_write_latency': 0.08,  # us/64 bytes
                'load_latency': 0.1,         # us
                'store_latency': 0.1,        # us
                'adc_latency': 6.25,         # us
                'adc_power': 16.0 / 8,       # mW
                'dac_latency': 1.0,          # us
                'dac_power': 4.0 / (8 * 128),# mW
                'sa_latency': 0.0,           # us
                'sa_power': 0.2,             # mW
                'sh_latency': 0.0,           # us
                'sh_power': 0.0055,          # mW
                'transport_bw': 27.2,        # GB/s
                'transport_power': 10400,    # mW
            }
        self._hub = {
            'vgg16': {
                'relu': [185.0, 256.0, 237.0, 193.0, 155.0, 148.0, 170.0, 135.0, 147.0, 130.0, 88.0, 82.0, 80.0, 32.0, 12.0],
                'maxpool2d': [425.0, 233.0, 545.0, 321.0, 93.0, 425.0, 233.0, 545.0, 321.0, 93.0, ],
                'transpose': [7.0, 10.0, 3.0],  
            },
            'resnet18': {
                'relu': [95.0, 76.0, 47.0, 69.0, 47.0, 78.0, 45.0, 76.0, 44.0, 78.0, 45.0, 93.0, 45.0, 73.0, 6.0, 31.0, 5.0],
                'maxpool2d': [175.0],
                'transpose': [3],
            }
        }
    
    def im2col(self, input: List, kernel: List)-> float:
        # input     :   [N, H, W, C]
        # kernel    :   [F, H, W, C]
        # output    :   [N, H, W, F]
        channel = input[3]
        input_row = (input[1] - kernel[1] + 1) * (input[2] - input[2] + 1)
        input_col = kernel[1] * kernel[2] * channel
        kernel_row = kernel[1] * kernel[2] * channel
        kernel_col = kernel[0]
        
        return [input_row, input_col], [kernel_row, kernel_col]
    
    def mapping_and_calc(
        self,
        M: int, K: int, N: int,
        data_precision: int = 16
    )-> float:
        from math import ceil
        latency = 0.
        rows = self._piminfo['tile_rows'] * self._piminfo['ima_rows'] *  self._piminfo['xbar_rows']
        cols = self._piminfo['tile_cols'] * self._piminfo['ima_cols']* self._piminfo['xbar_cols'] * self._piminfo['precision'] / data_precision
        weight_mapping_times = min(ceil(K / rows) * ceil(N / cols), ceil(N / rows) * ceil(K / cols))
        self.write_times = self.write_times + weight_mapping_times
        input_comp_times = min(ceil(M / rows) * K, ceil(K / rows) * M)
        # 1. for each mapping, we need write weight to PIM Acc.
        mapping_latency = self._piminfo['write_latency']
        # print("mapping once latency: {}".format(mapping_latency))
        # print("rows : {}, cols : {}, weight_mapping_times : {}, input_comp_times : {}".format(rows, cols, weight_mapping_times, input_comp_times))
        # 2. for each computing, we need go through DAC, Gevm(read), S&H, ADC, S&A, store
        # TODO: 可能 compute latency 已经包含了其他单元的时延，需要 double check.
        computing_latency = self._piminfo['dac_latency'] + \
                            self._piminfo['compute_latency'] + \
                            self._piminfo['sh_latency'] + self._piminfo['adc_latency'] + self._piminfo['sa_latency'] +\
                            self._piminfo['store_latency']
        # computing_latency = self._piminfo['compute_latency'] + self._piminfo['store_latency']
        latency = mapping_latency * weight_mapping_times + computing_latency * input_comp_times * weight_mapping_times
        
        return latency
    
    def memcpy(
        self,
        data_bits: int,
        type: int
    )-> float:
        bandwidth = self._piminfo['transport_bw'] * 1024 * 1024 * 1024 * 8 # GB/s -> bit/s
        latency = 0.
        # transfer
        latency = latency + 1000 * 1000 * data_bits / bandwidth # us
        
        return latency
    
    def evaluate(
        self,
        op: Op,
        inputs: List
    )-> float:
        latency = 0.
        if isinstance(op, ReluOp):
            latency = latency + self._relu_latency
        elif isinstance(op, TransposeOp):
            latency = latency + self._transpose_latency
        elif isinstance(op, MaxPool2dOp):
            latency = latency + self._max_pooling_latency
        elif  isinstance(op, AddMMOp):
            # -> tosa.matmul and tosa.add
            input_mat, mat1, mat2 = inputs[0], inputs[1], inputs[2]
            M, K, N = mat1[0], mat1[1], mat2[1]
            latency = latency + self.mapping_and_calc(M, K, N)
        elif isinstance(op, MatmulOp):
            mat1, mat2 = inputs[0], inputs[1]
            M, K, N = mat1[0], mat1[1], mat2[1]
            latency = latency + self.mapping_and_calc(M, K, N)
        elif isinstance(op, BatchMatmulOp):
            # -> tosa.matmul
            mat1, mat2 = inputs[0], inputs[1]
            M, K, N = mat1[0], mat1[1], mat2[1]
            latency = latency + self.mapping_and_calc(M, K, N)
        elif isinstance(op, Conv2dOp):
            # -> tosa.conv
            input, kernel = inputs[0], inputs[1]
            assert len(input) == 4 and len(kernel) == 4
            inp, krnl = self.im2col(input, kernel)
            M, K, N = inp[0], inp[1], krnl[1]
            latency = latency + self.mapping_and_calc(M, K, N)
        else:
            logging.error("Operation %s is not supported on PIM Acc" % (op._name))
        
        return latency


class CPU(Hardware):
    def __init__(
        self,
        device: DeviceType
    ) -> None:
        super().__init__(device)
        # Set cpu info (Single Core).
        import psutil
        self._freq = psutil.cpu_freq() # MHz
        self._latency_per_cycle = 1.0 / self._freq # us
        # this hub provided by pytorch profiling with AMD EPYC 9554 64-Core Processor.
        # we only consider operations which can be executed on both CPU and device.
        self._hub = {
            'vgg16': {
                'linear': [12093.0, 2830.0, 727.0],
                'conv2d': [2939.0, 2413.0, 3407.0, 4198.0, 1760.0, 3642.0, 1701.0, 1073.0, 2280.0, 1533.0, 1404.0, 1429.0, 1453.0],
                'relu': [185.0, 256.0, 237.0, 193.0, 155.0, 148.0, 170.0, 135.0, 147.0, 130.0, 88.0, 82.0, 80.0, 32.0, 12.0],
                'maxpool2d': [425.0, 233.0, 545.0, 321.0, 93.0, 425.0, 233.0, 545.0, 321.0, 93.0, ],
                'transpose': [7.0, 10.0, 3.0],  
            },
            'resnet18': {
                'linear': [148.0],
                'conv2d': [517.0, 505.0, 411.0, 405.0, 358.0, 464.0, 602.0, 354.0, 693.0, 676.0, 596.0, 698.0, 358.0, 756.0, 748.0, 640.0, 893.0, 381.0, 894.0, 902.0],
                'relu': [95.0, 76.0, 47.0, 69.0, 47.0, 78.0, 45.0, 76.0, 44.0, 78.0, 45.0, 93.0, 45.0, 73.0, 6.0, 31.0, 5.0],
                'maxpool2d': [175.0],
                'transpose': [3],
            }
        }
    
    def evaluate(
        self,
        op: Op,
        inputs: List
    )-> float:
        latency = 0.
        if isinstance(op, ReluOp):
            latency = latency + self._relu_latency
        elif isinstance(op, TransposeOp):
            latency = latency + self._transpose_latency
        elif isinstance(op, MaxPool2dOp):
            latency = latency + self._max_pooling_latency
        elif isinstance(op, MatmulOp) or isinstance(op, AddMMOp):
            pass
        elif isinstance(op, BatchMatmulOp):
            pass
        elif isinstance(op, Conv2dOp):
            pass
        else:
            logging.error("Operation %s is not supported on CPU" % (op._name))
        
        return latency


class GPU(Hardware):
    def __init__(
        self,
        device: DeviceType
    ) -> None:
        super().__init__(device)
        # Set GPU hardware info by reading 'nvidia-smi -q 0'.
        pass
        
    def evaluate(
        self,
        op: Op,
        inputs: List
    )-> float:
        pass

