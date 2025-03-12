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
    model_name: str,
    device_type: DeviceType = DeviceType.PIM
):
    global acc
    # this function can be used by all Accelerators, we mainly focus on PIM Acc.
    if device_type is DeviceType.PIM:
        acc = PIMAcc(DeviceType.PIM)
    host = CPU(DeviceType.CPU)
    compute_latency = 0.
    compute_latencys = []
    host_latencys = []
    for subgraph_name in graph.op_groups.keys():
        subgraph_compute_latency = acc._piminfo['kern_launch_latency']
        subgraph_host_compute_latency = 0.
        device_type = subgraph_name.split('-')[2]
        if device_type != "pim":
            continue
        for op in graph.op_groups[subgraph_name]:
            # get op input
            inputs = []
            for arg in op._arguments:
                if type(arg) is str:
                    inputs.append(list(graph.node_table[arg]._tensor_meta['shape']))
            # evaluate this op on Acc
            latency = acc.evaluate(op, inputs)
            # evaluate this op on CPU
            host_latency = host.evaluate(model_name, op, inputs)
            # calculate compute latency
            subgraph_compute_latency = subgraph_compute_latency + latency
            subgraph_host_compute_latency = subgraph_host_compute_latency + host_latency
        compute_latencys.append(subgraph_compute_latency)
        host_latencys.append(subgraph_host_compute_latency)
    for latency in compute_latencys:
        compute_latency = compute_latency + latency
    print(Fore.GREEN + "    - Total compute latency of subgraphs offloaded to Acc is {} us".format(compute_latency) + Fore.RESET)
    print(Fore.GREEN + "    - Total xbar write time is {}/{}".format(acc.write_times, acc._piminfo['endurance']) + Fore.RESET)
    return compute_latencys, host_latencys


def get_cpu_whole_latency(
    model_name: str
):
    host = CPU(DeviceType.CPU)
    if model_name in host._graph_cache.keys():
        return host._graph_cache[model_name]
    else:
        return 0.


def get_subgraph_names(
    graph: Graph
):
    subgraph_names = []
    for subgraph_name in graph.op_groups.keys():
        device_type = subgraph_name.split('-')[2]
        if device_type != "pim":
            continue
        subgraph_names.append(subgraph_name)
    
    return subgraph_names


def double_buffer(
    memcpy_h2d_latency,
    acc_compute_latency,
    memcpy_d2h_latency,
    device_type: DeviceType = DeviceType.PIM
):
    global acc
    # this function can be used by all Accelerators, we mainly focus on PIM Acc.
    if device_type is DeviceType.PIM:
        acc = PIMAcc(DeviceType.PIM)
    latency = 0.
    half_scratchpad = acc._piminfo['scratchpad_capicity'] * 1024 / 2 # bytes
    half_scratchpad_memcpy_time = acc.memcpy(half_scratchpad * 8, 0) # us

    # When we use double buffer, we should make sure the data transferred should be less than half of scratchpad.
    if memcpy_h2d_latency > half_scratchpad_memcpy_time:
        split_times = math.floor(memcpy_h2d_latency / half_scratchpad_memcpy_time)
        compute_latency = acc_compute_latency / split_times
        last_time_memcpy_latency = memcpy_h2d_latency - half_scratchpad_memcpy_time * split_times
        last_time_compute_latency = acc_compute_latency - compute_latency * split_times
        latency = latency + half_scratchpad_memcpy_time + (split_times - 2) * max(half_scratchpad_memcpy_time, compute_latency) + max(last_time_memcpy_latency, compute_latency) + last_time_compute_latency + memcpy_d2h_latency
    else:
        latency = latency + memcpy_h2d_latency + acc_compute_latency + memcpy_d2h_latency
    
    return latency
    


def evaluate_graph(
    graph: Graph,
    model_name: str,
):
    memcpy_h2d_list, memcpy_d2h_list = calc_memcpy(graph)
    acc_compute_latency_list, host_compute_latency_list = calc_computation(graph, model_name)
    assert len(memcpy_h2d_list) == len(acc_compute_latency_list), "Number of subgraphs need to be same."
    
    # W/O double buffer
    latency = 0.
    for i in range(len(memcpy_h2d_list)):
        # print("h2d : {}, compute : {}, d2h : {}".format(memcpy_h2d_list[i], acc_compute_latency_list[i], memcpy_d2h_list[i]))
        latency = latency + memcpy_h2d_list[i] + acc_compute_latency_list[i] + memcpy_d2h_list[i]
    print(Fore.GREEN + "W/O double buffer, total latency of subgraphs offloaded to Acc is {} us".format(latency) + Fore.RESET)
    
    # With double buffer
    # memcpy -> memcpy -> memcpy
    #       compute   ->  compute  -> compute
    total_latency = 0.
    for i in range(0, len(memcpy_h2d_list)):
        total_latency = total_latency + double_buffer(memcpy_h2d_list[i], acc_compute_latency_list[i], memcpy_d2h_list[i])
    print(Fore.GREEN + "With double buffer, total latency of subgraphs offloaded to Acc is {} us".format(total_latency) + Fore.RESET)
    
    # With Offload Strategy to decide whether this subgraph should be offloaded to PIM Acc.
    subgraph_names = get_subgraph_names(graph)
    reduced_subgraphs = 0
    remain_memcpy_h2d_list, remain_acc_compute_list, remain_memcpy_d2h_list = [], [], []
    cpu_whole_latency = get_cpu_whole_latency(model_name)
    cpu_subgraphs_latency = 0.
    cpu_extra_latency = 0.
    print(Fore.GREEN + "We compare the latency of subgraphs offloaded to PIM Acc with latency on CPU." + Fore.RESET)
    for i in range(len(memcpy_h2d_list)):
        acc_latency = memcpy_h2d_list[i] + acc_compute_latency_list[i] + memcpy_d2h_list[i]
        host_latency = host_compute_latency_list[i]
        cpu_subgraphs_latency = cpu_subgraphs_latency + host_latency
        # print("     - {}, host latency : {}, acc latency : {}".format(subgraph_names[i], host_latency, acc_latency))
        if host_latency < acc_latency:
            reduced_subgraphs = reduced_subgraphs + 1
            cpu_extra_latency = cpu_extra_latency + host_latency
            # print(Fore.GREEN + "    Subgraph {} should not be offloaded to PIM Acc, host latency : {} us, PIM Acc latency : {} us".format(subgraph_names[i], host_latency, acc_latency) + Fore.RESET)
        else:
            remain_memcpy_h2d_list.append(memcpy_h2d_list[i])
            remain_acc_compute_list.append(acc_compute_latency_list[i])
            remain_memcpy_d2h_list.append(memcpy_d2h_list[i])
    # TODO: update the latency
    acc_latency = 0.
    for i in range(0, len(remain_memcpy_h2d_list)):
        acc_latency = acc_latency + double_buffer(remain_memcpy_h2d_list[i], remain_acc_compute_list[i], remain_memcpy_d2h_list[i])
    print(Fore.GREEN + "1. The total latency of all subgraphs offloaded to CPU is {} us".format(cpu_whole_latency) + Fore.RESET)
    print(Fore.GREEN + "2. The total latency of graphs W/O double buffer is {} us".format(cpu_whole_latency - cpu_subgraphs_latency + latency) + Fore.RESET)
    print(Fore.GREEN + "3. The total latency of graphs With double buffer is {} us".format(cpu_whole_latency - cpu_subgraphs_latency + total_latency) + Fore.RESET)
    print(Fore.GREEN + "    - The total latency of subgraphs offloaded to CPU is {} us".format(cpu_whole_latency - cpu_subgraphs_latency) + Fore.RESET)
    print(Fore.GREEN + "    - The total latency of subgraphs offloaded to PIMAcc is {} us".format(total_latency) + Fore.RESET)
    print(Fore.GREEN + "We will remove {} subgraphs, and benefit from this stategy with {} us".format(reduced_subgraphs, cpu_extra_latency + total_latency - acc_latency) + Fore.RESET)
    print(Fore.GREEN + "4. Finally, the total latency of these graph is {} us".format(cpu_whole_latency - cpu_subgraphs_latency + cpu_extra_latency + acc_latency) + Fore.RESET)
    print(Fore.GREEN + "    - The total latency of subgraphs offloaded to PIMAcc is {} us".format(acc_latency) + Fore.RESET)


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
                'tile_nums': 168,             # nums
                'tile_rows': 14,              # nums
                'tile_cols': 12,              # nums
                'ima_nums' : 12,              # nums
                'ima_rows' : 3,               # nums
                'ima_cols' : 4,               # nums
                'xbar_nums': 8,               # nums
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
                'scratchpad_capicity': 256,  # KB
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
        data_precision: int = 32
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
        device: DeviceType = DeviceType.CPU
    ) -> None:
        super().__init__(device)
        # Set cpu info (Single Core).
        # import psutil
        # self._freq = float(psutil.cpu_freq()) # MHz
        # self._latency_per_cycle = 1.0 / self._freq # us
        # this hub provided by pytorch profiling with AMD EPYC 9554 64-Core Processor.
        # we only consider operations which can be executed on both CPU and device.
        # Conv2d, Addmm, Matmul, BatchMatmul
        self.op_cnt = 0
        self._graph_cache = {
            "fc3": 24.4140625,
            'lenet5': 721.502304077148,
            'resnet18': 10733.413696289,
            'resnet34': 23912.3582839965,
            'resnet50': 32108.2353591918,
            'resnet101': 52931.785583496,
        }
        self._hub = {
            "fc3": [16.826171875, 1.46484375],
            'lenet5': [393.43520641326904, 150.36108016967773, 53.9683723449707, 7.647924423217773, 4.47331428527832],
            'resnet18': [590.3377532958984, 243.64849090576172, 199.64149475097656, 206.08154296875, 213.59493255615234, 238.28178405761722, 254.38190460205078, 262.9686355590821, 250.08853912353516, 298.3889007568359, 268.33534240722656, 297.31555938720703, 193.20144653320312, 272.6287078857422, 272.6287078857422, 287.6554870605469, 479.7835922241211, 180.32135009765625, 463.6834716796875, 476.56356811523443, 2735.947151184082],
            'resnet34': [731.7181634902954, 650.4161453247071, 758.0217576026917, 473.4646940231323, 432.81368494033813, 339.55548763275146, 750.8480501174927, 373.0327892303467, 315.6431293487549, 246.2972903251648, 301.29571437835693, 308.4694218635559, 358.68537425994873, 423.2487416267395, 356.2941384315491, 356.2941384315491, 339.55548763275146, 466.29098653793335, 270.2096486091613, 483.02963733673096, 475.855929851532, 514.1157031059265, 540.4192972183226, 590.6352496147157, 609.7651362419127, 645.6336736679078, 576.2878346443176, 576.2878346443176, 573.896598815918, 437.59615659713745, 669.5460319519043, 322.8168368339539, 753.2392859458923, 772.3691725730896, 765.1954650878906, 753.2392859458923, 263.03594112396246],
            'resnet50': [459.2049837112427, 335.4192924499512, 395.31559467315674, 371.3570737838745, 367.36398696899414, 359.3778133392334, 387.329421043396, 375.3501605987549, 343.4054660797119, 379.34324741363525, 383.3363342285156, 491.14967823028564, 507.12202548980713, 347.3985528945923, 2819.119291305542, 511.1151123046875, 507.12202548980713, 363.37090015411377, 339.41237926483154, 559.032154083252, 311.46077156066895, 323.4400320053101, 539.0667200088501, 295.48842430114746, 487.1565914154053, 694.7971057891846, 355.384726524353, 2415.8175230026245, 395.31559467315674, 646.8800640106202, 367.36398696899414, 515.1081991195679, 686.8109321594238, 535.0736331939697, 347.3985528945923, 662.8524112701416, 339.41237926483154, 435.24646282196045, 682.8178453445435, 363.37090015411377, 551.0459804534911, 447.2257232666016, 511.1151123046875, 503.12893867492676, 818.5827970504761, 387.329421043396, 1050.1818323135376, 455.21189689636225, 694.7971057891846, 431.2533760070801, 467.1911573410034, 686.8109321594238, 427.2602891921997, 355.384726524353],
            'resnet101': [391.20211601257324, 404.2421865463257, 326.00176334381104, 326.00176334381104, 319.4817280769348, 443.362398147583, 326.00176334381104, 371.6420102119446, 397.72215127944946, 319.4817280769348, 358.60193967819214, 475.9625744819641, 410.7622218132019, 502.042715549469, 1760.4095220565796, 508.5627508163452, 365.12197494506836, 462.92250394821167, 449.88243341445923, 358.60193967819214, 456.40246868133545, 417.2822570800781, 371.6420102119446, 443.362398147583, 352.0819044113159, 462.92250394821167, 352.0819044113159, 1343.1272649765015, 365.12197494506836, 443.362398147583, 339.0418338775635, 345.5618691444397, 515.0827860832214, 339.0418338775635, 312.9616928100586, 384.682080745697, 339.0418338775635, 332.52179861068726, 417.2822570800781, 312.9616928100586, 339.0418338775635, 423.80229234695435, 319.4817280769348, 319.4817280769348, 462.92250394821167, 345.5618691444397, 339.0418338775635, 417.2822570800781, 339.0418338775635, 332.52179861068726, 443.362398147583, 345.5618691444397, 326.00176334381104, 462.92250394821167, 352.0819044113159, 358.60193967819214, 417.2822570800781, 319.4817280769348, 332.52179861068726, 436.8423628807068, 358.60193967819214, 358.60193967819214, 449.88243341445923, 332.52179861068726, 319.4817280769348, 430.32232761383057, 345.5618691444397, 339.0418338775635, 443.362398147583, 352.0819044113159, 462.92250394821167, 704.1638088226318, 547.6829624176025, 593.3232092857361, 515.0827860832214, 339.0418338775635, 345.5618691444397, 560.723032951355, 391.20211601257324, 612.8833150863647, 710.683844089508, 665.0435972213745, 365.12197494506836, 443.362398147583, 345.5618691444397, 371.6420102119446, 710.683844089508, 456.40246868133545, 730.2439498901367, 678.083667755127, 612.8833150863647, 371.6420102119446, 652.0035266876221, 482.48260974884033, 665.0435972213745, 1108.4059953689575, 508.5627508163452, 573.7631034851074, 443.362398147583, 847.6045846939087, 371.6420102119446, 482.48260974884033, 925.8450078964233, 449.88243341445923, 391.20211601257324]
        }
    
    def evaluate(
        self,
        model_name: str,
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
            if model_name in self._hub.keys():
                latency = self._hub[model_name][self.op_cnt]
                self.op_cnt = self.op_cnt + 1
        elif isinstance(op, BatchMatmulOp):
            if model_name in self._hub.keys():
                latency = self._hub[model_name][self.op_cnt]
                self.op_cnt = self.op_cnt + 1
        elif isinstance(op, Conv2dOp):
            if model_name in self._hub.keys():
                latency = self._hub[model_name][self.op_cnt]
                self.op_cnt = self.op_cnt + 1
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

