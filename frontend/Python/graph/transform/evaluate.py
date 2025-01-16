from typing import Dict, List
import logging
import math

from .. import Graph
from ..operation import *
from .. import DeviceType

logging.basicConfig(
    level=logging.DEBUG
)

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
        self._relu_latency = None
        self._relu_energy = 0.52            # mW
        self._max_pooling_latency = None
        self._max_pooling_energy = 0.4      # mW
        self._transpose_latency = None
        self._transpose_energy = None


class PIMAcc(Hardware):
    def __init__(
        self,
        device: DeviceType,
        piminfo: Dict = None
    ) -> None:
        super().__init__(device)
        # Set PIM Acc hardware info.
        if piminfo is not None:
            self._piminfo = piminfo
        else:
            self._piminfo = {
                # 'OCC' default settings
                {'tile_size': 2 * 2},           # nums
                {'tile_rows': 2},               # nums
                {'tile_cols': 2},               # nums
                {'xbar_size': 64 * 64},         # nums
                {'xbar_rows': 64},              # nums
                {'xbar_cols': 64},              # nums
                {'precision': 8},               # 8-bit per cell
                {'compute_latency': 1.0},       # us/8-bit
                {'write_latency': 2.5},         # us/8-bit
                {'compute_energy': 200.0},      # fJ/8-bit
                {'read_energy': 200.0},         # fJ/8-bit
                {'write_energy': 200000.0},     # fJ/8-bit
                {'endurance': 3.2 * 1e7},       # times
                {'circuit_energy': 3.9 * 1e6},  # fJ @ 1.2GHz
                {'input_buffer_energy': 5400},  # fJ/byte @ 1.5KB
                {'output_buffer_energy': 5400}, # fJ/byte @ 1.5KB
                {'gevm_energy': 40.0 * 1e3},    # fJ/GEVM for weighted sum
                {'alu_energy': 2.11 * 1e3},     # fJ/ALU Operation
                {'control_energy': 0.78 * 1e6}, # fJ
                # other settings
                {'load_latency': 0.1},          # us
                {'store_latency': 0.1},         # us
                {'adc_latency': 6.25},          # us
                {'adc_power': 16.0 / 8},        # mW
                {'dac_latency': 1.0},           # us
                {'dac_power': 4.0 / (8 * 128)}, # mW
                {'sa_latency': 0.0},            # us
                {'sa_power': 0.2},              # mW
                {'sh_latency': 0.0},            # us
                {'sh_power': 0.0055},           # mW
                {'transport_bw': 6.4},          # GB/s
                {'transport_power': 10400},     # mW
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
    
    def im2col(input: List, kernel: List)-> float:
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
        rows = self._piminfo['tile_rows'] * self._piminfo['xbar_rows']
        cols = self._piminfo['tile_cols'] * self._piminfo['xbar_cols'] * self._piminfo['precision'] / data_precision
        weight_mapping_times = min(ceil(K, rows) * ceil(N, cols), ceil(N, rows) * ceil(K, cols))
        input_comp_times = min(ceil(M, rows) * K, ceil(K, rows) * M)
        # 1. for each mapping, we need write weight to PIM Acc.
        mapping_latency = self._piminfo['write_latency'] * rows * cols * data_precision / 8
        # 2. for each computing, we need go through DAC, Gevm(read), S&H, ADC, S&A, store
        # TODO: 可能 compute latency 已经包含了其他单元的时延，需要 double check.
        computing_latency = self._piminfo['dac_latency'] + self._piminfo['compute_latency'] * rows * cols * data_precision / 8 +\
                            self._piminfo['sh_latency'] + self._piminfo['adc_latency'] + self._piminfo['sa_latency'] +\
                            self._piminfo['store_latency']
        latency = mapping_latency * weight_mapping_times + computing_latency + input_comp_times
        
        return latency
    
    def memcpy(
        self,
        data_bits: int
    )-> float:
        bandwidth = self._piminfo['transport_bw'] / 1024 / 1024 / 8 # GB/s -> bit/s
        latency = 1000 * 1000 * data_bits / bandwidth # us
        
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
            latency = latency + self._transpose_latency
        elif isinstance(op, MatmulOp) or isinstance(op, AddMMOp):
            # -> tosa.matmul and tosa.add
            input_mat, mat1, mat2 = inputs[0], inputs[1], inputs[2]
            M, K, N = mat1[0], mat1[1], mat2[1]
            latency = latency + self.mapping_and_calc(M, K, N)
        elif isinstance(op, BatchMatmulOp):
            # -> tosa.matmul
            mat1, mat2 = inputs[0], inputs[1]
            M, K, N = mat1[0], mat1[1], mat2[1]
            latency = latency + self.mapping_and_calc(M, K, N)
        elif isinstance(op, Conv2dOp):
            # -> tosa.conv
            input, kernel, bias = inputs[0], inputs[1], inputs[2]
            assert len(input) == 4 and len(kernel) == 4
            inp, krnl = self.im2col(input, kernel)
            M, K, N = inp[0], inp[1], inp[2]
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

