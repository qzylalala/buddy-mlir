//===----------------------------- pim.hpp --------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// PIM Operations(functions)
//
//===----------------------------------------------------------------------===//

#ifndef _MLIR_INTERFACE__PIM__PIM_HPP_
#define _MLIR_INTERFACE__PIM__PIM_HPP_

#include "mlir_interface/core/container.hpp"
#include "utility/utility.hpp"

#include <cstdint>
#include <stdlib.h>

extern "C" {

// MLIR pim.flush
void _mlir_ciface_pim_flush();

// MLIR pim.barrier
void _mlir_ciface_pim_barrier(int32_t tile_id);

// MLIR pim.gevm

// MLIR pim.matmul
void _mlir_ciface_pim_matmul_i8(MemRef<int8_t, 2> *A, 
                                MemRef<int8_t, 2> *B,
                                MemRef<int8_t, 2> *C);

void _mlir_ciface_pim_matmul_i16(MemRef<int16_t, 2> *A,
                                MemRef<int16_t, 2> *B,
                                MemRef<int16_t, 2> *C);

// void _mlir_ciface_pim_matmul_f16(MemRef<_Float16, 2> *A, 
//                                 MemRef<_Float16, 2> *B,
//                                 MemRef<_Float16, 2> *C);

void _mlir_ciface_pim_matmul_i32(MemRef<int32_t, 2> *A, 
                                MemRef<int32_t, 2> *B,
                                MemRef<int32_t, 2> *C);

void _mlir_ciface_pim_matmul_f32(MemRef<_Float32, 2> *A, 
                                MemRef<_Float32, 2> *B,
                                MemRef<_Float32, 2> *C);

// MLIR pim.conv
void _mlir_ciface_pim_conv_i8(MemRef<int8_t, 4> *input,
                            MemRef<int8_t, 2> *kernel,
                            MemRef<int8_t, 1> *bias,
                            MemRef<int8_t, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim);

void _mlir_ciface_pim_conv_i16(MemRef<int16_t, 4> *input,
                            MemRef<int16_t, 2> *kernel,
                            MemRef<int16_t, 1> *bias,
                            MemRef<int16_t, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim);

// void _mlir_ciface_pim_conv_f16(MemRef<_Float16, 4> *input,
//                             MemRef<_Float16, 4> *kernel);

void _mlir_ciface_pim_conv_i32(MemRef<int32_t, 4> *input,
                            MemRef<int32_t, 2> *kernel,
                            MemRef<int32_t, 1> *bias,
                            MemRef<int32_t, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim);

void _mlir_ciface_pim_conv_f32(MemRef<_Float32, 4> *input,
                            MemRef<_Float32, 2> *kernel,
                            MemRef<_Float32, 1> *bias,
                            MemRef<_Float32, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim);

// MLIR pim.load


// MLIR pim.store


// MLIR pim.send


// MLIR pim.receive


// MLIR pim.memcpy_host_to_device
void _mlir_ciface_pim_memcpy_host_to_device_i8(
    MemRef<int8_t, 2>* input, int64_t addr);

void _mlir_ciface_pim_memcpy_host_to_device_i16(
    MemRef<int16_t, 2>* input, int64_t addr);

void _mlir_ciface_pim_memcpy_host_to_device_i32(
    MemRef<int32_t, 2>* input, int64_t addr);

void _mlir_ciface_pim_memcpy_host_to_device_f32(
    MemRef<_Float32, 2>* input, int64_t addr);

// MLIR pim.memcpy_device_to_host
void _mlir_ciface_pim_memcpy_device_to_host_i8(
    MemRef<int8_t, 2>* output, int64_t addr);

void _mlir_ciface_pim_memcpy_device_to_host_i16(
    MemRef<int16_t, 2>* output, int64_t addr);

void _mlir_ciface_pim_memcpy_device_to_host_i32(
    MemRef<int32_t, 2>* output, int64_t addr);

void _mlir_ciface_pim_memcpy_device_to_host_f32(
    MemRef<_Float32, 2>* output, int64_t addr);

} /* extern C */

#endif /* _MLIR_INTERFACE__PIM__PIM_HPP_ */
