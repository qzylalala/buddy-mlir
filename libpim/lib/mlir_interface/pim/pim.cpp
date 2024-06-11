//===----------------------------- pim.cpp --------------------------------===//
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

#include "mlir_interface/pim/pim.hpp"
#include "utility/utility.hpp"

#include <stdlib.h>
#include <string.h>
#include <unordered_map>

#define TEXT_SET_COLOR "\033[0;32m"
#define TEXT_RESET_COLOR "\033[0m"

/* --------------- Architecture config ----------------- */
#define TILE_ROWS 4
#define TILE_COLS 4
#define PE_ROWS 4
#define PE_COLS 4
#define XBAR_ROWS 4
#define XBAR_COLS 4

#define NUM_PIM_TILES (TILE_ROWS * TILE_COLS)
#define NUM_PIM_PES (PE_ROWS * PE_COLS)
#define NUM_PIM_XBARs (XBAR_ROWS * XBAR_COLS)

#define XBAR_PRECISION uint32_t
#define NUM_PIM_BITS 2
// 按照 xbar 为最小单位来分配, 再证明当矩阵较小时, 应该 offload 到 cpu 上, xbar 的利用率不会低于多少
#define DIM (128 * NUM_PIM_BITS / sizeof(XBAR_PRECISION))
#define ADDR_LEN 128

#define RELU 1
#define Tanh 2
#define Sigmoid 3

/**
 * TODO:
 * 
 * 1. Pre-Mapping : record weight matrix location
 * ...
*/


template <typename elementType>
static void pim_gevm_helper(MemRef<elementType, 1> *A,
                            MemRef<elementType, 2> *B,
                            MemRef<elementType, 1> *C) {
  printf("[PIM Info] GEVM started...\n");

  // Gevm : C[N] = A[K] * B[K][N]
  const uint32_t K = B->getSizes()[0];
  const uint32_t N = B->getSizes()[1];

  for (uint32_t n = 0; n < N; n ++) {
    elementType sum = 0;

    for (uint32_t k = 0; k < K; k ++) {
      sum += (*A)[k] * (*B)[k * N + n];
    }

    (*C)[n] = sum;
  }
}

/**
 * Matmul Operation on PIM
 * 
 * 
 * 
*/
template <typename elementType>
static void pim_matmul_helper(MemRef<elementType, 2> *A, 
                              MemRef<elementType, 2> *B,
                              MemRef<elementType, 2> *C) {
  printf("[PIM Info] Matmul started...\n");

  // Matmul : C[M][N] =  A[M][K] * B[K][N]
  const uint32_t M = C->getSizes()[0];
  const uint32_t N = C->getSizes()[1];
  const uint32_t K = A->getSizes()[1];

  for (uint32_t m = 0; m < M; m ++) {
    for (uint32_t n = 0; n < N; n ++) {
      elementType sum = 0;

      for (uint32_t k = 0; k < K; k ++) {
        sum += (*A)[m * K + k] * (*B)[k * N + n];
      }

      (*C)[m * N + n] = sum;
    }
  }

  // Activation function
}

template <typename elementType>
static void conv_without_pool(
  uint32_t batchSize, uint32_t inRowDim, uint32_t inColDim,
  uint32_t inChannels, uint32_t outChannels, uint32_t outRowDim,
  uint32_t outColDim, uint32_t stride, uint32_t inDilation,
  uint32_t kernelDilation, uint32_t padding, uint32_t kernelDim,
  uint32_t inStride, uint32_t kernelStride, uint32_t outStride,
  const elementType* input, const elementType* kernel,
  const elementType* bias, elementType* output,
  uint32_t act) {

  bool hasBias = bias == NULL;

  for (uint32_t b = 0; b < batchSize; b ++) {
    for (uint32_t row = 0; row < outRowDim; row ++) {
      for (uint32_t col = 0; col < outColDim; col ++) {
        for (uint32_t channel = 0; channel < outChannels; channel ++) {
          elementType pixel = hasBias ? bias[channel] : 0;

          for (uint32_t krow = 0; krow < kernelDim; krow ++) {
            if ((row * stride + krow * kernelDilation - padding) % inDilation != 0)
              continue;
            
            const uint32_t irow = (row * stride + krow * kernelDilation - padding) / inDilation;

            for (uint32_t kcol = 0; kcol < kernelDim; kcol ++) {
              if ((col * stride + kcol * kernelDilation - padding) % inDilation != 0)
                continue;
              
              const uint32_t icol = (col * stride + kcol * kernelDilation - padding) / inDilation;

              for (uint32_t kch = 0; kch < inChannels; kch ++) {
                const elementType* in = input + (b * inRowDim * inColDim + irow * inColDim + icol) * inStride  + kch;
                elementType ipixel = irow < 0 || irow >= inRowDim || icol < 0 || icol >= inColDim ? 0 : *in;
                elementType weight = *(kernel + (kch * kernelDim * inChannels + kcol * inChannels + kch) * kernelStride + channel);

                pixel += weight * ipixel;
              }
            }
          }

          elementType* out = output + (b * outRowDim * outColDim + row * outColDim + col) * outStride + channel;
          *out = pixel;
        }
      }
    }
  }
}

/**
 * Convolution operation on PIM
 * 
 * input  : N H W C
 * kernel :
 * bias   : 
 * output : 
*/
template <typename elementType>
static void pim_conv_helper(MemRef<elementType, 4> *input,
                            MemRef<elementType, 2> *kernel,
                            MemRef<elementType, 1> *bias,
                            MemRef<elementType, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim,
                            int64_t stride = 1,
                            int64_t inDilation = 1,
                            int64_t kernelDilation = 1,
                            int64_t padding = 0,
                            int64_t act = 0) {
  printf("[PIM Info] Conv started...\n");

  // Kernel dims map
  const std::unordered_map<uint32_t, uint32_t> kernelDims = {
    {1 , 1}, {4 , 2}, {9 , 3}, {16 , 4}, {25 , 5}, {36 , 6},
    {49 , 7}, {64 , 8}, {81 , 9}, {100 , 10}, {121 , 11}, {144 , 12},
    {169 , 13}, {196 , 14}, {225 , 15}, {256 , 16}, {289, 17}, {324 , 18},
    {361 , 19}, {400 , 20}, {441 , 21}, {484 , 22}, {529 , 23}
  };

  const uint32_t batch = input->getSizes()[0];
  const uint32_t inRow = input->getSizes()[1];
  const uint32_t inCol = input->getSizes()[2];
  const uint32_t inChannels = input->getSizes()[3];
  const uint32_t outChannels = kernel->getSizes()[1];

  conv_without_pool<elementType>(batch, inRow, inCol, inChannels,
                    outChannels, outRowDim, outColDim,
                    stride, inDilation, kernelDilation,
                    padding, kernelDim, 1, 1, 1,
                    input->getData(), kernel->getData(),
                    bias->getData(), output->getData(),
                    act);
}

void pim_await(uint32_t pim_id) {
  printf("[PIM Info] Computation completed\n");
}

// MLIR pim.barrier
void _mlir_ciface_pim_barrier(int32_t tile_id) { pim_await(tile_id); }

// MLIR pim.matmul
void _mlir_ciface_pim_matmul_i8(MemRef<int8_t, 2> *A, 
                                MemRef<int8_t, 2> *B,
                                MemRef<int8_t, 2> *C) {
  pim_matmul_helper<int8_t>(A, B, C);
}

void _mlir_ciface_pim_matmul_i16(MemRef<int16_t, 2> *A,
                                MemRef<int16_t, 2> *B,
                                MemRef<int16_t, 2> *C) {
  pim_matmul_helper<int16_t>(A, B, C);
}

void _mlir_ciface_pim_matmul_i32(MemRef<int32_t, 2> *A, 
                                MemRef<int32_t, 2> *B,
                                MemRef<int32_t, 2> *C) {
  pim_matmul_helper<int32_t>(A, B, C);
}

void _mlir_ciface_pim_matmul_f32(MemRef<_Float32, 2> *A,
                                MemRef<_Float32, 2> *B,
                                MemRef<_Float32, 2> *C) {
  pim_matmul_helper<_Float32>(A, B, C);
}

// MLIR pim.conv
void _mlir_ciface_pim_conv_i8(MemRef<int8_t, 4> *input,
                            MemRef<int8_t, 2> *kernel,
                            MemRef<int8_t, 1> *bias,
                            MemRef<int8_t, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim) {
  pim_conv_helper<int8_t>(input, kernel, bias, output, 
                            outRowDim, outColDim, kernelDim);
}

void _mlir_ciface_pim_conv_i16(MemRef<int16_t, 4> *input,
                            MemRef<int16_t, 2> *kernel,
                            MemRef<int16_t, 1> *bias,
                            MemRef<int16_t, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim) {
  pim_conv_helper<int16_t>(input, kernel, bias, output, 
                              outRowDim, outColDim, kernelDim);
}

// void _mlir_ciface_pim_conv_f16(MemRef<_Float16, 4> *input,
//                             MemRef<_Float16, 2> *kernel,
//                             MemRef<_Float16, 1> *bias,
//                             MemRef<_Float16, 2> *output,
//                             int64_t outRowDim,
//                             int64_t outColDim,
//                             int64_t kernelDim) {
//   pim_conv_helper<_Float16>(input, kernel, bias, output, 
//                               outRowDim, outColDim, kernelDim);
// }

void _mlir_ciface_pim_conv_i32(MemRef<int32_t, 4> *input,
                            MemRef<int32_t, 2> *kernel,
                            MemRef<int32_t, 1> *bias,
                            MemRef<int32_t, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim) {
  pim_conv_helper<int32_t>(input, kernel, bias, output, 
                              outRowDim, outColDim, kernelDim);
}

void _mlir_ciface_pim_conv_f32(MemRef<_Float32, 4> *input,
                            MemRef<_Float32, 2> *kernel,
                            MemRef<_Float32, 1> *bias,
                            MemRef<_Float32, 2> *output,
                            int64_t outRowDim,
                            int64_t outColDim,
                            int64_t kernelDim) {
  pim_conv_helper<_Float32>(input, kernel, bias, output, 
                              outRowDim, outColDim, kernelDim);
}