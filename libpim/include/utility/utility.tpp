//===------------------------- utility.tpp --------------------------------===//
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
// Utils for cpp interact with MLIR
//
//===----------------------------------------------------------------------===//

#ifndef _UTILITY__UTILITY_TPP_
#define _UTILITY__UTILITY_TPP_

#include "utility/utility.hpp"

#include <cstdio>
#include <iostream>

namespace utility {

template <typename T, size_t rank>
void printTensor(MemRef<T, rank> &memRef) {
  std::cout << "Cannot print - unsupported tensor rank\n";
  std::cout << "Tensor dims: ";
  for (auto dim : memRef.getSizes()) {
    std::cout << dim << " ";
  }
  std::cout << "\n";
}

template <typename T>
void printTensor(MemRef<T, 1> &tensor) {
  printVector(tensor);
}

template <typename T>
void printTensor(MemRef<T, 2> &tensor) {
  printMatrix(tensor);
}

template <typename T>
void printTensor(MemRef<T, 4> &tensor) {
  const uint32_t batch = tensor.getSizes()[0];
  const uint32_t row = tensor.getSizes()[1];
  const uint32_t col = tensor.getSizes()[2];
  const uint32_t channel = tensor.getSizes()[3];

  std::cout << "Tensor dims:\n";
  printf("N: %d H: %d W: %d C: %d\n", batch, row, col, channel);

  for (uint32_t b = 0; b < batch; b ++) {
    for (uint32_t h = 0; h < row; h ++) {
      printf("N : %d H : %d\n", b, h);
      for (uint32_t w = 0; w < col; w ++) {
        printf("| ");
        for (uint32_t c = 0; c < channel; c ++) {
          printf("%lf ", tensor[b * row * col * channel + h * col * channel + w * channel + c]);
        }
        printf("|\n");
      }
    }
  }
}

template <typename T>
void printMatrix3D(MemRef<T, 3> &mat) {
  const int numRows = mat.getSizes()[1];
  const int numCols = mat.getSizes()[2];
  const int depthSize = mat.getSizes()[0];

  std::cout << "Matrix dims:\n";
  printf("H: %d W: %d D: %d\n", numRows, numCols, depthSize);

  for (int k = 0; k < depthSize; ++k) {
    printf("D: %d\n", k);
    for (int i = 0; i < numRows; ++i) {
      printf("| ");

      for (int j = 0; j < numCols; ++j) {
        printf("%d ",
               mat[i * numCols * depthSize + j * depthSize + k]);
      }
      printf("|\n");
    }
  }
}

template <typename T>
void printMatrix(MemRef<T, 2> &mat) {
  const int numRows = mat.getSizes()[0];
  const int numCols = mat.getSizes()[1];

  for (int i = 0; i < numRows; ++i) {
    printf("| ");

    for (int j = 0; j < numCols; ++j) {
      printf("%d ", mat[i * numCols + j]);
    }
    printf("|\n");
  }
}

template <typename T>
void printVector(MemRef<T, 1> &vec) {
  printf("| ");

  for (int i = 0; i < vec.getSizes()[0]; ++i) {
    printf("%d ", vec[i]);
  }
  printf("|\n");
}


template <typename T>
void computeGemm(MemRef<T, 2> &A, MemRef<T, 2> &B,
                 MemRef<T, 2> &C) {
  const uint32_t M = C.getSizes()[0];
  const uint32_t N = C.getSizes()[1];
  const uint32_t K = A.getSizes()[1];

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      T sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }

      C[m * N + n] = sum;
    }
  }
}

} // namespace utility

#endif /* _UTILITY__UTILITY_TPP_ */
