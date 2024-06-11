//===------------------------- utility.hpp --------------------------------===//
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

#ifndef _UTILITY__UTILITY_HPP_
#define _UTILITY__UTILITY_HPP_

#include "mlir_interface/core/container.hpp"

namespace utility {

template <typename T, size_t rank>
void printTensor(MemRef<T, rank> &memRef);

template <typename T>
void printMatrix3D(MemRef<T, 3> &memRef);

template <typename T>
void printMatrix(MemRef<T, 2> &memRef);

template <typename T>
void printVector(MemRef<T, 1> &memRef);

template <typename T>
void computeGemm(MemRef<T, 2> &A, MemRef<T, 2> &B,
                 MemRef<T, 2> &C);

} // namespace utility

#include "utility.tpp"

#endif /* _UTILITY__UTILITY_HPP_ */
