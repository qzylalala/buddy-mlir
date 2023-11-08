//===- Registration.cpp - C Interface for MLIR Registration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "buddy-mlir-c/Registration.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "buddy-mlir-c/InitAll.h"

void buddyMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  mlir::buddy::registerAllDialects(registry);

  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void buddyMlirRegisterAllPasses() { mlir::buddy::registerAllPasses(); }