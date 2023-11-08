//===---- DialectDAP.cpp - Pybind module for DAP dialect API support ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "buddy-mlir-c/Dialects.h"
#include "buddy-mlir-c/Registration.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_dapDialects, m) {

  auto dapM = m.def_submodule("dap");

  buddyMlirRegisterAllPasses();

  dapM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__dap__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}