set(MLIR_INTERFACE_SOURCES
  pim/pim.cpp
)

add_library(mlirInterface SHARED
  ${MLIR_INTERFACE_SOURCES}
)