# Copyright (c) 2022, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(InternalUtils)
include(build_cl2hpp)

add_library(afoneapi
  Array.cpp
  Event.cpp
  approx.cpp
  approx.hpp
  arith.hpp
  assign.cpp
  assign.hpp
  backend.hpp
  bilateral.cpp
  bilateral.hpp
  binary.hpp
  blas.cpp
  blas.hpp
  canny.cpp
  canny.hpp
  cast.hpp
  cholesky.cpp
  cholesky.hpp
  compile_module.cpp
  complex.hpp
  convolve.cpp
  convolve.hpp
  convolve_separable.cpp
  copy.cpp
  copy.hpp
  count.cpp
  device_manager.cpp
  device_manager.hpp
  platform.cpp
  platform.hpp
  wrap.cpp
  where.cpp
  unwrap.cpp
  product.cpp
  qr.cpp
  random_engine.cpp
  range.cpp
  regions.cpp
  reorder.cpp
  reshape.cpp
  resize.cpp
  rotate.cpp
  scan.cpp
  scan_by_key.cpp
  select.cpp
  set.cpp
  sift.cpp
  sobel.cpp
  solve.cpp
  sort.cpp
  sort_index.cpp
  sparse.cpp
  sparse_arith.cpp
  sparse_blas.cpp
  sum.cpp
  surface.cpp
  susan.cpp
  svd.cpp
  tile.cpp
  topk.cpp
  transform.cpp
  transpose.cpp
  triangle.cpp
  )

add_library(ArrayFire::afoneapi ALIAS afoneapi)

arrayfire_set_default_cxx_flags(afoneapi)

target_include_directories(afoneapi
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SYCL_INCLUDE_DIR}
  )

target_compile_definitions(afoneapi
  PRIVATE
    AF_ONEAPI
  )

target_link_libraries(afoneapi
  PRIVATE
    c_api_interface
    cpp_api_interface
    afcommon_interface
    ${SYCL_FLAGS}
  )

source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c   REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group(backend  REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*)
source_group("generated files" FILES ${ArrayFire_BINARY_DIR}/version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h)
source_group("" FILES CMakeLists.txt)
