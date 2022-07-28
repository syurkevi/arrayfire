/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/sparse_diag.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {

constexpr unsigned TX      = 32;
constexpr unsigned TY      = 8;
constexpr unsigned THREADS = TX * TY;

template<typename T>
void sparseDiag(Param out, const Param values, const Param rowIdx,
                const Param colIdx, const int nRows, const int nCols) {
//void sparseDiag(Param out, const Param values, const Param rowIdx,
                //const Param colIdx, const Param rhs, const bool reverse) {

    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
    };

    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(IS_CPLX, (IsComplex ? 1 : 0)), //TODO: needed?
    };

    options.emplace_back(getTypeBuildDefinition<T>());
    auto sparseExtractDiagKernel = common::getKernel("extract_diag",
                          {sparse_diag_cl_src}, tmpltArgs, options);

    cl::NDRange local(TX, TY, 1);
    cl::NDRange global(divup(out.info.dims[0], TY) * TX, TY, 1);
    printf("t<%d %d> b<%d %d>\n", TX, TY, divup(out.info.dims[0], TY) * TX, TY);

    sparseExtractDiagKernel(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                            out.info, *values.data, *rowIdx.data, *colIdx.data,
                            nRows, nCols);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
