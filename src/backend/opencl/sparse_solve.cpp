/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>

#include <err_opencl.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <blas.hpp>
#include <copy.hpp>
#include <cpu/cpu_sparse_solve.hpp>
#include <lu.hpp>
#include <magma/magma.h>
#include <magma/magma_blas.h>
#include <magma/magma_data.h>
#include <magma/magma_helper.h>
#include <math.hpp>
#include <platform.hpp>
#include <transpose.hpp>
#include <af/opencl.h>

#include <algorithm>
#include <vector>

using cl::Buffer;
using std::min;
using std::vector;

namespace opencl {

template<typename T>
Array<T> generalSolveSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];

    int K = b.dims()[1];

    Array<T> B = copyArray<T>(b);


    /*
    Buffer *A_buf      = A.get();
    int info           = 0;
    cl_command_queue q = getQueue()();
    auto aoffset =
        A.getOffset() + j * A.strides()[2] + i * A.strides()[3];
    magma_getrf_gpu<T>(M, N, (*A_buf)(), aoffset, A.strides()[1],
                       &ipiv[0], q, &info);

    Buffer *B_buf = B.get();
    int K         = B.dims()[1];

    auto boffset =
        B.getOffset() + j * B.strides()[2] + i * B.strides()[3];
    magma_getrs_gpu<T>(MagmaNoTrans, M, K, (*A_buf)(), aoffset,
                       A.strides()[1], &ipiv[0], (*B_buf)(), boffset,
                       B.strides()[1], q, &info);
    */

    return B;
}

template<typename T>
Array<T> leastSquaresSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];

    int K = b.dims()[1];

    Array<T> B = copyArray<T>(b);

    /*
    Buffer *A_buf      = A.get();
    int info           = 0;
    cl_command_queue q = getQueue()();
    auto aoffset =
        A.getOffset() + j * A.strides()[2] + i * A.strides()[3];
    magma_getrf_gpu<T>(M, N, (*A_buf)(), aoffset, A.strides()[1],
                       &ipiv[0], q, &info);

    Buffer *B_buf = B.get();
    int K         = B.dims()[1];

    auto boffset =
        B.getOffset() + j * B.strides()[2] + i * B.strides()[3];
    magma_getrs_gpu<T>(MagmaNoTrans, M, K, (*A_buf)(), aoffset,
                       A.strides()[1], &ipiv[0], (*B_buf)(), boffset,
                       B.strides()[1], q, &info);
    */

    return B;
}

template<typename T>
Array<T> sparse_solve(const common::SparseArray<T> &a, const Array<T> &b,
                      const af_mat_prop options) {
    if (OpenCLCPUOffload()) { return cpu::sparse_solve(a, b, options); }

    Array<T> X = copyArray(b);

    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];
    if (M == N) {
        return generalSolveSparse<T>(a, b);
    } else {
        return leastSquaresSparse<T>(a, b);
    }

    return X;
}

#define INSTANTIATE_SOLVE(T)                                           \
    template Array<T> sparse_solve<T>(const common::SparseArray<T> &a, \
                                      const Array<T> &b,               \
                                      const af_mat_prop options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}  // namespace opencl

#else  // WITH_LINEAR_ALGEBRA

namespace opencl {

template<typename T>
Array<T> sparse_solve(const common::SparseArray<T> &a, const Array<T> &b,
                      const af_mat_prop options) {
    AF_ERROR(
        "This version of ArrayFire was built without linear algebra routines",
        AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                           \
    template Array<T> sparse_solve<T>(const common::SparseArray<T> &a, \
                                      const Array<T> &b,               \
                                      const af_mat_prop options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}  // namespace opencl

#endif  // WITH_LINEAR_ALGEBRA
