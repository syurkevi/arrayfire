/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_solve.hpp>

#include <common/err_common.hpp>
#include <complex.hpp>
#include <cudaDataType.hpp>
#include <cuda_runtime.h>
#include <cusparse.hpp>

#include <cusparse.h>
#include <cusolverSp.h>
#include <cusparseModule.hpp>
#include <cusparse_descriptor_helpers.hpp>
#include <math.hpp>
#include <platform.hpp>

#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

namespace cuda {

#if defined(AF_USE_NEW_CUSPARSE_API)
// cusolverStatus_t
// cusolverSp<>csrlsvlu(cusolverSpHandle_t handle,
//    int n,
//    int nnzA,
//    const cusparseMatDescr_t descrA,
//    const <> *csrValA,
//    const int *csrRowPtrA,
//    const int *csrColIndA,
//    const <> *b,
//    float tol,
//    int reorder,
//    <> *x,
//    int *singularity);

template<typename T>
struct csrlsvlu_func_def_t {
    typedef cusolverStatus_t (*csrlsvlu_func_def)(cusolverSpHandle_t,
                                                  int, int, cusparseMatDescr_t,
                                                  T*, int*, int*,
                                                  T*,
                                                  float, int,
                                                  T*,
                                                  int*);
};

#define SOLVE_FUNC_DEF(FUNC) \
    template<typename T>     \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func();

#define SOLVE_FUNC(FUNC, TYPE, PREFIX)                                      \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               cusolverSp##PREFIX##FUNC;                                    \
    }

SOLVE_FUNC_DEF(csrlsvlu)
SOLVE_FUNC(csrlsvlu, float, S)
SOLVE_FUNC(csrlsvlu, double, D)
SOLVE_FUNC(csrlsvlu, cfloat, C)
SOLVE_FUNC(csrlsvlu, cdouble, Z)
#endif


template<typename T>
Array<T> generalSolveSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];
    int K = b.dims()[1];

    Array<T> B = copyArray<T>(b);

#if defined(AF_USE_NEW_CUSPARSE_API)
    auto info = memAlloc<int>(1);

    //csrlsvlu_func<T>()(
    /*
    CUSOLVER_CHECK(getrs_func<T>()(solverDnHandle(), CUBLAS_OP_N, N, K, A.get(),
                                   A.strides()[1], pivot.get(), B.get(),
                                   B.strides()[1], info.get()));
    */

#endif
    return B;
}

template<typename T>
Array<T> leastSquaresSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];

    int K = b.dims()[1];

    Array<T> B = copyArray<T>(b);

#if defined(AF_USE_NEW_CUSPARSE_API)
    //TODO
#endif

    return B;
}

template<typename T>
Array<T> sparse_solve(const common::SparseArray<T> &a, const Array<T> &b,
                      const af_mat_prop options) {
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

}  // namespace cuda
