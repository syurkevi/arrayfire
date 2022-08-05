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
#include <copy.hpp>

#include <cusparse.h>
#include <cusolverSp.hpp>
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

//cusolverStatus_t
//cusolverSpScsrlsvqr[Host](cusolverSpHandle_t handle,
//   int m,
//   int nnz,
//   const cusparseMatDescr_t descrA,
//   const float *csrValA,
//   const int *csrRowPtrA,
//   const int *csrColIndA,
//   const float *b,
//   float tol,
//   int reorder,
//   float *x,
//   int *singularity);
template<typename T>
struct csrlsvqr_func_def_t {
    typedef cusolverStatus_t (*csrlsvqr_func_def)(cusolverSpHandle_t,
                                                  int, int, cusparseMatDescr_t,
                                                  const T*, const int*, const int*,
                                                  const T*,
                                                  float, int,
                                                  T*,
                                                  int*);
};

//cusolverStatus_t
//cusolverSpScsrlsvchol[Host](cusolverSpHandle_t handle,
//   int m,
//   int nnz,
//   const cusparseMatDescr_t descrA,
//   const float *csrVal,
//   const int *csrRowPtr,
//   const int *csrColInd,
//   const float *b,
//   float tol,
//   int reorder,
//   float *x,
//   int *singularity);
template<typename T>
struct csrlsvchol_func_def_t {
    typedef cusolverStatus_t (*csrlsvchol_func_def)(cusolverSpHandle_t,
                                                  int, int, cusparseMatDescr_t,
                                                  const T*, const int*, const int*,
                                                  const T*,
                                                  float, int,
                                                  T*,
                                                  int*);
};

//cusolverStatus_t
//cusolverSpScsrlsqvqr[Host](cusolverSpHandle_t handle,
//   int m,
//   int n,
//   int nnz,
//   const cusparseMatDescr_t descrA,
//   const float *csrValA,
//   const int *csrRowPtrA,
//   const int *csrColIndA,
//   const float *b,
//   float tol,
//   int *rankA,
//   float *x,
//   int *p,
//   float *min_norm);
template<typename T>
struct csrlsqvqr_func_def_t {
    typedef cusolverStatus_t (*csrlsqvqr_func_def)(cusolverSpHandle_t,
                                                  int, int, int, cusparseMatDescr_t,
                                                  const T*, const int*, const int*,
                                                  const T*,
                                                  float, int*,
                                                  T*, int*,
                                                  float*);
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

//TODO: only CPU version currently supported,
//update once added to cuSolver
//lu linear solvers
//SOLVE_FUNC_DEF(csrlsvlu)
//SOLVE_FUNC(csrlsvlu, float, S)
//SOLVE_FUNC(csrlsvlu, double, D)
//SOLVE_FUNC(csrlsvlu, cfloat, C)
//SOLVE_FUNC(csrlsvlu, cdouble, Z)

// qr linear solvers
SOLVE_FUNC_DEF(csrlsvqr)
SOLVE_FUNC(csrlsvqr, float, S)
SOLVE_FUNC(csrlsvqr, double, D)
SOLVE_FUNC(csrlsvqr, cfloat, C)
SOLVE_FUNC(csrlsvqr, cdouble, Z)

// cholesky factorization linear solvers
SOLVE_FUNC_DEF(csrlsvchol)
SOLVE_FUNC(csrlsvchol, float, S)
SOLVE_FUNC(csrlsvchol, double, D)
SOLVE_FUNC(csrlsvchol, cfloat, C)
SOLVE_FUNC(csrlsvchol, cdouble, Z)

//TODO: only CPU version currently supported,
//update once added to cuSolver
//least-squares solvers
//SOLVE_FUNC_DEF(csrlsqvqr)
//SOLVE_FUNC(csrlsqvqr, float, S)
//SOLVE_FUNC(csrlsqvqr, double, D)
//SOLVE_FUNC(csrlsqvqr, cfloat, C)
//SOLVE_FUNC(csrlsqvqr, cdouble, Z)
#endif


template<typename T>
Array<T> generalSolveSparse(const common::SparseArray<T> &a, const Array<T> &b, bool symmetric_positive_definite) {
    static constexpr double abstol = 1e-8;
    static constexpr int reorder = 0;

    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];
    int K = b.dims()[1];

    Array<T> X = createValueArray<T>(af::dim4(N, K), scalar<T>(0));

    Array<T>   vals   = a.getValues();
    Array<int> rowIdx = a.getRowIdx();
    Array<int> colIdx = a.getColIdx();

#if defined(AF_USE_NEW_CUSPARSE_API)
    auto info = memAlloc<int>(1);

    cusparseMatDescr_t descrA;
    cusparseModule &_ = getCusparsePlugin();
    CUSPARSE_CHECK(_.cusparseCreateMatDescr(&descrA));
    CUSPARSE_CHECK(_.cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(_.cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    int singularity;

    std::vector<af_seq> vecIndex(4, af_span);
    for(size_t k=0; k<K; ++k) {
        vecIndex[1] = {double(k), double(k), 1};
        Array<T> bvec = createSubArray(b, vecIndex, false);
        Array<T> xvec = createEmptyArray<T>(af::dim4(N));

        if(symmetric_positive_definite) {
            CUSOLVER_SP_CHECK(csrlsvchol_func<T>()(solverSpHandle(),
                                                   M, vals.elements(),
                                                   descrA,
                                                   vals.get(),
                                                   rowIdx.get(),
                                                   colIdx.get(),
                                                   bvec.get(),
                                                   abstol,
                                                   reorder,
                                                   xvec.get(),
                                                   &singularity));
            Array<T> Xsub = createSubArray(X, vecIndex, false);
            copyArray(Xsub, xvec);
        } else {
            CUSOLVER_SP_CHECK(csrlsvqr_func<T>()(solverSpHandle(),
                                                 M, vals.elements(),
                                                 descrA,
                                                 vals.get(),
                                                 rowIdx.get(),
                                                 colIdx.get(),
                                                 bvec.get(),
                                                 abstol,
                                                 reorder,
                                                 xvec.get(),
                                                 &singularity));
            Array<T> Xsub = createSubArray(X, vecIndex, false);
            copyArray(Xsub, xvec);
        }
        //TODO: error on singular matrix?
    }

    CUSPARSE_CHECK(_.cusparseDestroyMatDescr(descrA));
#else
    AF_ERROR("sparse solve unsupported for older (<10.x) "
             "CUDA toolkit versions", AF_ERR_RUNTIME);
#endif

    return X;
}

template<typename T>
Array<T> leastSquaresSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    // Not supported for sparse!
    AF_ERROR("Overdetermined or underdetermined sparse systems "
             "are not supported for the CPU backend", AF_ERR_RUNTIME);

    Array<T> B = copyArray<T>(b);
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
        bool symmetric_positive_definite = false;
        if (options & AF_MAT_SYM &&
            options & AF_MAT_POSDEF) {
            symmetric_positive_definite = true;
        }
        return generalSolveSparse<T>(a, b, symmetric_positive_definite);
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
