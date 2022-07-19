/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_solve.hpp>

#include <err_cpu.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <copy.hpp>
#include <lapack_helper.hpp>
#include <math.hpp>

#if USE_MKL
#include <mkl_version.h>
#include <mkl_spblas.h>
#include <mkl_pardiso.h>
#include <mkl_types.h>
#endif

#include <queue.hpp>
#include <af/dim4.hpp>
#include <algorithm>
#include <complex>
#include <vector>


using af::dim4;

namespace cpu {

#if USE_MKL
template<typename T>
void pardiso_sparse_solve(CParam<T> A_vals, CParam<int> A_rows, CParam<int> A_cols,
                          Param<T> B, Param<T> X, int N, int K) {
    // PARDISO INIT
    void *pt[64];

    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    constexpr bool IsSingle =
        std::is_same<T, float>::value || std::is_same<T, cfloat>::value;

    MKL_INT mtype = 11; // real-unsymmetric
    if(IsComplex) {
        mtype = 13; // complex-unsymmetric
    }

    MKL_INT nrhs = K;
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;

    // zero out params as default
    for(int i=0; i<64; ++i) {
        iparm[i] = 0;
    }

    for(int i=0; i<64; ++i) {
        pt[i] = 0;
    }

    iparm[0] = 1;   // No solver default
    iparm[1] = 2;   // Fill-in reordering from METIS
    iparm[3] = 0;   // No iterative-direct algorithm
    iparm[4] = 0;   // No user fill-in reducing permutation
    iparm[5] = 0;   // Write solution into x
    iparm[7] = 2;   // Max numbers of iterative refinement steps
    iparm[8] = 0;   // Not in use
    iparm[9] = 13;  // Perturb the pivot elements with 1E-13
    iparm[10] = 1;  // Use nonsymmetric permutation and scaling MPS
    iparm[11] = 0;  // Conjugate transposed/transpose solve
    iparm[12] = 1;  // Maximum weighted matching algorithm is switched-on (default for non-    symmetric)
    iparm[13] = 0;  // Output: Number of perturbed pivots
    iparm[14] = 0;  // Not in use
    iparm[15] = 0;  // Not in use
    iparm[16] = 0;  // Not in use
    iparm[17] = -1; // Output: Number of nonzeros in the factor LU
    iparm[18] = -1; // Output: Mflops for LU factorization
    iparm[19] = 0;  // Output: Numbers of CG Iterations
    iparm[27] = IsSingle ? 1 : 0;  // 1 - single precision, 0 - double
    iparm[34] = 1;  // zero based indexing

    maxfct = 1;
    mnum  = 1;
    msglvl = 0;
    error = 0;

    MKL_INT perm;

    // solver runs in multiple phases
    phase = 11; // memory allocation & analysis
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &N, A_vals.get(), A_rows.get(), A_cols.get(),
            &perm, &nrhs, iparm, &msglvl, B.get(), X.get(), &error);

    if (error != 0) {
        std::string errormsg =
            "PARDISO solver failed in phase=11, with error: " +
            std::to_string(error);
        AF_ERROR(errormsg, AF_ERR_RUNTIME);
    }

    phase = 22; // numerical factorization
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &N, A_vals.get(), A_rows.get(), A_cols.get(),
             &perm, &nrhs, iparm, &msglvl, B.get(), X.get(), &error);
    if (error != 0) {
        std::string errormsg =
            "PARDISO solver failed in phase=22, with error: " +
            std::to_string(error);
        AF_ERROR(errormsg, AF_ERR_RUNTIME);
    }

    phase = 33; // solve, iterative refinement
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &N, A_vals.get(), A_rows.get(), A_cols.get(),
             &perm, &nrhs, iparm, &msglvl, B.get(), X.get(), &error);
    if (error != 0) {
        std::string errormsg =
            "PARDISO solver failed in phase=33, with error: " +
            std::to_string(error);
        AF_ERROR(errormsg, AF_ERR_RUNTIME);
    }

    phase = -1; // release internal memory
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &N, A_vals.get(), A_rows.get(), A_cols.get(), &perm, &nrhs,
             iparm, &msglvl, B.get(), X.get(), &error);

    if (error != 0) {
        std::string errormsg =
            "PARDISO solver failed in cleanup phase=-1, with error: " +
            std::to_string(error);
        AF_ERROR(errormsg, AF_ERR_RUNTIME);
    }
}
#endif

template<typename T>
Array<T> sparse_solve(const common::SparseArray<T> &a, const Array<T> &b,
                      const af_mat_prop options) {
    const dim4 NullShape(0, 0, 0, 0);

    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];

    int K = b.dims()[1];

    Array<T> values   = a.getValues();
    Array<int> rowIdx = a.getRowIdx();
    Array<int> colIdx = a.getColIdx();

    dim4 endPadding(max(M, N) - b.dims()[0], K - b.dims()[1], 0, 0);
    Array<T> B = (endPadding == NullShape
                      ? copyArray(b)
                      : padArrayBorders(b, NullShape, endPadding, AF_PAD_ZERO));

    Array<T> X = copyArray(b);

    // A matrix in CSR TODO: check CSR format only?
    Param<T>   pValues(values.get(), values.dims(), values.strides());
    Param<int> pRowIdx(rowIdx.get(), rowIdx.dims(), rowIdx.strides());
    Param<int> pColIdx(colIdx.get(), colIdx.dims(), colIdx.strides());

    // b,x vectors MxK
    Param<T>   pB(B.get(), B.dims(), B.strides());
    Param<T>   pX(X.get(), X.dims(), X.strides());

    if (M == N) {

        auto func = [](Param<T> A_vals, Param<int> A_rows, Param<int>A_cols,
                       Param<T> B, Param<T> X, int N, int K) {
#if USE_MKL
            pardiso_sparse_solve<T>(A_vals, A_rows, A_cols, B, X, N, K);
#else
            AF_ERROR(
                "This version of ArrayFire was built without a sparse solver"
                "for the CPU backend",
                AF_ERR_NOT_CONFIGURED);
#endif
        };

        getQueue().enqueue(func, pValues, pRowIdx, pColIdx, pB, pX, N, K);
        return X;
    } else {
        // Not supported for sparse!
        AF_ERROR("Overdetermined or underdetermined sparse systems "
                 "are not supported for the CPU backend", AF_ERR_RUNTIME);
    }

    return X;
}

}  // namespace cpu

#else  // WITH_LINEAR_ALGEBRA

namespace cpu {

template<typename T>
Array<T> sparse_solve(const common::SparseArray<T> &a, const Array<T> &b,
                      const af_mat_prop options) {
    AF_ERROR(
        "This version of ArrayFire was built without linear algebra routines",
        AF_ERR_NOT_CONFIGURED);
}

}  // namespace cpu

#endif  // WITH_LINEAR_ALGEBRA

namespace cpu {

#define INSTANTIATE_SOLVE(T)                                           \
    template Array<T> sparse_solve<T>(const common::SparseArray<T> &a, \
                                      const Array<T> &b,               \
                                      const af_mat_prop options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}  // namespace cpu
