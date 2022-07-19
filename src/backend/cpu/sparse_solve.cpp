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

/*
template<typename T>
using gesv_func_def = int (*)(ORDER_TYPE, int, int, T *, int, int *, T *, int);

template<typename T>
using gels_func_def = int (*)(ORDER_TYPE, char, int, int, int, T *, int, T *,
                              int);

#ifdef AF_USE_MKL_BATCH
template<typename T>
using getrf_batch_strided_func_def =
    void (*)(const MKL_INT *m, const MKL_INT *n, T *a, const MKL_INT *lda,
             const MKL_INT *stride_a, MKL_INT *ipiv, const MKL_INT *stride_ipiv,
             const MKL_INT *batch_size, MKL_INT *info);

#if INTEL_MKL_VERSION >= 20210004
template<typename T>
using getrs_batch_strided_func_def = void (*)(
    const char *trans, const MKL_INT *n, const MKL_INT *nrhs, const T *a,
    const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv,
    const MKL_INT *stride_ipiv, T *b, const MKL_INT *ldb,
    const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info);
#else
template<typename T>
using getrs_batch_strided_func_def =
    void (*)(const char *trans, const MKL_INT *n, const MKL_INT *nrhs, T *a,
             const MKL_INT *lda, const MKL_INT *stride_a, MKL_INT *ipiv,
             const MKL_INT *stride_ipiv, T *b, const MKL_INT *ldb,
             const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info);
#endif
#endif


template<typename T>
using getrs_func_def = int (*)(ORDER_TYPE, char, int, int, const T *, int,
                               const int *, T *, int);

template<typename T>
using trtrs_func_def = int (*)(ORDER_TYPE, char, char, char, int, int,
                               const T *, int, T *, int);

#define SOLVE_FUNC_DEF(FUNC) \
    template<typename T>     \
    FUNC##_func_def<T> FUNC##_func();

#define SOLVE_FUNC(FUNC, TYPE, PREFIX)          \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

SOLVE_FUNC_DEF(gesv)
SOLVE_FUNC(gesv, float, s)
SOLVE_FUNC(gesv, double, d)
SOLVE_FUNC(gesv, cfloat, c)
SOLVE_FUNC(gesv, cdouble, z)

SOLVE_FUNC_DEF(gels)
SOLVE_FUNC(gels, float, s)
SOLVE_FUNC(gels, double, d)
SOLVE_FUNC(gels, cfloat, c)
SOLVE_FUNC(gels, cdouble, z)

#ifdef AF_USE_MKL_BATCH

template<typename T>
struct mkl_type {
    using type = T;
};
template<>
struct mkl_type<std::complex<float>> {
    using type = MKL_Complex8;
};
template<>
struct mkl_type<std::complex<double>> {
    using type = MKL_Complex16;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnoexcept-type"
template<typename T>
getrf_batch_strided_func_def<T> getrf_batch_strided_func();

template<>
getrf_batch_strided_func_def<float> getrf_batch_strided_func<float>() {
    return &sgetrf_batch_strided;
}
template<>
getrf_batch_strided_func_def<double> getrf_batch_strided_func<double>() {
    return &dgetrf_batch_strided;
}
template<>
getrf_batch_strided_func_def<MKL_Complex8>
getrf_batch_strided_func<MKL_Complex8>() {
    return &cgetrf_batch_strided;
}
template<>
getrf_batch_strided_func_def<MKL_Complex16>
getrf_batch_strided_func<MKL_Complex16>() {
    return &zgetrf_batch_strided;
}

template<typename T>
getrs_batch_strided_func_def<T> getrs_batch_strided_func();

template<>
getrs_batch_strided_func_def<float> getrs_batch_strided_func<float>() {
    return &sgetrs_batch_strided;
}
template<>
getrs_batch_strided_func_def<double> getrs_batch_strided_func<double>() {
    return &dgetrs_batch_strided;
}
template<>
getrs_batch_strided_func_def<MKL_Complex8>
getrs_batch_strided_func<MKL_Complex8>() {
    return &cgetrs_batch_strided;
}
template<>
getrs_batch_strided_func_def<MKL_Complex16>
getrs_batch_strided_func<MKL_Complex16>() {
    return &zgetrs_batch_strided;
}

#pragma GCC diagnostic pop
#endif

SOLVE_FUNC_DEF(getrs)
SOLVE_FUNC(getrs, float, s)
SOLVE_FUNC(getrs, double, d)
SOLVE_FUNC(getrs, cfloat, c)
SOLVE_FUNC(getrs, cdouble, z)

SOLVE_FUNC_DEF(trtrs)
SOLVE_FUNC(trtrs, float, s)
SOLVE_FUNC(trtrs, double, d)
SOLVE_FUNC(trtrs, cfloat, c)
SOLVE_FUNC(trtrs, cdouble, z)
*/

/*
template<typename T>
Array<T> triangleSolveSparse(const common::SparseArray<T> &A, const Array<T> &b,
                       const af_mat_prop options) {
    Array<T> B = copyArray<T>(b);
    int N      = B.dims()[0];
    int NRHS   = B.dims()[1];

    auto func = [=](const CParam<T> A, Param<T> B, int N, int NRHS,
                    const af_mat_prop options) {
        trtrs_func<T>()(AF_LAPACK_COL_MAJOR, options & AF_MAT_UPPER ? 'U' : 'L',
                        'N',  // transpose flag
                        options & AF_MAT_DIAG_UNIT ? 'U' : 'N', N, NRHS,
                        A.get(), A.strides(1), B.get(), B.strides(1));
    };
    getQueue().enqueue(func, A, B, N, NRHS, options);

    return B;
}

#ifdef AF_USE_MKL_BATCH

template<typename T>
Array<T> generalSolveSparse(const common::SparseArray<T> &a, const Array<T> &b,
                             const af_mat_prop options) {
    using std::vector;
    int batches = a.dims()[2] * a.dims()[3];

    dim4 aDims = a.dims();
    dim4 bDims = b.dims();
    int M      = aDims[0];
    int N      = aDims[1];
    int K      = bDims[1];
    int MN     = std::min(M, N);

    int lda     = a.strides()[1];
    int astride = a.strides()[2];

    vector<int> ipiv(MN * batches);
    int ipivstride = MN;

    int ldb     = b.strides()[1];
    int bstride = b.strides()[2];

    vector<int> info(batches, 0);

    char trans = 'N';

    Array<T> A = copyArray<T>(a);
    Array<T> B = copyArray<T>(b);

    auto getrf_rs = [](char TRANS, int M, int N, int K, Param<T> a, int LDA,
                       int ASTRIDE, vector<int> IPIV, int IPIVSTRIDE,
                       Param<T> b, int LDB, int BSTRIDE, int BATCH_SIZE,
                       vector<int> INFO) {
        getrf_batch_strided_func<typename mkl_type<T>::type>()(
            &M, &N, reinterpret_cast<typename mkl_type<T>::type *>(a.get()),
            &LDA, &ASTRIDE, IPIV.data(), &IPIVSTRIDE, &BATCH_SIZE, INFO.data());

        getrs_batch_strided_func<typename mkl_type<T>::type>()(
            &TRANS, &M, &K,
            reinterpret_cast<typename mkl_type<T>::type *>(a.get()), &LDA,
            &ASTRIDE, IPIV.data(), &IPIVSTRIDE,
            reinterpret_cast<typename mkl_type<T>::type *>(b.get()), &LDB,
            &BSTRIDE, &BATCH_SIZE, INFO.data());
    };

    getQueue().enqueue(getrf_rs, trans, M, N, K, A, lda, astride, ipiv,
                       ipivstride, B, ldb, bstride, batches, info);

    return B;
}
#endif
*/

template<typename T>
Array<T> sparse_solve(const common::SparseArray<T> &a, const Array<T> &b,
                      const af_mat_prop options) {
    const dim4 NullShape(0, 0, 0, 0);

    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];
    printf("(MxN) %d %d\n", M, N);

    int K = b.dims()[1];
    printf("(MxK) %d %d\n", M, K);

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
            // PARDISO INIT
            void *pt[64];
            /*
            //ia
            for(int i=0; i<rowIdx.dims()[0]; ++i) {
                printf("%d ", pRowIdx.get()[i]);
            }printf("\n");
            //ja
            for(int i=0; i<colIdx.dims()[0]; ++i) {
                printf("%d ", pColIdx.get()[i]);
            }printf("\n");
            //a
            for(int i=0; i<values.dims()[0]; ++i) {
                printf("%f ", pValues.get()[i]);
            }printf("\n");
            */
            constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

            constexpr bool IsSingle =
        std::is_same<T, float>::value || std::is_same<T, cfloat>::value;
            printf("check single%d complex %d\n", IsSingle, IsComplex);

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
