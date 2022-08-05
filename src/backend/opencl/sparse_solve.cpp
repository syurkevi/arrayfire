/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>

#include <common/cast.hpp>
#include <err_opencl.hpp>
#include <arith.hpp>
#include <complex.hpp>
#include <Array.hpp>
#include <lookup.hpp>
#include <range.hpp>
#include <reduce.hpp>
#include <scan.hpp>
#include <where.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <blas.hpp>
#include <sparse_blas.hpp>
#include <copy.hpp>
#include <cpu/cpu_sparse_solve.hpp>
#include <common/debug.hpp>
#include <lu.hpp>
#include <magma/magma.h>
#include <magma/magma_blas.h>
#include <magma/magma_data.h>
#include <magma/magma_helper.h>
#include <kernel/sparse_diag.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <transpose.hpp>
#include <af/opencl.h>

#include <algorithm>
#include <vector>
#include <complex>

using cl::Buffer;
using common::cast;
using common::createArrayDataSparseArray;
using detail::createSubArray;
using std::min;
using std::vector;

namespace opencl {

template<typename T>
T dot_host_helper(Array<T> &vec0, Array<T> &vec1) {
    Array<T> dotres = dot<T>(vec0, vec1, AF_MAT_NONE, AF_MAT_NONE);
    T res{};
    getQueue().enqueueReadBuffer(*dotres.get(), CL_TRUE,
                sizeof(T) * dotres.getOffset(), sizeof(T), &res);
    return res;
}

template<>
cfloat dot_host_helper(Array<cfloat> &vec0, Array<cfloat> &vec1) {
    // complex dot product
    Array<cfloat> dotres = dot<cfloat>(vec0, vec1, AF_MAT_NONE, AF_MAT_CONJ);
    cfloat res{};
    getQueue().enqueueReadBuffer(*dotres.get(), CL_TRUE,
                sizeof(cfloat) * dotres.getOffset(), sizeof(cfloat), &res);
    return res;
}

template<>
cdouble dot_host_helper(Array<cdouble> &vec0, Array<cdouble> &vec1) {
    // complex dot product
    Array<cdouble> dotres = dot<cdouble>(vec0, vec1, AF_MAT_NONE, AF_MAT_CONJ);
    cdouble res{};
    getQueue().enqueueReadBuffer(*dotres.get(), CL_TRUE,
                sizeof(cdouble) * dotres.getOffset(), sizeof(cdouble), &res);
    return res;
}

template<typename T, af_op_t op>
Array<T> sv_arith_helper(T scalar, Array<T> arr) {
    return arithOp<T, op>(createValueArray<T>(arr.dims(), scalar), arr, arr.dims());
}

template<typename T>
common::SparseArray<T> jacobiPreconditionMatrix(const common::SparseArray<T> &A) {
    Array<T>   vals   = A.getValues();
    Array<int> rowIdx = A.getRowIdx();
    Array<int> colIdx = A.getColIdx();

    Array<T> diag = createValueArray<T>(af::dim4(rowIdx.elements()), scalar<T>(0));
    kernel::invDiag<T>(diag, vals, rowIdx, colIdx, A.dims()[0], A.dims()[1]);

    Array<uint> nonZeroIdx_ = where<T>(diag);
    Array<int> nonZeroIdx   = cast<int, uint>(nonZeroIdx_);
    Array<T> diagValues    = lookup<T, int>(diag, nonZeroIdx, 0);

    Array<int> diagColIdx    = lookup<int, int>(range<int>(diag.dims()), nonZeroIdx, 0);

    Array<T> zero = createValueArray<T>(diag.dims(), scalar<T>(0));
    Array<int> diagRowIdx = cast<int, uint>(scan<af_add_t, char, uint>(common::createBinaryNode<char, T, af_neq_t>(diag, zero, diag.dims()), 0, false));

    return createArrayDataSparseArray<T>(A.dims(), diagValues, diagRowIdx, diagColIdx,
                                         AF_STORAGE_CSR);
}

template<typename T>
struct CG {
    static constexpr double abstol = 1e-8;
    static constexpr double reltol = 1e-6;

    vector<double> residual_history;
    bool precondition;
    int max_iterations;
    int iterations;

    Array<T> solve(const common::SparseArray<T> &a, const Array<T> &b) {
        //residual_history.clear();
        const int N = b.dims()[0];

        Array<T> x  = createValueArray<T>(af::dim4(N), scalar<T>(0));
        //r0 == b - Ax0
        Array<T> r0 = arithOp<T, af_sub_t>(copyArray<T>(b),
                matmul(a, x, AF_MAT_NONE, AF_MAT_NONE), b.dims());

        double residual_norm = sqrt(abs(dot_host_helper<T>(r0, r0)));
        double threshold = max(CG::abstol, CG::reltol * residual_norm);

        if(residual_norm <= threshold) {
            return x;
        }

        Array<T> r = copyArray<T>(r0);
        Array<T> p = copyArray<T>(r0);

        // recalculate full residual b - Ax,
        // instead of partial update every so often
        int recalculate_residual_iter = min(10, 1 + (max_iterations / 4));

        while(iterations < max_iterations && residual_norm > threshold) {
            Array<T> Ap = matmul(a, p, AF_MAT_NONE, AF_MAT_NONE);
            double r0r0_dot = dot_host_helper<T>(r, r);

            double alpha  = r0r0_dot /
                            dot_host_helper<T>(p, Ap);

            // x1 = x0 + alpha*p0
            x = arithOp<T, af_add_t>(x,
                    sv_arith_helper<T, af_mul_t>(scalar<T>(alpha),
                    p), x.dims());

            if(iterations % recalculate_residual_iter == 0) {
                // more accurate resudual calculation,
                // not as efficient, both should be used
                r = arithOp<T, af_sub_t>(copyArray<T>(b),
                        matmul(a, x, AF_MAT_NONE, AF_MAT_NONE), b.dims());
            } else {
                // r1 = r0 - alpha*Ap
                // susceptible to roundoff error, faster
                r = arithOp<T, af_sub_t>(r,
                        sv_arith_helper<T, af_mul_t>(scalar<T>(alpha),
                        Ap), r.dims());
            }
            residual_norm = sqrt(dot_host_helper<T>(r, r));
            if(residual_norm <= threshold) {
                return x;
            }

            double beta = dot_host_helper<T>(r, r) / r0r0_dot;
            p = arithOp<T, af_sub_t>(r,
                    sv_arith_helper<T, af_mul_t>(
                        scalar<T>(beta), p), r.dims());

            iterations++;
        }
        return x;
    }

    Array<T> solve(const common::SparseArray<T> &a, const common::SparseArray<T> &Pinv,
                   const Array<T> &b) {
        //residual_history.clear();
        const int N = b.dims()[0];

        Array<T> x  = createValueArray<T>(af::dim4(N), scalar<T>(0));
        //r0 == b - Ax0
        Array<T> r0 = arithOp<T, af_sub_t>(copyArray<T>(b),
                matmul(a, x, AF_MAT_NONE, AF_MAT_NONE), b.dims());

        double residual_norm = sqrt(abs(dot_host_helper<T>(r0, r0)));
        double threshold = max(CG::abstol, CG::reltol * residual_norm);

        if(residual_norm <= threshold) {
            return x;
        }

        Array<T> r = copyArray<T>(r0);
        Array<T> z = copyArray<T>(r0);
        Array<T> p = copyArray<T>(r0);
        if(precondition) {
            z = matmul(Pinv, r0, AF_MAT_NONE, AF_MAT_NONE);
            p = copyArray<T>(z);
        }

        // recalculate full residual b - Ax,
        // instead of partial update every so often
        int recalculate_residual_iter = min(10, 1 + (max_iterations / 4));

        while(iterations < max_iterations && residual_norm > threshold) {
            Array<T> Ap = matmul(a, p, AF_MAT_NONE, AF_MAT_NONE);
            T r0z0_dot = dot_host_helper<T>(r, z);

            T alpha  = r0z0_dot /
                       dot_host_helper<T>(p, Ap);

            // x1 = x0 + alpha*p0
            x = arithOp<T, af_add_t>(x,
                    //sv_arith_helper<T, af_mul_t>(scalar<T>(alpha),
                    sv_arith_helper<T, af_mul_t>(alpha,
                    p), x.dims());
            // r1 = r0 - alpha*Ap
            if(iterations % recalculate_residual_iter == 0) {
                // more accurate resudual calculation,
                // not as efficient, both should be used
                r = arithOp<T, af_sub_t>(copyArray<T>(b),
                        matmul(a, x, AF_MAT_NONE, AF_MAT_NONE), b.dims());
            } else {
                // susceptible to roundoff error, faster
                r = arithOp<T, af_sub_t>(r,
                        //sv_arith_helper<T, af_mul_t>(scalar<T>(alpha),
                        sv_arith_helper<T, af_mul_t>(alpha,
                        Ap), r.dims());
            }
            residual_norm = sqrt(abs(dot_host_helper<T>(r, r)));
            SHOW(residual_norm);
            if(residual_norm <= threshold) {
                return x;
            }

            z = matmul(Pinv, r, AF_MAT_NONE, AF_MAT_NONE);

            T beta = dot_host_helper<T>(r, z) / r0z0_dot;
            p = arithOp<T, af_add_t>(z,
                    sv_arith_helper<T, af_mul_t>(
                        //scalar<T>(beta), p), z.dims());
                        beta, p), z.dims());

            iterations++;
        }
        return x;
    }
};

template<typename T>
struct BiCGSTAB {
    static constexpr double abstol = 1e-8;
    static constexpr double reltol = 1e-6;

    vector<double> residual_history;
    bool precondition;
    int max_iterations;
    int iterations;

    Array<T> solve(const common::SparseArray<T> &a, const common::SparseArray<T> &P, const Array<T> &b) {
        //residual_history.clear();
        const int N = b.dims()[0];

        T rho   = scalar<T>(1.0);
        T omega = scalar<T>(1.0);
        T alpha = scalar<T>(1.0);

        Array<T> x  = createValueArray<T>(af::dim4(N), scalar<T>(0));
        Array<T> r0 = copyArray<T>(b);

        T rho_next = dot_host_helper<T>(r0, r0);
        double residual_norm = sqrt(abs(rho_next));

        double threshold = max(BiCGSTAB::abstol, BiCGSTAB::reltol * residual_norm);

        if(residual_norm <= threshold) {
            return x;
        }

        Array<T> r = copyArray<T>(r0);
        Array<T> p = createValueArray<T>(af::dim4(N), scalar<T>(0));
        Array<T> v = createValueArray<T>(af::dim4(N), scalar<T>(0));

        while(iterations < max_iterations && residual_norm > threshold) {
            T beta = (rho_next / rho) * (alpha / omega);

            rho = rho_next; //?

            // p = r + beta(p - omega*v)
            p = arithOp<T, af_add_t>(r,
                    //sv_arith_helper<T, af_mul_t>(scalar<T>(beta),
                    sv_arith_helper<T, af_mul_t>(beta,
                    //arithOp<T, af_sub_t>(p, sv_arith_helper<T, af_mul_t>(scalar<T>(omega), v),
                    arithOp<T, af_sub_t>(p, sv_arith_helper<T, af_mul_t>(omega, v),
                        p.dims())) , r.dims());

            Array<T> q = copyArray<T>(p);
            if(precondition) {
                q = matmul(P, p, AF_MAT_NONE, AF_MAT_NONE);
            }

            v = matmul(a, q, AF_MAT_NONE, AF_MAT_NONE);

            alpha = rho / dot_host_helper<T>(r0, v);
            // s = r - alpha*v
            Array<T> s = arithOp<T, af_sub_t>(r, sv_arith_helper<T, af_mul_t>(alpha, v), r.dims());
            //Array<T> s = arithOp<T, af_sub_t>(r, sv_arith_helper<T, af_mul_t>(scalar<T>(alpha), v), r.dims());

            residual_norm = sqrt(abs(dot_host_helper<T>(s, s)));
            if(residual_norm <= threshold) {
                x = arithOp<T, af_add_t>(x,
                        //sv_arith_helper<T, af_mul_t>(scalar<T>(alpha), q),
                        sv_arith_helper<T, af_mul_t>(alpha, q),
                                         x.dims());
                break;
            }

            Array<T> z = copyArray<T>(s);
            if(precondition) {
                z = matmul(P, s, AF_MAT_NONE, AF_MAT_NONE);
            }

            Array<T> t = matmul(a, z, AF_MAT_NONE, AF_MAT_NONE);
            omega = dot_host_helper(t, s) / dot_host_helper(t, t);
            rho_next = scalar<T>(-1.0) * omega * dot_host_helper(r0, t);

            r = arithOp<T, af_sub_t>(s,
                    //sv_arith_helper<T, af_mul_t>(scalar<T>(omega), t),
                    sv_arith_helper<T, af_mul_t>(omega, t),
                                     s.dims());

            // h = x + alpha*q
            // x = h + omega*z
            x = arithOp<T, af_add_t>(x,
                    arithOp<T, af_add_t>(
                        //sv_arith_helper<T, af_mul_t>(scalar<T>(alpha), q),
                        sv_arith_helper<T, af_mul_t>(alpha, q),
                        //sv_arith_helper<T, af_mul_t>(scalar<T>(omega), z),
                        sv_arith_helper<T, af_mul_t>(omega, z),
                                     p.dims()), x.dims());

            residual_norm = sqrt(abs(dot_host_helper<T>(r, r)));
            //SHOW(residual_norm);
            //residual_history.push_back(residual_norm);
            if(residual_norm <= threshold) {
                break;
            }
            iterations++;
        }

        return x;
    }

};

template<typename T>
Array<T> symmetricSolveSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    dim4 aDims = a.dims();
    size_t M = aDims[0];
    size_t N = aDims[1];

    size_t K = b.dims()[1];

    Array<T> X = createValueArray<T>(af::dim4(N, K), scalar<T>(0));

    CG<T> solver;
    solver.precondition = true;

    // use jacobi preconditioned CG as default solver
    common::SparseArray<T> Pinv = jacobiPreconditionMatrix<T>(a);

    vector<af_seq> vecIndex(4, af_span);
    for(size_t k=0; k<K; ++k) {
        vecIndex[1] = {double(k), double(k), 1};
        Array<T> bvec = createSubArray(b, vecIndex, false);
        Array<T> Xsub = createSubArray(X, vecIndex, false);

        solver.iterations = 0;
        solver.max_iterations = M * 4;
        Array<T> x = solver.solve(a, Pinv, bvec);
        copyArray(Xsub, x);
    }

    return X;
}

template<typename T>
Array<T> generalSolveSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    dim4 aDims = a.dims();
    size_t M = aDims[0];
    size_t N = aDims[1];

    size_t K = b.dims()[1];

    Array<T> X = createValueArray<T>(af::dim4(N, K), scalar<T>(0));

    common::SparseArray<T> Pinv = jacobiPreconditionMatrix<T>(a);

    vector<af_seq> vecIndex(4, af_span);
    for(size_t k=0; k<K; ++k) {
        vecIndex[1] = {double(k), double(k), 1};
        Array<T> bvec = createSubArray(b, vecIndex, false);

        BiCGSTAB<T> solver;
        solver.iterations = 0;
        solver.max_iterations = M * 2;
        solver.precondition = true;

        Array<T> x = solver.solve(a, Pinv, bvec);

        Array<T> Xsub = createSubArray(X, vecIndex, false);
        copyArray(Xsub, x);
    }

    return X;
}

template<typename T>
Array<T> leastSquaresSparse(const common::SparseArray<T> &a, const Array<T> &b) {
    dim4 aDims = a.dims();
    int M = aDims[0];
    int N = aDims[1];

    int K = b.dims()[1];

    // TODO: implement
    AF_ERROR("leastSquares sparse solver currently unsupported",
        AF_ERR_RUNTIME);

    Array<T> B = copyArray<T>(b);
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

    constexpr bool IsComplex =
        std::is_same<T, cl_float2>::value || std::is_same<T, cl_double2>::value;

    if (M == N) {
        if (options & AF_MAT_SYM
         && options & AF_MAT_POSDEF) {
            return symmetricSolveSparse<T>(a, b);
        }
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
