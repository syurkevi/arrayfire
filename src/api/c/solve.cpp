/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <solve.hpp>
#include <sparse_solve.hpp>
#include <af/array.h>
#include <af/defines.h>
#include <af/lapack.h>
#include <sparse_handle.hpp>

using af::dim4;
using common::SparseArrayBase;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::solveLU;

template<typename T>
static inline af_array solve(const af_array a, const af_array b,
                             const af_mat_prop options) {
    return getHandle(solve<T>(getArray<T>(a), getArray<T>(b), options));
}

template<typename T>
static inline af_array sparse_solve(const af_array a, const af_array b,
                                    const af_mat_prop options) {
    return getHandle(sparse_solve<T>(getSparseArray<T>(a), getArray<T>(b), options));
}

af_err af_sparse_solve(af_array* out, const af_array a, const af_array b,
                       const af_mat_prop options) {
    try {
        const SparseArrayBase arrBase = getSparseArrayBase(a);
        const ArrayInfo &bInfo        = getInfo(b);

        ARG_ASSERT(1, arrBase.isSparse() == true);
        ARG_ASSERT(2, bInfo.isSparse() == false);

        af_dtype a_type = arrBase.getType();
        af_dtype b_type = bInfo.getType();

        //TODO: support others?
        ARG_ASSERT(1, arrBase.getStorage() == AF_STORAGE_CSR);

        dim4 adims = arrBase.dims();
        dim4 bdims = bInfo.dims();

        ARG_ASSERT(1, arrBase.isFloating());  // Only floating and complex types
        ARG_ASSERT(2, bInfo.isFloating());  // Only floating and complex types

        TYPE_ASSERT(a_type == b_type);

        DIM_ASSERT(1, bdims[0] == adims[0]);
        DIM_ASSERT(1, bdims[2] == adims[2]);
        DIM_ASSERT(1, bdims[3] == adims[3]);

        if (arrBase.ndims() == 0 || bInfo.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, a_type);
        }

        bool is_triangle_solve =
            (options & AF_MAT_LOWER) || (options & AF_MAT_UPPER);

        if (options != AF_MAT_NONE && !is_triangle_solve) {
            AF_ERROR("Using this property is not yet supported in solve",
                     AF_ERR_NOT_SUPPORTED);
        }

        if (is_triangle_solve) {
            DIM_ASSERT(1, adims[0] == adims[1]);
            if ((options & AF_MAT_TRANS || options & AF_MAT_CTRANS)) {
                AF_ERROR("Using AF_MAT_TRANS is not yet supported in solve",
                         AF_ERR_NOT_SUPPORTED);
            }
        }

        af_array output = 0;
        switch (a_type) {
            case f32:
                output = sparse_solve<float>(a, b, options);
                break;
            case c32:
                output = sparse_solve<cfloat>(a, b, options);
                break;
            case f64:
                output = sparse_solve<double>(a, b, options);
                break;
            case c64:
                output = sparse_solve<cdouble>(a, b, options);
                break;
            default: TYPE_ERROR(1, a_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;


}

af_err af_solve(af_array* out, const af_array a, const af_array b,
                const af_mat_prop options) {
    try {
        const ArrayInfo &a_info = getInfo(a, false, true);
        const ArrayInfo &b_info = getInfo(b, true, true);

        if(a_info.isSparse()) {
            return af_sparse_solve(out, a, b, options);
        }

        af_dtype a_type = a_info.getType();
        af_dtype b_type = b_info.getType();

        dim4 adims = a_info.dims();
        dim4 bdims = b_info.dims();

        ARG_ASSERT(1, a_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(2, b_info.isFloating());  // Only floating and complex types

        TYPE_ASSERT(a_type == b_type);

        DIM_ASSERT(1, bdims[0] == adims[0]);
        DIM_ASSERT(1, bdims[2] == adims[2]);
        DIM_ASSERT(1, bdims[3] == adims[3]);

        if (a_info.ndims() == 0 || b_info.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, a_type);
        }

        bool is_triangle_solve =
            (options & AF_MAT_LOWER) || (options & AF_MAT_UPPER);

        if (options != AF_MAT_NONE && !is_triangle_solve) {
            AF_ERROR("Using this property is not yet supported in solve",
                     AF_ERR_NOT_SUPPORTED);
        }

        if (is_triangle_solve) {
            DIM_ASSERT(1, adims[0] == adims[1]);
            if ((options & AF_MAT_TRANS || options & AF_MAT_CTRANS)) {
                AF_ERROR("Using AF_MAT_TRANS is not yet supported in solve",
                         AF_ERR_NOT_SUPPORTED);
            }
        }

        af_array output;

        switch (a_type) {
            case f32: output = solve<float>(a, b, options); break;
            case f64: output = solve<double>(a, b, options); break;
            case c32: output = solve<cfloat>(a, b, options); break;
            case c64: output = solve<cdouble>(a, b, options); break;
            default: TYPE_ERROR(1, a_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline af_array solve_lu(const af_array a, const af_array pivot,
                                const af_array b, const af_mat_prop options) {
    return getHandle(solveLU<T>(getArray<T>(a), getArray<int>(pivot),
                                getArray<T>(b), options));
}

af_err af_solve_lu(af_array* out, const af_array a, const af_array piv,
                   const af_array b, const af_mat_prop options) {
    try {
        const ArrayInfo& a_info = getInfo(a);
        const ArrayInfo& b_info = getInfo(b);

        if (a_info.ndims() > 2 || b_info.ndims() > 2) {
            AF_ERROR("solveLU can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype a_type = a_info.getType();
        af_dtype b_type = b_info.getType();

        dim4 adims = a_info.dims();
        dim4 bdims = b_info.dims();
        if (a_info.ndims() == 0 || b_info.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, a_type);
        }

        ARG_ASSERT(1, a_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(2, b_info.isFloating());  // Only floating and complex types

        TYPE_ASSERT(a_type == b_type);

        DIM_ASSERT(1, adims[0] == adims[1]);
        DIM_ASSERT(1, bdims[0] == adims[0]);
        DIM_ASSERT(1, bdims[2] == adims[2]);
        DIM_ASSERT(1, bdims[3] == adims[3]);

        if (options != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in solveLU",
                     AF_ERR_NOT_SUPPORTED);
        }

        af_array output;

        switch (a_type) {
            case f32: output = solve_lu<float>(a, piv, b, options); break;
            case f64: output = solve_lu<double>(a, piv, b, options); break;
            case c32: output = solve_lu<cfloat>(a, piv, b, options); break;
            case c64: output = solve_lu<cdouble>(a, piv, b, options); break;
            default: TYPE_ERROR(1, a_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
