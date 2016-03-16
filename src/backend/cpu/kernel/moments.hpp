/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <Array.hpp>
#include <utility.hpp>
#include <math.hpp>

namespace cpu
{
namespace kernel
{


template<typename T, af_moment MM>
struct moments_op
{
    T operator()(T const * const in,  af::dim4 const & idims,
                 dim_t const iElems, af::dim4 const & istrides, bool const pBatch,
                 dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        return;
    }
};

template<typename T>
struct moments_op<T, M00>
{
    T operator()(T const * const in,  af::dim4 const & idims,
                 dim_t const iElems, af::dim4 const & istrides, bool const pBatch,
                 dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t mId = idy * istrides[1] + idx;
        if(pBatch) {
            mId += idw * istrides[3] + idz * istrides[2];
        }

        return in[mId];
    }
};

template<typename T>
struct moments_op<T, M01>
{
    T operator()(T const * const in,  af::dim4 const & idims,
                 dim_t const iElems, af::dim4 const & istrides, bool const pBatch,
                 dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t mId = idy * istrides[1] + idx;
        if(pBatch) {
            mId += idw * istrides[3] + idz * istrides[2];
        }

        return idx * in[mId];
    }
};

template<typename T>
struct moments_op<T, M10>
{
    T operator()(T const * const in,  af::dim4 const & idims,
                 dim_t const iElems, af::dim4 const & istrides, bool const pBatch,
                 dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t mId = idy * istrides[1] + idx;
        if(pBatch) {
            mId += idw * istrides[3] + idz * istrides[2];
        }

        return idy * in[mId];
    }
};

template<typename T>
struct moments_op<T, M11>
{
    T operator()(T const * const in,  af::dim4 const & idims,
                 dim_t const iElems, af::dim4 const & istrides, bool const pBatch,
                 dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t mId = idy * istrides[1] + idx;
        if(pBatch) {
            mId += idw * istrides[3] + idz * istrides[2];
        }

        return idx * idy * in[mId];
    }
};

template<typename T, af_moment Method>
void moments(T* output, Array<T> const input)
{
    T const * const in       = input.get();
    af::dim4  const idims    = input.dims();
    af::dim4  const istrides = input.strides();
    dim_t     const iElems   = input.elements();

    moments_op<T, Method> op;
    bool pBatch = !(idims[2] == 1 && idims[3] == 1);

    T val = scalar<T>(0);
    for(dim_t w = 0; w < idims[3]; w++) {
        for(dim_t z = 0; z < idims[2]; z++) {
            for(dim_t y = 0; y < idims[1]; y++) {
                for(dim_t x = 0; x < idims[0]; x++) {
                    val += op(in, idims, iElems, istrides, pBatch, x, y, z, w);
                }
            }
        }
    }
    *output = val;
}


}
}
