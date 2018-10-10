/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/ml.h>
#include <af/seq.h>

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <pool.hpp>

using af::dim4;
using namespace detail;

template<typename T>
af_array poolCall(const Array<T>& in,
                  const dim_t pool_width, const dim_t pool_height,
                  const dim_t padding_width, const dim_t padding_height,
                  const int stride_width, const int stride_height,
                  af_pooling_type pool_type)
{
    return getHandle(detail::pool2<T>(in, pool_width, pool_height,
                                      padding_width, padding_height,
                                      stride_width, stride_height,
                                      pool_type));
}

af_err af_pool2(af_array *out, const af_array in,
             const dim_t pool_width, const dim_t pool_height,
             const dim_t padding_width, const dim_t padding_height,
             const dim_t stride_width, const dim_t stride_height,
             af_pooling_type pool_type)
{
    try {
        const ArrayInfo& info = getInfo(in);
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims() >= 2));
        DIM_ASSERT(1, (dims[2] == 3 || dims[2] == 1)); //# channels should be 1 or 3

        af_array output;

        af_dtype type  = info.getType();
        switch(type) {
            case f32:
                output = poolCall<float> (getArray<float>(in), pool_width, pool_height,
                                          padding_width, padding_height, stride_width, stride_height,
                                          pool_type); break;
            case f64:
                output = poolCall<double>(getArray<double>(in), pool_width, pool_height,
                                          padding_width, padding_height, stride_width, stride_height,
                                          pool_type); break;
            case s32:
                output = poolCall<int>(getArray<int>(in), pool_width, pool_height,
                                       padding_width, padding_height, stride_width, stride_height,
                                       pool_type); break;
            case u32:
                output = poolCall<uint>(getArray<uint>(in), pool_width, pool_height,
                                        padding_width, padding_height, stride_width, stride_height,
                                        pool_type); break;
            case s16:
                output = poolCall<short>(getArray<short>(in), pool_width, pool_height,
                                        padding_width, padding_height, stride_width, stride_height,
                                        pool_type); break;
            case u16:
                output = poolCall<ushort>(getArray<ushort>(in), pool_width, pool_height,
                                          padding_width, padding_height, stride_width, stride_height,
                                          pool_type); break;
            case u8:
                output = poolCall<uchar>(getArray<uchar>(in), pool_width, pool_height,
                                         padding_width, padding_height, stride_width, stride_height,
                                         pool_type); break;
            default : TYPE_ERROR(1, type);
        }
        // output array is pooled array
        std::swap(output, *out);
    }
    CATCHALL;

    return AF_SUCCESS;
}
