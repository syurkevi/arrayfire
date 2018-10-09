/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <pool.hpp>

#include <Array.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <Param.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <reduce.hpp>
#include <wrap.hpp>
#include <unwrap.hpp>

namespace cpu
{
Array<float> pool2(const Array<float>& in,
                   const dim_t pool_width, const dim_t pool_height,
                   const dim_t padding_width, const dim_t padding_height,
                   const dim_t stride_width, const dim_t stride_height,
                   af_pooling_type pool_type)
{
    in.eval();


    Array<float> unwrapped = unwrap(in, pool_width, pool_height,
                                   stride_width, stride_height,
                                   padding_width, padding_height, true);

    dim4 udims = unwrapped.dims();
    udims[0] -= 1;
    Array<float> max_unwrapped = reduce<af_max_t, float, float>(unwrapped, 0);


    dim_t outputWidth = 1 + (in.dims()[0] + 2*padding_width - pool_width)/stride_width;
    dim_t outputHeight = 1 + (in.dims()[1] + 2*padding_height - pool_height)/stride_height;

    max_unwrapped.modDims(dim4(outputWidth, outputHeight, in.dims()[2], in.dims()[3]));
    Array<float> out = createValueArray<float>(max_unwrapped.dims(), 0);
    out = max_unwrapped;

    return out;
}

}
