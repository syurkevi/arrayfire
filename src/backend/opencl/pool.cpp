/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <pool.hpp>
#include <err_opencl.hpp>
//#include <kernel/pool.hpp>

using af::dim4;

namespace opencl
{
Array<float> pool2(const Array<float>& in,
                   const dim_t pool_width, const dim_t pool_height,
                   const dim_t padding_width, const dim_t padding_height,
                   const dim_t stride_width, const dim_t stride_height,
                   af_pooling_type pool_type)
{
    Array<float> out = createValueArray<float>(in.dims(), 0);

    return out;
}

}
