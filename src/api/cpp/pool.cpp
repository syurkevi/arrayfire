/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/ml.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array pool2(const array& in,
            const dim_t pool_width, const dim_t pool_height,
            const dim_t padding_width, const dim_t padding_height,
            const dim_t stride_width, const dim_t stride_height,
            af_pooling_type pool_type)
{
    af_array temp_out = 0;
    AF_THROW(af_pool2(&temp_out, in.get(),
                       pool_width, pool_height,
                       padding_width, padding_height,
                       stride_width, stride_height,
                       pool_type));
    return array(temp_out);
}

array pool2Gradient(const array &incoming_gradient,
                    const array& original_input,const array& pooled_output,
                    const dim_t pool_width,     const dim_t pool_height,
                    const dim_t padding_width,  const dim_t padding_height,
                    const dim_t stride_width,   const dim_t stride_height,
                    af_pooling_type pool_type) {

    af_array temp_out = 0;
    AF_THROW(af_pool2Gradient(&temp_out,
                              incoming_gradient.get(),
                              original_input.get(),
                              pooled_output.get(),
                              pool_width, pool_height,
                              padding_width, padding_height,
                              stride_width, stride_height,
                              pool_type));
    return array(temp_out);

}

}
