/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace cpu
{
template<typename T>
Array<T> pool2(const Array<T>& in,
               const dim_t pool_width, const dim_t pool_height,
               const dim_t padding_width, const dim_t padding_height,
               const dim_t stride_width, const dim_t stride_height,
               af_pooling_type pool_type);

template<typename T>
Array<T> pool2Gradient(const Array<T>& incoming_gradient,
                       const Array<T>& original_input,
                       const Array<T>& pooled_output,
                       const dim_t pool_width,    const dim_t pool_height,
                       const dim_t padding_width, const dim_t padding_height,
                       const dim_t stride_width,  const dim_t stride_height,
                       af_pooling_type pool_type);

}
