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
#include <assign.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <Param.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <reduce.hpp>
#include <ireduce.hpp>
#include <unwrap.hpp>
#include <vector>

namespace cpu
{

template<typename T>
Array<T> pool2(const Array<T>& in,
               const dim_t pool_width,    const dim_t pool_height,
               const dim_t padding_width, const dim_t padding_height,
               const dim_t stride_width,  const dim_t stride_height,
                   af_pooling_type pool_type)
{
    in.eval();

    Array<T> unwrapped = unwrap(in, pool_width, pool_height,
                                   stride_width, stride_height,
                                   padding_width, padding_height, true);

    dim4 udims = unwrapped.dims();
    udims[0] -= 1;
    Array<T> max_unwrapped = reduce<af_max_t, T, T>(unwrapped, 0);

    dim_t outputWidth  = 1 + (in.dims()[0] + 2 * padding_width - pool_width)   / stride_width;
    dim_t outputHeight = 1 + (in.dims()[1] + 2 * padding_height - pool_height) / stride_height;

    max_unwrapped.modDims(dim4(outputWidth, outputHeight, in.dims()[2], in.dims()[3]));
    Array<T> out = createValueArray<T>(max_unwrapped.dims(), 0);
    out = max_unwrapped;

    return out;
}


#define INSTANTIATE(T)                                                  \
    template Array<T> pool2<T>(const Array<T> &in,                      \
               const dim_t pool_width,    const dim_t pool_height,      \
               const dim_t padding_width, const dim_t padding_height,   \
               const dim_t stride_width,  const dim_t stride_height,    \
               af_pooling_type pool_type);


INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

#undef INSTANTIATE


template<typename T>
Array<T> pool2Gradient(const Array<T>& incoming_gradient,
                       const Array<T>& original_input,
                       const Array<T>& pooled_output,
                       const dim_t pool_width,    const dim_t pool_height,
                       const dim_t padding_width, const dim_t padding_height,
                       const dim_t stride_width,  const dim_t stride_height,
                       af_pooling_type pool_type)
{
    original_input.eval();

    Array<T> unwrapped = unwrap(original_input, pool_width, pool_height,
                                   stride_width, stride_height,
                                   padding_width, padding_height, true);

    dim4 udims = unwrapped.dims();
    udims[0] -= 1;
    Array<T> max_unwrapped = createEmptyArray<T>(udims);
    Array<unsigned int> idx_maxunwrapped = createEmptyArray<unsigned int>(udims);
    ireduce<af_max_t, T>(max_unwrapped, idx_maxunwrapped, unwrapped, 0);

    //dim_t outputWidth  = 1 + (in.dims()[0] + 2*padding_width - pool_width)/stride_width;
    //dim_t outputHeight = 1 + (in.dims()[1] + 2*padding_height - pool_height)/stride_height;

    Array<T> out   = createValueArray<T>(original_input.dims(), 0);
    Array<T> ones  = createValueArray<T>(original_input.dims(), 1);
    af_index_t idxrs[] = { getHandle(idx_maxunwrapped)};
    assign<T>(out, idxrs, ones);
    af_print_array(getHandle(out));

    //out


    return out;
}


#define INSTANTIATE(T)                                                                      \
    template Array<T> pool2Gradient(const Array<T>& incoming_gradient,                      \
                                    const Array<T>& original_input,                         \
                                    const Array<T>& pooled_output,                          \
                                    const dim_t pool_width,    const dim_t pool_height,     \
                                    const dim_t padding_width, const dim_t padding_height,  \
                                    const dim_t stride_width,  const dim_t stride_height,   \
                                    af_pooling_type pool_type);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

#undef INSTANTIATE

}
