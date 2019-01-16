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
#include <arith.hpp>
#include <assign.hpp>
#include <handle.hpp>
#include <ireduce.hpp>
#include <range.hpp>
#include <reduce.hpp>
#include <unwrap.hpp>
#include <wrap.hpp>

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
    Array<T> unwrapped = unwrap(original_input, pool_width, pool_height,
                                   stride_width, stride_height,
                                   padding_width, padding_height, true);

    dim4 udims = unwrapped.dims();
    dim4 r_udims = udims;
    r_udims[0] = 1;
    Array<T> max_unwrapped = createEmptyArray<T>(r_udims);
    Array<unsigned int> idx_maxunwrapped = createEmptyArray<unsigned int>(r_udims);
    ireduce<af_max_t, T>(max_unwrapped, idx_maxunwrapped, unwrapped, 0);

    Array<T> out  = createValueArray<T>(original_input.dims(), 0);
    Array<T> ones = createValueArray<T>(original_input.dims(), 1);

    //figure out linear indices of maximum values in unwrapped array
    dim4 mdims = idx_maxunwrapped.dims();
    idx_maxunwrapped.modDims(af::dim4(mdims[0] * mdims[1], 1, mdims[2], mdims[3]));
    Array<unsigned int> l_idx = idx_maxunwrapped;
    dim4 idxdims(udims[1], 1, mdims[2], mdims[3]);
    l_idx = arithOp<unsigned int, af_add_t>(arithOp<unsigned int, af_mul_t>(range<unsigned int>(idxdims), createValueArray<unsigned int>(idxdims, udims[0]), idxdims), l_idx, idxdims);
    l_idx = arithOp<unsigned int, af_add_t>(arithOp<unsigned int, af_mul_t>(range<unsigned int>(idxdims, 2), createValueArray<unsigned int>(idxdims, udims[0] * udims[1]), idxdims), l_idx, idxdims);
    l_idx = arithOp<unsigned int, af_add_t>(arithOp<unsigned int, af_mul_t>(range<unsigned int>(idxdims, 3), createValueArray<unsigned int>(idxdims, udims[0] * udims[1] * udims[2]), idxdims), l_idx, idxdims);

    Array<T> unwrapped_grad = createValueArray<T>(unwrapped.dims(), scalar<T>(0));
    Array<T> grad_output = copyArray(incoming_gradient);
    grad_output.eval();

    grad_output.modDims(dim4(grad_output.elements()));
    af_index_t idx;
    idx.idx.arr = getHandle(l_idx);
    idx.isSeq = false;
    idx.isBatch = false;

    af_index_t sp;
    sp.idx.seq = af_span;
    sp.isSeq = true;
    sp.isBatch = false;

    af_index_t idxrs[4] = {idx, sp, sp, sp};

    unwrapped_grad.modDims(dim4(unwrapped_grad.elements()));
    assign(unwrapped_grad, idxrs, grad_output);
    unwrapped_grad.modDims(unwrapped.dims());

    dim4 d = original_input.dims();
    out = wrap(unwrapped_grad, d[0], d[1], pool_width, pool_height, stride_width, stride_height, padding_width, padding_height, true);

    return out;
}

#define INSTANTIATE(T)                                                                      \
    template Array<T> pool2Gradient<T>(const Array<T>& incoming_gradient,                   \
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
