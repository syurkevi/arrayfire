/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <blas.hpp>
#include <convolve.hpp>
#include <handle.hpp>
#include <kernel/convolve.hpp>
#include <err_opencl.hpp>
#include <reorder.hpp>
#include <transpose.hpp>
#include <unwrap.hpp>
#include <vector>
#include <wrap.hpp>

using af::dim4;
using std::vector;

namespace opencl
{

template<typename T, typename accT, dim_t baseDim, bool expand>
Array<T> convolve(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind)
{
    const dim4 sDims    = signal.dims();
    const dim4 fDims    = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for(dim_t d=0; d<4; ++d) {
            if (kind==AF_BATCH_NONE || kind==AF_BATCH_RHS) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==AF_BATCH_RHS) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fDims[i];
        }
    }

    Array<T> out   = createEmptyArray<T>(oDims);
    bool callKernel = true;

    dim_t MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    dim_t MCFL3 = kernel::MAX_CONV3_FILTER_LEN;
    switch(baseDim) {
        case 1: if (fDims[0]>kernel::MAX_CONV1_FILTER_LEN) callKernel = false; break;
        case 2: if ((fDims[0]*fDims[1]) > (MCFL2 * MCFL2)) callKernel = false; break;
        case 3: if ((fDims[0]*fDims[1]*fDims[2]) > (MCFL3 * MCFL3 * MCFL3)) callKernel = false; break;
    }

    if(!callKernel) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nOpenCL N Dimensional Convolution doesn't support %llux%llux%llu kernel\n",
                 fDims[0], fDims[1], fDims[2]);
        OPENCL_NOT_SUPPORTED(errMessage);
    }

    kernel::convolve_nd<T, accT, baseDim, expand>(out, signal, filter, kind);

    return out;
}

#define INSTANTIATE(T, accT) \
    template Array<T> convolve <T, accT, 1, true >(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 1, false>(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 2, true >(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 2, false>(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 3, true >(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 3, false>(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)
INSTANTIATE(ushort ,   float)
INSTANTIATE(short  ,   float)
INSTANTIATE(uintl  ,   float)
INSTANTIATE(intl   ,   float)
#undef INSTANTIATE

template<typename T, typename accT>
Array<T> convolve2_unwrap(const Array<T>& signal,
                          const Array<accT>& filter,
                          const dim4 stride,
                          const dim4 padding,
                          const dim4 dilation)
{
    dim4 sDims = signal.dims();
    dim4 fDims = filter.dims();

    dim_t outputWidth  = 1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) / stride[0];
    dim_t outputHeight = 1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) / stride[1];
    Array<T> out = createValueArray<T>(dim4(outputWidth, outputHeight, fDims[3], sDims[3]), scalar<T>(0));

    const bool retCols = false;
    Array<accT> unwrapped = unwrap_dilated(cast<accT>(signal), fDims[0], fDims[1],
                                                             stride[0], stride[1],
                                                           padding[0], padding[1],
                                                         dilation[0], dilation[1], retCols);

    unwrapped = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped.modDims(dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<accT> collapsedFilter = filter;

    vector<af_seq> flip_index(4);
    af_seq s = {(double)(fDims[0] - 1), 0, -1};
    flip_index[0] = s;
    s = {(double)(fDims[1] - 1), 0, -1};
    flip_index[1] = s;
    flip_index[2] = af_span;
    flip_index[3] = af_span;

    collapsedFilter = createSubArray(collapsedFilter, flip_index);
    collapsedFilter.modDims(dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<accT> res = matmul(collapsedFilter, unwrapped, AF_MAT_TRANS, AF_MAT_NONE);
    res.modDims(dim4(collapsedFilter.dims()[1], out.dims()[0], out.dims()[1], signal.dims()[3]));
    out = cast<T>(reorder(res, dim4(1, 2, 0, 3)));

    return out;
}


template<typename T, typename accT>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    signal.eval();
    filter.eval();

    Array<T> out  = createEmptyArray<T>(dim4());
    out = convolve2_unwrap<T, accT>(signal, filter, stride, padding, dilation);

    return out;
}

#define INSTANTIATE(T, accT)                                                                \
    template Array<T> convolve2<T, accT>(Array<T> const& signal, Array<accT> const& filter, \
                                         const dim4 stride, const dim4 padding, const dim4 dilation);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)
INSTANTIATE(ushort ,   float)
INSTANTIATE(short  ,   float)
INSTANTIATE(uintl  ,   float)
INSTANTIATE(intl   ,   float)
#undef INSTANTIATE

template<typename T, typename accT>
Array<T> conv2DataGradient(const Array<T>& incoming_gradient,
                           const Array<T>& original_signal,
                           const Array<accT>& original_filter,
                           const Array<T>& convolved_output,
                           af::dim4 stride, af::dim4 padding, af::dim4 dilation) {
    original_signal.eval();
    original_filter.eval();

    const dim4 cDims = incoming_gradient.dims();
    const dim4 sDims = original_signal.dims();
    const dim4 fDims = original_filter.dims();

    Array<accT> collapsed_filter = original_filter;

    vector<af_seq> flip_index(4);
    af_seq s = {(double)(fDims[0] - 1), 0, -1};
    flip_index[0] = s;
    s = {(double)(fDims[1] - 1), 0, -1};
    flip_index[1] = s;
    flip_index[2] = af_span;
    flip_index[3] = af_span;

    collapsed_filter = createSubArray(collapsed_filter, flip_index);
    collapsed_filter.modDims(dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<accT> collapsed_gradient = cast<accT>(incoming_gradient);
    collapsed_gradient = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient.modDims(dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    Array<accT> res = matmul(collapsed_gradient, collapsed_filter, AF_MAT_NONE, AF_MAT_TRANS);
    res.modDims(dim4(res.dims()[0]/sDims[3], sDims[3], fDims[0] * fDims[1], sDims[2]));
    res = reorder(res, dim4(0, 2, 3, 1));

    const bool retCols = false;
    res = wrap_dilated(res, sDims[0], sDims[1],
                            fDims[0], fDims[1],
                           stride[0], stride[1],
                          padding[0], padding[1],
                         dilation[0], dilation[1], retCols);

    return cast<T>(res);
}

template<typename T, typename accT>
Array<T> conv2FilterGradient(const Array<T>& incoming_gradient,
                             const Array<T>& original_signal,
                             const Array<accT>& original_filter,
                             const Array<T>& convolved_output,
                             af::dim4 stride, af::dim4 padding, af::dim4 dilation) {

    original_signal.eval();
    original_filter.eval();

    const dim4 cDims = incoming_gradient.dims();
    const dim4 sDims = original_signal.dims();
    const dim4 fDims = original_filter.dims();

    const bool retCols = false;
    Array<accT> unwrapped = unwrap_dilated(cast<accT>(original_signal), fDims[0], fDims[1],
                                                                       stride[0], stride[1],
                                                                      padding[0], padding[1],
                                                                     dilation[0], dilation[1], retCols);

    unwrapped = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped.modDims(dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<accT> collapsed_gradient = cast<accT>(incoming_gradient);
    collapsed_gradient = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient.modDims(dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    Array<accT> res = matmul(unwrapped, collapsed_gradient, AF_MAT_NONE, AF_MAT_NONE);
    res.modDims(dim4(fDims[0], fDims[1], fDims[2], fDims[3]));

    vector<af_seq> flip_index(4);
    af_seq s = {(double)(fDims[0] - 1), 0, -1};
    flip_index[0] = s;
    s = {(double)(fDims[1] - 1), 0, -1};
    flip_index[1] = s;
    flip_index[2] = af_span;
    flip_index[3] = af_span;

    return cast<T>(createSubArray(res, flip_index));
}

#define INSTANTIATE(T, accT)                                                          \
    template Array<T> conv2DataGradient<T, accT>(Array<T> const& incoming_gradient,   \
                                             Array<T> const& original_signal,         \
                                             Array<accT> const& original_filter,      \
                                             Array<T> const& convolved_output,        \
                                             const dim4 stride,                       \
                                             const dim4 padding,                      \
                                             const dim4 dilation);                    \
    template Array<T> conv2FilterGradient<T, accT>(Array<T> const& incoming_gradient, \
                                             Array<T> const& original_signal,         \
                                             Array<accT> const& original_filter,      \
                                             Array<T> const& convolved_output,        \
                                             const dim4 stride,                       \
                                             const dim4 padding,                      \
                                             const dim4 dilation);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)
INSTANTIATE(ushort ,   float)
INSTANTIATE(short  ,   float)
INSTANTIATE(uintl  ,   float)
INSTANTIATE(intl   ,   float)
#undef INSTANTIATE

}
