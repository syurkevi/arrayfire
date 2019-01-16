/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <pool.hpp>

#include <af/dim4.hpp>
#include <af/index.h>
#include <Array.hpp>
#include <arith.hpp>
#include <assign.hpp>
#include <cudnn.hpp>
#include <err_cuda.hpp>
#include <handle.hpp>
#include <ireduce.hpp>
#include <platform.hpp>
#include <range.hpp>
#include <reduce.hpp>
#include <unwrap.hpp>
#include <wrap.hpp>

using af::dim4;

/////
/// cuDNN helper functions
/////
template <typename T>
static cudnnDataType_t getCudnnDataType();
template <typename T>
static cudnnDataType_t getCudnnDataType() {
    static_assert("no matching cudnnDataType_t");
    return CUDNN_DATA_FLOAT;
}
template <>
cudnnDataType_t getCudnnDataType<float>() {
    return CUDNN_DATA_FLOAT;
}
template <>
cudnnDataType_t getCudnnDataType<double>() {
    return CUDNN_DATA_DOUBLE;
}
template <>
cudnnDataType_t getCudnnDataType<int>() {
    return CUDNN_DATA_INT32;
}
template <>
cudnnDataType_t getCudnnDataType<unsigned char>() {
    return CUDNN_DATA_UINT8;
}

template <typename T>
static bool supported_cudnn_type();
template <typename T>
static bool supported_cudnn_type() {
    if (std::is_same<float, T>::value || std::is_same<double, T>::value ||
        std::is_same<char, T>::value) {
        return true;
    }
    return false;
}

namespace cuda {

template <typename T>
Array<T> pool2_cudnn(const Array<T>& in, const dim_t pool_width,
                     const dim_t pool_height, const dim_t padding_width,
                     const dim_t padding_height, const dim_t stride_width,
                     const dim_t stride_height, af_pooling_type pool_type) {
    dim4 idims = in.dims();

    cudnnHandle_t cudnn = nnHandle();

    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnPoolingDescriptor_t pooling_descriptor;

    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));

    const int n = idims[3];
    const int c = idims[2];
    const int h = idims[1];
    const int w = idims[0];

    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, n, c, h, w));

    const int pool_dims         = 2;
    int window_dims[pool_dims]  = {(int)pool_width, (int)pool_height};
    int padding_dims[pool_dims] = {(int)padding_width, (int)padding_height};
    int stride_dims[pool_dims]  = {(int)stride_width, (int)stride_height};

    CUDNN_CHECK(cudnnSetPoolingNdDescriptor(
        pooling_descriptor, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pool_dims,
        window_dims, padding_dims, stride_dims));

    const int tensorDims = 4;
    int pooled_output_dim[tensorDims];
    CUDNN_CHECK(cudnnGetPoolingNdForwardOutputDim(
        pooling_descriptor, input_descriptor, tensorDims, pooled_output_dim));

    const int n_out = pooled_output_dim[0];
    const int c_out = pooled_output_dim[1];
    const int h_out = pooled_output_dim[2];
    const int w_out = pooled_output_dim[3];

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, n_out, c_out, h_out,
                                           w_out));

    dim4 odims(w_out, h_out, c_out, n_out);
    Array<T> out = createValueArray<T>(odims, 0);

    T alpha = scalar<T>(1.0);
    T beta  = scalar<T>(0.0);
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_descriptor, &alpha,
                                    input_descriptor, in.device(), &beta,
                                    output_descriptor, out.device()));

    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_descriptor));

    return out;
}

template <typename T>
Array<T> pool2_native(const Array<T>& in, const dim_t pool_width,
                      const dim_t pool_height, const dim_t padding_width,
                      const dim_t padding_height, const dim_t stride_width,
                      const dim_t stride_height, af_pooling_type pool_type) {
    in.eval();

    Array<T> unwrapped =
        unwrap(in, pool_width, pool_height, stride_width, stride_height,
               padding_width, padding_height, true);

    dim4 udims = unwrapped.dims();
    udims[0] -= 1;
    Array<T> max_unwrapped = reduce<af_max_t, T, T>(unwrapped, 0);

    dim_t outputWidth =
        1 + (in.dims()[0] + 2 * padding_width - pool_width) / stride_width;
    dim_t outputHeight =
        1 + (in.dims()[1] + 2 * padding_height - pool_height) / stride_height;

    max_unwrapped.modDims(
        dim4(outputWidth, outputHeight, in.dims()[2], in.dims()[3]));
    Array<T> out = createValueArray<T>(max_unwrapped.dims(), 0);
    out          = max_unwrapped;

    return out;
}

template <typename T>
Array<T> pool2(const Array<T>& in, const dim_t pool_width,
               const dim_t pool_height, const dim_t padding_width,
               const dim_t padding_height, const dim_t stride_width,
               const dim_t stride_height, af_pooling_type pool_type) {
    if (supported_cudnn_type<T>()) {
        return pool2_cudnn(in, pool_width, pool_height, padding_width,
                           padding_height, stride_width, stride_height,
                           pool_type);
    } else
        return pool2_native(in, pool_width, pool_height, padding_width,
                            padding_height, stride_width, stride_height,
                            pool_type);
}

#define INSTANTIATE(T)                                                       \
    template Array<T> pool2<T>(                                              \
        const Array<T>& in, const dim_t pool_width, const dim_t pool_height, \
        const dim_t padding_width, const dim_t padding_height,               \
        const dim_t stride_width, const dim_t stride_height,                 \
        af_pooling_type pool_type);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

#undef INSTANTIATE

template <typename T>
Array<T> pool2Grad_native(const Array<T>& incoming_gradient,
                          const Array<T>& original_input,
                          const Array<T>& pooled_output, const dim_t pool_width,
                          const dim_t pool_height, const dim_t padding_width,
                          const dim_t padding_height, const int stride_width,
                          const int stride_height, af_pooling_type pool_type) {
    Array<T> unwrapped =
        unwrap(original_input, pool_width, pool_height, stride_width,
               stride_height, padding_width, padding_height, true);

    dim4 udims             = unwrapped.dims();
    dim4 r_udims           = udims;
    r_udims[0]             = 1;
    Array<T> max_unwrapped = createEmptyArray<T>(r_udims);
    Array<unsigned int> idx_maxunwrapped =
        createEmptyArray<unsigned int>(r_udims);
    ireduce<af_max_t, T>(max_unwrapped, idx_maxunwrapped, unwrapped, 0);

    Array<T> out  = createValueArray<T>(original_input.dims(), 0);
    Array<T> ones = createValueArray<T>(original_input.dims(), 1);

    // figure out linear indices of maximum values in unwrapped array
    dim4 mdims = idx_maxunwrapped.dims();
    idx_maxunwrapped.modDims(
        af::dim4(mdims[0] * mdims[1], 1, mdims[2], mdims[3]));
    Array<unsigned int> l_idx = idx_maxunwrapped;
    dim4 idxdims(udims[1], 1, mdims[2], mdims[3]);
    l_idx = arithOp<unsigned int, af_add_t>(
        arithOp<unsigned int, af_mul_t>(
            range<unsigned int>(idxdims),
            createValueArray<unsigned int>(idxdims, udims[0]), idxdims),
        l_idx, idxdims);
    l_idx = arithOp<unsigned int, af_add_t>(
        arithOp<unsigned int, af_mul_t>(
            range<unsigned int>(idxdims, 2),
            createValueArray<unsigned int>(idxdims, udims[0] * udims[1]),
            idxdims),
        l_idx, idxdims);
    l_idx = arithOp<unsigned int, af_add_t>(
        arithOp<unsigned int, af_mul_t>(
            range<unsigned int>(idxdims, 3),
            createValueArray<unsigned int>(idxdims,
                                           udims[0] * udims[1] * udims[2]),
            idxdims),
        l_idx, idxdims);

    Array<T> unwrapped_grad =
        createValueArray<T>(unwrapped.dims(), scalar<T>(0));
    Array<T> grad_output = copyArray(incoming_gradient);
    grad_output.eval();

    grad_output.modDims(dim4(grad_output.elements()));
    af_index_t idx;
    idx.idx.arr = getHandle(l_idx);
    idx.isSeq   = false;
    idx.isBatch = false;

    af_index_t sp;
    sp.idx.seq = af_span;
    sp.isSeq   = true;
    sp.isBatch = false;

    af_index_t idxrs[4] = {idx, sp, sp, sp};

    unwrapped_grad.modDims(dim4(unwrapped_grad.elements()));
    assign(unwrapped_grad, idxrs, grad_output);
    unwrapped_grad.modDims(unwrapped.dims());

    dim4 d = original_input.dims();
    out =
        wrap(unwrapped_grad, d[0], d[1], pool_width, pool_height, stride_width,
             stride_height, padding_width, padding_height, true);

    return out;
}

template <typename T>
Array<T> pool2Grad_cudnn(const Array<T>& incoming_gradient,
                         const Array<T>& original_input,
                         const Array<T>& pooled_output, const dim_t pool_width,
                         const dim_t pool_height, const dim_t padding_width,
                         const dim_t padding_height, const int stride_width,
                         const int stride_height, af_pooling_type pool_type) {

    cudnnHandle_t cudnn = nnHandle();

    dim4 gdims = incoming_gradient.dims();
    const int n_grad = gdims[3];
    const int c_grad = gdims[2];
    const int h_grad = gdims[1];
    const int w_grad = gdims[0];

    dim4 idims = original_input.dims();
    const int n_in = idims[3];
    const int c_in = idims[2];
    const int h_in = idims[1];
    const int w_in = idims[0];

    // create original_input descriptor
    cudnnTensorDescriptor_t original_input_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&original_input_desc));
    cudnnDataType_t dtype = getCudnnDataType<T>();
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(original_input_desc, CUDNN_TENSOR_NCHW,
                                           dtype, n_in, c_in, h_in, w_in));

    // create pooled_output descriptor
    cudnnTensorDescriptor_t pooled_output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&pooled_output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(pooled_output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           dtype,
                                           n_grad,
                                           c_grad,
                                           h_grad,
                                           w_grad));

    // create incoming gradient descriptor
    cudnnTensorDescriptor_t ingrad_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&ingrad_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(ingrad_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           dtype,
                                           n_grad,
                                           c_grad,
                                           h_grad,
                                           w_grad));


    // prepare output array and descriptor
    Array<T> grad = createValueArray<T>(idims, 0);

    cudnnTensorDescriptor_t grad_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&grad_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(grad_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           dtype, n_in, c_in, h_in, w_in));


    // create pooling descriptor
    const int pool_dims         = 2;
    int window_dims[pool_dims]  = {(int)pool_width, (int)pool_height};
    int padding_dims[pool_dims] = {(int)padding_width, (int)padding_height};
    int stride_dims[pool_dims]  = {(int)stride_width, (int)stride_height};

    cudnnPoolingDescriptor_t pooling_descriptor;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    CUDNN_CHECK(cudnnSetPoolingNdDescriptor(pooling_descriptor,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_PROPAGATE_NAN,
                                            pool_dims,
                                            window_dims,
                                            padding_dims,
                                            stride_dims));

    // perform gradient pooling calculation
    T alpha = scalar<T>(1.0);
    T beta  = scalar<T>(0.0);

    CUDNN_CHECK(cudnnPoolingBackward(cudnn,
                                     pooling_descriptor,
                                     &alpha,
                                     pooled_output_desc,
                                     pooled_output.device(),
                                     ingrad_descriptor,
                                     incoming_gradient.device(),
                                     original_input_desc,
                                     original_input.device(),
                                     &beta,
                                     grad_descriptor,
                                     grad.device()
                                     ));

    // destroy descriptors
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(original_input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(pooled_output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(grad_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(ingrad_descriptor));

    return grad;
}

template <typename T>
Array<T> pool2Gradient(const Array<T>& incoming_gradient,
                       const Array<T>& original_input,
                       const Array<T>& pooled_output, const dim_t pool_width,
                       const dim_t pool_height, const dim_t padding_width,
                       const dim_t padding_height, const dim_t stride_width,
                       const dim_t stride_height, af_pooling_type pool_type) {
    if (supported_cudnn_type<T>()) {
        return pool2Grad_cudnn(incoming_gradient, original_input, pooled_output,
                               pool_width, pool_height, padding_width,
                               padding_height, stride_width, stride_height,
                               pool_type);
    } else {
        return pool2Grad_native(incoming_gradient, original_input,
                                pooled_output, pool_width, pool_height,
                                padding_width, padding_height, stride_width,
                                stride_height, pool_type);
    }
}

#define INSTANTIATE(T)                                                     \
    template Array<T> pool2Gradient(                                       \
        const Array<T>& incoming_gradient, const Array<T>& original_input, \
        const Array<T>& pooled_output, const dim_t pool_width,             \
        const dim_t pool_height, const dim_t padding_width,                \
        const dim_t padding_height, const dim_t stride_width,              \
        const dim_t stride_height, af_pooling_type pool_type);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

#undef INSTANTIATE
}
