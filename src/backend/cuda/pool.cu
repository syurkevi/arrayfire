/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <pool.hpp>

#include <Array.hpp>
#include <cudnn.h>
#include <err_cuda.hpp>
#include <handle.hpp>
#include <reduce.hpp>
#include <unwrap.hpp>
//#include <kernel/canny.hpp>

#define CUDNN(call) do {                                \
        cudnnStatus_t s = (call);                       \
        if (s == CUDNN_STATUS_SUCCESS) break;           \
        fprintf(stderr, __FILE__": %d: %s (%d)\n",      \
               __LINE__, cudnnGetErrorString(s), s);    \
        exit(1);                                        \
    } while(0);


using af::dim4;


namespace cuda
{

template<typename T>
cudnnDataType_t getCudnnDataType() {
    switch((af_dtype) dtype_traits<T>::af_type) {
        case f32:
            return CUDNN_DATA_FLOAT; break;
        case f64:
            return CUDNN_DATA_DOUBLE; break;
        case s32:
            return CUDNN_DATA_INT32; break;
        case u8:
            return CUDNN_DATA_UINT8; break;
        default:
            return CUDNN_DATA_FLOAT;
    }
}

template<typename T>
bool supported_cudnn_type() {
    af_dtype dtype = ((af_dtype) dtype_traits<T>::af_type);
    if(dtype == f32 || dtype == f64 || dtype == s32 || dtype == u8) {
        return true;
    }
    return false;
}

template<typename T>
Array<T> pool2_cudnn(const Array<T>& in,
                     const dim_t pool_width, const dim_t pool_height,
                     const dim_t padding_width, const dim_t padding_height,
                     const dim_t stride_width, const dim_t stride_height,
                     af_pooling_type pool_type)
{
    dim4 idims = in.dims();

    cudnnHandle_t cudnn;
    CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t  input_descriptor;
    cudnnTensorDescriptor_t  output_descriptor;
    cudnnPoolingDescriptor_t pooling_descriptor;

    CUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));

    const int n = idims[3];
    const int c = idims[2];
    const int h = idims[1];
    const int w = idims[0];

    cudnnDataType_t dtype = getCudnnDataType<T>();
    CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                     CUDNN_TENSOR_NCHW,
                                     dtype,
                                     n, c, h, w));


    const int pool_dims = 2;
    int window_dims[pool_dims] = {(int)pool_width, (int)pool_height};
    int padding_dims[pool_dims] = {(int)padding_width, (int)padding_height};
    int stride_dims[pool_dims] = {(int)stride_width, (int)stride_height};

    CUDNN(cudnnSetPoolingNdDescriptor(pooling_descriptor,
                                      CUDNN_POOLING_MAX,
                                      CUDNN_PROPAGATE_NAN,
                                      pool_dims,
                                      window_dims,
                                      padding_dims,
                                      stride_dims));


    const int tensorDims = 4;
    int pooled_output_dim[tensorDims];
    CUDNN(cudnnGetPoolingNdForwardOutputDim(pooling_descriptor,
                                            input_descriptor,
                                            tensorDims,
                                            pooled_output_dim));

    const int n_out = pooled_output_dim[0];
    const int c_out = pooled_output_dim[1];
    const int h_out = pooled_output_dim[2];
    const int w_out = pooled_output_dim[3];

    CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                     CUDNN_TENSOR_NCHW,
                                     dtype,
                                     n_out,
                                     c_out,
                                     h_out,
                                     w_out));

    dim4 odims(w_out, h_out, c_out, n_out);
    Array<T> out = createValueArray<T>(odims, 0);

    float alpha = 1.f;
    float beta  = 0.f;
    CUDNN(cudnnPoolingForward(cudnn,
                              pooling_descriptor,
                              &alpha,
                              input_descriptor,
                              in.device(),
                              &beta,
                              output_descriptor,
                              out.device()));


    /*
       TEST BACKWARDS PASS
     */

    dim4 grad_dims = odims;
    Array<T> ingrad = createValueArray<T>(grad_dims, 1);

    cudnnTensorDescriptor_t  ingrad_descriptor;
    CUDNN(cudnnCreateTensorDescriptor(&ingrad_descriptor));
    CUDNN(cudnnSetTensor4dDescriptor(ingrad_descriptor,
                                     CUDNN_TENSOR_NCHW,
                                     dtype,
                                     grad_dims[3], grad_dims[2], grad_dims[1], grad_dims[0]));

    Array<T> grad = createValueArray<T>(idims, 0);
    cudnnTensorDescriptor_t  grad_descriptor;
    CUDNN(cudnnCreateTensorDescriptor(&grad_descriptor));
    CUDNN(cudnnSetTensor4dDescriptor(grad_descriptor,
                                     CUDNN_TENSOR_NCHW,
                                     dtype,
                                     n, c, h, w));


    cudnnPoolingBackward(cudnn,
                         pooling_descriptor,
                         &alpha,
                         output_descriptor,
                         out.device(),
                         ingrad_descriptor,
                         ingrad.device(),
                         input_descriptor,
                         in.device(),
                         &beta,
                         grad_descriptor,
                         grad.device() );
    af_print_array(getHandle(grad));
    /*
        /TEST
    */


    CUDNN(cudnnDestroyPoolingDescriptor(pooling_descriptor));
    CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN(cudnnDestroyTensorDescriptor(output_descriptor));

    return out;
}

template<typename T>
Array<T> pool2_native(const Array<T>& in,
                      const dim_t pool_width, const dim_t pool_height,
                      const dim_t padding_width, const dim_t padding_height,
                      const dim_t stride_width, const dim_t stride_height,
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

template<typename T>
Array<T> pool2(const Array<T>& in,
               const dim_t pool_width, const dim_t pool_height,
               const dim_t padding_width, const dim_t padding_height,
               const dim_t stride_width, const dim_t stride_height,
               af_pooling_type pool_type)
{
    if(supported_cudnn_type<T>())
        return pool2_cudnn(in, pool_width, pool_height, padding_width, padding_height, stride_width, stride_height, pool_type);
    else
        return pool2_native(in, pool_width, pool_height, padding_width, padding_height, stride_width, stride_height, pool_type);
}

#define INSTANTIATE(T)                                                  \
    template Array<T> pool2<T>(const Array<T> &in,                      \
               const dim_t pool_width, const dim_t pool_height,         \
               const dim_t padding_width, const dim_t padding_height,   \
               const dim_t stride_width, const dim_t stride_height,     \
               af_pooling_type pool_type);


INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}
