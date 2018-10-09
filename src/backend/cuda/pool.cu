/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <handle.hpp>
#include <pool.hpp>
#include <err_cuda.hpp>
#include <cudnn.h>
#include <wrap.hpp>
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

/*
   // possible supported types

   CUDNN_DATA_FLOAT
   CUDNN_DATA_DOUBLE
   CUDNN_DATA_HALF
   CUDNN_DATA_INT8
   CUDNN_DATA_UINT8
   CUDNN_DATA_INT32
   CUDNN_DATA_INT8x4
   CUDNN_DATA_INT8x32
   CUDNN_DATA_UINT8x4

*/

namespace cuda
{
Array<float> pool2(const Array<float>& in,
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

    CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                     CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT,
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
                                     CUDNN_DATA_FLOAT,
                                     n_out,
                                     c_out,
                                     h_out,
                                     w_out));

    dim4 odims(w_out, h_out, c_out, n_out);
    Array<float> out = createValueArray<float>(odims, 0);

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

    CUDNN(cudnnDestroyPoolingDescriptor(pooling_descriptor));
    CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN(cudnnDestroyTensorDescriptor(output_descriptor));

    return out;
}

}
