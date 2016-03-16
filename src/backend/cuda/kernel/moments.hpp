/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
namespace kernel
{

    // Kernel Launch Config Values
    static const int THREADS = 256;

    // Moment functions
    template<typename T>
    __device__ inline static
    T moments_m00(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                       CParam<T> in, const bool pBatch)
    {
        dim_t mId = idy * in.strides[1] + idx;
        if(pBatch) {
            mId += idw * in.strides[3] + idz * in.strides[2];
        }

        return in.ptr[mId];
    }
template<typename T>
    __device__ inline static
    T moments_m01(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                       CParam<T> in, const bool pBatch)
    {
        dim_t mId = idy * in.strides[1] + idx;
        if(pBatch) {
            mId += idw * in.strides[3] + idz * in.strides[2];
        }

        return idx * in.ptr[mId];
    }

    template<typename T>
    __device__ inline static
    T moments_m10(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                       CParam<T> in, const bool pBatch)
    {
        dim_t mId = idy * in.strides[1] + idx;
        if(pBatch) {
            mId += idw * in.strides[3] + idz * in.strides[2];
        }

        return idy * in.ptr[mId];
    }

    template<typename T>
    __device__ inline static
    T moments_m11(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                       CParam<T> in, const bool pBatch)
    {
        dim_t mId = idy * in.strides[1] + idx;
        if(pBatch) {
            mId += idw * in.strides[3] + idz * in.strides[2];
        }

        return idx * idy * in.ptr[mId];
    }

    template<typename T, af_moment moment>
    __global__
    void moments_kernel(double* res, CParam<T> in,
                  const dim_t blocksMatX, const bool pBatch)
    {
        const dim_t idw = blockIdx.y / in.dims[2];
        const dim_t idz = blockIdx.y - idw * in.dims[2];

        const dim_t idy = blockIdx.x / blocksMatX;
        const dim_t blockIdx_x = blockIdx.x - idy * blocksMatX;
        const dim_t idx = blockIdx_x * blockDim.x + threadIdx.x;

        if (idx >= in.dims[0] || idy >= in.dims[1] ||
            idz >= in.dims[2] || idw >= in.dims[3] )
            return;

        switch(moment) {
            case M00:
                *res = (double)moments_m00(idx, idy, idz, idw, in, pBatch);
                break;
            case M01:
                *res = (double)moments_m01(idx, idy, idz, idw, in, pBatch);
                break;
            case M10:
                *res = (double)moments_m10(idx, idy, idz, idw, in, pBatch);
                break;
            case M11:
                *res = (double)moments_m11(idx, idy, idz, idw, in, pBatch);
                break;
            default:
                break;
        }
    }

    // Wrapper functions
    template <typename T, af_moment moment>
    void moments(T *val, CParam<T> in) {
        dim3 threads(THREADS, 1, 1);
        dim_t blocksPerMat = divup(in.dims[0], threads.x);
        dim3 blocks(blocksPerMat * in.dims[1], in.dims[2] * in.dims[3]);

        bool pBatch = !(in.dims[2] == 1 && in.dims[3] == 1);

        double *res;
        CUDA_CHECK(cudaMalloc((void**)&res, sizeof(double)));
        CUDA_LAUNCH((moments_kernel<T, moment>), blocks, threads,
                     (double*) res, in, blocksPerMat, pBatch);
        POST_LAUNCH_CHECK();
        CUDA_CHECK(cudaMemcpy(val, res, sizeof(double), cudaMemcpyDeviceToHost));
    }

}
}
