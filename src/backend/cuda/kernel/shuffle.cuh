/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <shared.hpp>
#include <math.hpp>

namespace cuda {

template<typename T>
__global__ void shuffle(Param<T> out, CParam<T> in, const int dim,
        const int nPerThread) {

    //const int oz = blockIdx.x / blocksPerMatX;
    //const int ow = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;

    //const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
    //const int blockIdx_y =
        //(blockIdx.y + blockIdx.z * gridDim.y) - ow * blocksPerMatY;

    SharedMemory<T> shared;
    T* shuffle_cache = shared.getPointer();

    const int stride = blockDim.x * nPerThread;

    int n=0;
    int idx = (blockIdx.x * stride) + threadIdx.x;

    for(idx, n; idx < out.dims[0] && n < nPerThread; ++n, idx += stride) {
        shuffle_cache[(n*blockDim.x)  + threadIdx.x] = in.ptr[idx];
    }

    __syncthreads();

    /*
    if(threadIdx.x == 0) {
        for(int i=0; i<nPerThread * blockDim.x; ++i)
            printf("%f ", shuffle_cache[i]);
    }
    */
    
    //for(int i=nPerThread; i>0; --i) {
        
    //}

    //out.ptr[oidx] = val;
}

}  // namespace cuda
