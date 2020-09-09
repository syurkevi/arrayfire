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
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/shuffle_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

template<typename T>
void shuffle(Param<T> out, CParam<T> in, const int dim) {
    constexpr unsigned N_SHUFFLE_PER_THREAD = 32;
    constexpr unsigned SHUFFLE_TX           = 32;
    constexpr unsigned SHUFFLE_PER_BLOCK    = N_SHUFFLE_PER_THREAD * SHUFFLE_TX;

    static const std::string source(shuffle_cuh, shuffle_cuh_len);

    auto shuffle =
        common::getKernel("cuda::shuffle", {source}, {TemplateTypename<T>()});

    dim3 threads(SHUFFLE_TX, 1);

    int blocksPerMatX = divup(out.dims[0], SHUFFLE_PER_BLOCK);
    dim3 blocks(blocksPerMatX);

    //const int maxBlocksY =
        //cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    //blocks.z = divup(blocks.y, maxBlocksY);
    //blocks.y = divup(blocks.y, blocks.z);

    const size_t SHUFFLE_CACHE_SIZE = SHUFFLE_PER_BLOCK * sizeof(T);
    printf("shuffle per block %d * %d = %d, %d\n", N_SHUFFLE_PER_THREAD, SHUFFLE_TX, SHUFFLE_PER_BLOCK, SHUFFLE_CACHE_SIZE);
    EnqueueArgs qArgs(blocks, threads, getActiveStream(), SHUFFLE_CACHE_SIZE);

    shuffle(qArgs, out, in, dim, N_SHUFFLE_PER_THREAD);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
