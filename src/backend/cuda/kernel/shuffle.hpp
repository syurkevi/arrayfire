/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/VarPhilox.hpp>
#include <cub/block/block_scan.cuh>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <shuffle.hpp>
#include <af/defines.h>

#include <limits>
#include <random>

namespace cuda {
namespace kernel {

static inline __device__ void mulhilo(uint a, uint b, uint &hi, uint &lo) {
    hi = __umulhi(a, b);
    lo = a * b;
}

constexpr uint64_t PHILOX_M2_64_0 = 0xD2B74407B1CE6E93;
constexpr uint w32_0   = 0x9E3779B9;

constexpr static uint rounds = 24;
static inline uint64_t __device__ varlenPhilox(uint32_t &key, uint32_t state[2],
                                               uint64_t left_side_bits,
                                               uint64_t left_side_mask,
                                               uint64_t right_side_bits,
                                               uint64_t right_side_mask) {
    for(unsigned r=0; r<rounds; ++r) {
        uint32_t hi, lo;
        mulhilo(PHILOX_M2_64_0, state[0], hi, lo);
        lo = (lo << (right_side_bits - left_side_bits)) |
             (state[1] >> left_side_bits);

        state[0] = ((hi ^ key) ^ state[1]) & left_side_mask;
        state[1] = lo & right_side_mask;

        key += w32_0;
    }
    return (uint64_t)state[0] << right_side_bits | (uint64_t) state[1];
}

static const int THREADS = 256;

struct alignas(64) lookback {
    uint64_t localsum;
    uint64_t inclusive_prefix;
    bool local_complete;
    bool prefix_complete;
};

template<typename T>
__global__ void shufflePhilox1Pass(Param<T> out, CParam<T> in, volatile lookback *lookback_table,
                                   int *block_counter, uintl seed, uintl counter, uint nBits,
                                   uint64_t left_side_bits, uint64_t left_side_mask,
                                   uint64_t right_side_bits, uint64_t right_side_mask) {
    __shared__ uint partition_id;
    if(threadIdx.x == 0) {
        partition_id = atomicAdd(block_counter, 1);
    }
    __syncthreads();

    uint index  = partition_id * blockDim.x + threadIdx.x;

    uint32_t key;
    key = seed;
    uint32_t state[2] = { uint32_t((counter + index) >> right_side_bits),
                          uint32_t((counter + index) & right_side_mask) };

    uint64_t ridx = varlenPhilox(key, state,
                                 left_side_bits, left_side_mask,
                                 right_side_bits, right_side_mask);

    typedef cub::BlockScan<size_t, THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;

    //local scan
    size_t readFlag = (ridx < in.dims[0]);
    BlockScan(scan_storage).InclusiveSum(readFlag, readFlag);

    __shared__ uint64_t running_prefix;

    // first blocks to be scheduled will fill in lookback table and determine
    // inclusive prefixes for write indexes
    if(partition_id < gridDim.x) {

        bool amLastInBlock = threadIdx.x == blockDim.x-1;
        // final thread of each block will set local scan
        if(amLastInBlock) {
            lookback_table[partition_id].localsum = readFlag;
            __threadfence();
            lookback_table[partition_id].local_complete = true;

            // final thread of first block will also set completed
            // inclusive_prefix for further blocks
            if(partition_id == 0) {
                lookback_table[0].inclusive_prefix = readFlag;
                __threadfence();
                lookback_table[0].prefix_complete = true;
            }
        }
        __threadfence();

        if(partition_id > 0) {
            int lookback_id = partition_id - 1;
            bool check_prefix_ready = false;

            // traverse lookback table to obtain required ids for exclusive prefix
            if(amLastInBlock) {
                running_prefix = 0;
                lookback l;
                do {
                    // l = lookback_table[lookback_id];
                    memcpy((void*)&l, (const void*)&lookback_table[lookback_id], sizeof(lookback));
                    check_prefix_ready = l.prefix_complete;
                    bool check_local_ready  = l.local_complete;

                    if(check_local_ready && !check_prefix_ready && lookback_id > 0) {
                        running_prefix += l.localsum;
                        lookback_id--;
                    }
                } while(!check_prefix_ready);
                running_prefix += l.inclusive_prefix;
            }
            __syncthreads();


            readFlag += running_prefix;

            // final thread to update this block's final prefix
            if(amLastInBlock) {
                lookback_table[partition_id].inclusive_prefix = readFlag;
                __threadfence();
                lookback_table[partition_id].prefix_complete = true;
            }
        }
    }

    if(ridx < out.dims[0] && (readFlag - 1) < out.dims[0]) {
        out.ptr[(readFlag - 1)] = in.ptr[ridx];
    }
}

template<typename T>
__global__ void shufflePhilox2Pass(Param<T> out, CParam<T> in, volatile lookback *lookback_table,
                                   uintl seed, uintl counter, uint nBits,
                                   uint64_t left_side_bits, uint64_t left_side_mask,
                                   uint64_t right_side_bits, uint64_t right_side_mask,
                                   int nblk_x, int nblk_y) {

    const int bzid = blockIdx.x / nblk_x;
    const int bwid = (blockIdx.y + blockIdx.z * gridDim.y) / nblk_y;

    const int bxid = blockIdx.x - bzid * nblk_x;
    const int byid = (blockIdx.y + blockIdx.z * gridDim.y) - bwid * nblk_y;

    if (bwid >= in.dims[3] || bzid >= in.dims[2] || byid >= in.dims[1]) {
        return;
    }
    if(bwid == 0 && bzid == 0 && byid == 0) {
        return;
    }

    uint index  = bxid * blockDim.x + threadIdx.x;

    uint32_t key;
    key = seed;
    uint32_t state[2] = { uint32_t((counter + index) >> right_side_bits),
                          uint32_t((counter + index) & right_side_mask) };

    uint64_t ridx = varlenPhilox(key, state,
                                 left_side_bits, left_side_mask,
                                 right_side_bits, right_side_mask);

    typedef cub::BlockScan<size_t, THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;

    //local scan
    size_t readFlag = (ridx < in.dims[0]);
    BlockScan(scan_storage).InclusiveSum(readFlag, readFlag);


    __shared__ uint64_t running_prefix;
    if(bxid > 0) {
        bool amLastInBlock = threadIdx.x == blockDim.x-1;
        running_prefix = 0;
        lookback l;
        if(amLastInBlock) {
            memcpy((void*)&l, (const void*)&lookback_table[bxid-1], sizeof(lookback));
            running_prefix = l.inclusive_prefix;
        }
        __syncthreads();

        readFlag += running_prefix;
    }

    uint offYZW =
        bwid * out.strides[3] + bzid * out.strides[2] + byid * out.strides[1];

    if(ridx < out.dims[0] && (readFlag - 1) < out.dims[0]) {
        out.ptr[offYZW + (readFlag - 1)] = in.ptr[offYZW + ridx];
    }
}


template<typename T>
void shuffle(Param<T> out, const CParam<T> in, Param<int> block_counter, const uintl seed, uintl counter) {
    uint nBits = common::getVarPhiloxCipherBits(in.dims[0]);
    // number of elements to generate required
    uint capacity = (1 << nBits);

    //const int threads = THREADS;
    const int threads = 32;
    int nblk_x = divup(capacity, threads);
    int nblk_y = out.dims[1];

    dim3 blocks(nblk_x * out.dims[2], nblk_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    uint64_t left_side_bits = nBits / 2;
    uint64_t left_side_mask = (1ull << left_side_bits) - 1;

    uint64_t right_side_bits = nBits - left_side_bits;
    uint64_t right_side_mask = (1ull << right_side_bits) - 1;

    lookback *lookback_table;
    CUDA_CHECK(cudaMallocAsync(&lookback_table, nblk_x * sizeof(lookback),
                               cuda::getActiveStream()));
    CUDA_CHECK(cudaMemsetAsync(lookback_table, 0, nblk_x * sizeof(lookback),
                               cuda::getActiveStream()));

    CUDA_LAUNCH(shufflePhilox1Pass, nblk_x, threads, out, in, lookback_table,
                block_counter.ptr, seed, counter, nBits,
                left_side_bits, left_side_mask, right_side_bits, right_side_mask);

    CUDA_LAUNCH(shufflePhilox2Pass, blocks, threads, out, in, lookback_table,
                seed, counter, nBits,
                left_side_bits, left_side_mask, right_side_bits, right_side_mask,
                nblk_x, nblk_y);

    counter += capacity;
}

}  // namespace kernel
}  // namespace cuda
