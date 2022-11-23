/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <math.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <memory.hpp>
#include <type_traits>
#include <nvrtc_kernel_headers/scan_first_cuh.hpp>
#include <cub/block/block_scan.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include "config.hpp"

namespace cuda {
namespace kernel {

__SDH__ bool operator==(longlong2 a, longlong2 b) {
    return (a.x == b.x) && (a.y == b.y);
}
__SDH__ bool operator!=(longlong2 a, longlong2 b) { return !(a == b); }

template<typename Ti, typename To, af_op_t op>
static void scan_first_launcher(Param<To> out, Param<To> tmp, CParam<Ti> in,
                                const uint blocks_x, const uint blocks_y,
                                const uint threads_x, bool isFinalPass,
                                bool inclusive_scan) {
    auto scan_first = common::getKernel(
        "cuda::scan_first", std::array{scan_first_cuh_src},
        TemplateArgs(TemplateTypename<Ti>(), TemplateTypename<To>(),
                     TemplateArg(op), TemplateArg(isFinalPass),
                     TemplateArg(threads_x), TemplateArg(inclusive_scan)),
        std::array{DefineValue(THREADS_PER_BLOCK)});

    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    uint lim = divup(out.dims[0], (threads_x * blocks_x));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    scan_first(qArgs, out, tmp, in, blocks_x, blocks_y, lim);
    POST_LAUNCH_CHECK();
}

template<typename To, af_op_t op>
static void bcast_first_launcher(Param<To> out, CParam<To> tmp,
                                 const uint blocks_x, const uint blocks_y,
                                 const uint threads_x, bool inclusive_scan) {
    auto scan_first_bcast = common::getKernel(
        "cuda::scan_first_bcast", std::array{scan_first_cuh_src},
        TemplateArgs(TemplateTypename<To>(), TemplateArg(op)));

    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    uint lim = divup(out.dims[0], (threads_x * blocks_x));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    scan_first_bcast(qArgs, out, tmp, blocks_x, blocks_y, lim, inclusive_scan);
    POST_LAUNCH_CHECK();
}

enum blockStatus {
    INVALID = 0,
    LOCAL_AGGREGATE,
    PREFIX_COMPLETE,
};

template<typename T>
struct lookback {
    using VecT = std::conditional_t<sizeof(T)==16,
        longlong4, std::conditional_t<sizeof(T)==8,
            longlong2, std::conditional_t<sizeof(T)==4,
                int2, std::conditional_t<sizeof(T) == 2,
                    int, uchar2>>>>;
    using StatusT = std::conditional_t<sizeof(T)==16,
        longlong2, std::conditional_t<sizeof(T)==8,
            long long, std::conditional_t<sizeof(T)==4,
                int, std::conditional_t<sizeof(T) == 2,
                    short, char>>>>;

    StatusT status;
    T val;
};

template<typename To, typename Ti, af_op_t op>
__global__ void scan1Pass(Param<To> out, CParam<Ti> in, bool inclusive_scan,
                          lookback<To> *lookback_table, int *block_counter,
                          int nblk_x, int nblk_y) {
    __shared__ uint partition_id;
    if(threadIdx.x == 0) {
        partition_id = atomicAdd(block_counter, 1);
    }
    __syncthreads();

    uint idx  = partition_id * blockDim.x + threadIdx.x;

    common::Transform<Ti, To, op> transform;
    common::Binary<To, op> binop;
    const To init = common::Binary<To, op>::init();

    const Ti *iptr = in.ptr;
    To *optr       = out.ptr;

    typedef typename lookback<To>::StatusT StatusType;
    typedef typename lookback<To>::VecT VecLBType;
    typedef cub::BlockScan<To, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;

    //local scan
    bool cond = (idx < in.dims[0]);
    To scanval    = cond ? transform(iptr[idx]) : init;
    BlockScan(scan_storage).InclusiveScan(scanval, scanval, binop);

    __shared__ To running_prefix;

    // first blocks to be scheduled will fill in lookback table and determine
    // inclusive prefixes for write indexes
    if(partition_id < gridDim.x) {

        bool amLastInBlock = threadIdx.x == blockDim.x-1;
        // final thread of each block will set local scan
        if(amLastInBlock) {
            lookback<To> l;
            l.val = scanval;

            // final thread of first block will also set completed
            // inclusive_prefix for further blocks
            if(partition_id == 0) {
                l.status = StatusType{PREFIX_COMPLETE};
            } else {
                l.status = StatusType{LOCAL_AGGREGATE};
            }

            VecLBType *lv = reinterpret_cast<VecLBType*>(&l);
            // store lookback value and status flag at same time with vector instructions
            cub::ThreadStore<cub::STORE_CG>(reinterpret_cast<VecLBType*>(&lookback_table[partition_id]), *lv);
        }

        if(partition_id > 0) {
            int lookback_id = partition_id - 1;
            bool check_prefix_ready = false;

            // traverse lookback table to obtain required ids for exclusive prefix
            if(amLastInBlock) {
                running_prefix = init;
                lookback<To> l;
                do {
                    // load lookback value and status flag at same time with vector instructions
                    VecLBType lv = cub::ThreadLoad<cub::LOAD_CG>(reinterpret_cast<VecLBType*>(&lookback_table[lookback_id]));
                    l = *(reinterpret_cast<lookback<To>*>(&lv));

                    StatusType lstatus = l.status;
                    check_prefix_ready = (lstatus == StatusType{PREFIX_COMPLETE});
                    bool check_local_ready  = (lstatus == StatusType{LOCAL_AGGREGATE});

                    if(check_local_ready && !check_prefix_ready && lookback_id > 0) {
                        running_prefix = binop(running_prefix, l.val);
                        lookback_id--;
                    }
                } while(!check_prefix_ready);
                running_prefix = binop(running_prefix, l.val);
            }
            __syncthreads();


            scanval = binop(scanval, running_prefix);

            // final thread to update this block's final prefix
            if(amLastInBlock) {
                lookback<To> l {StatusType{PREFIX_COMPLETE}, scanval};
                VecLBType *lv = reinterpret_cast<VecLBType*>(&l);
                // store lookback value and status flag at same time with vector instructions
                cub::ThreadStore<cub::STORE_CG>(reinterpret_cast<VecLBType*>(&lookback_table[partition_id]), *lv);
            }
        }
    }

    if (inclusive_scan != 0) {
        if(idx < out.dims[0]) {
            optr[idx] = scanval;
        }
    } else {
        if (idx == (out.dims[0] - 1)) {
            optr[0] = init;
        } else if(idx < (out.dims[0] - 1)) {
            optr[idx + 1] = scanval;
        }
    }
}


template<typename Ti, typename To, af_op_t op>
static void scan_first(Param<To> out, CParam<Ti> in, bool inclusive_scan) {

#if 0
    uint threads_x = nextpow2(std::max(32u, (uint)out.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_BLOCK);
    uint threads_y = THREADS_PER_BLOCK / threads_x;

    uint blocks_x = divup(out.dims[0], threads_x * REPEAT);
    uint blocks_y = divup(out.dims[1], threads_y);

    if (blocks_x == 1) {
        scan_first_launcher<Ti, To, op>(out, out, in, blocks_x, blocks_y,
                                        threads_x, true, inclusive_scan);

    } else {
        Param<To> tmp = out;

        tmp.dims[0]    = blocks_x;
        tmp.strides[0] = 1;
        for (int k = 1; k < 4; k++)
            tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];

        int tmp_elements = tmp.strides[3] * tmp.dims[3];
        auto tmp_alloc   = memAlloc<To>(tmp_elements);
        tmp.ptr          = tmp_alloc.get();

        scan_first_launcher<Ti, To, op>(out, tmp, in, blocks_x, blocks_y,
                                        threads_x, false, inclusive_scan);

        // FIXME: Is there an alternative to the if condition ?
        if (op == af_notzero_t) {
            scan_first_launcher<To, To, af_add_t>(tmp, tmp, tmp, 1, blocks_y,
                                                  threads_x, true, true);
        } else {
            scan_first_launcher<To, To, op>(tmp, tmp, tmp, 1, blocks_y,
                                            threads_x, true, true);
        }

        bcast_first_launcher<To, op>(out, tmp, blocks_x, blocks_y, threads_x,
                                     inclusive_scan);
    }

#else

    uint threads_x = nextpow2(std::max(32u, (uint)out.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_BLOCK);

    int nblk_x = divup(out.dims[0], threads_x);
    int nblk_y = out.dims[1];

    dim3 blocks(nblk_x * out.dims[2], nblk_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    lookback<To> *lookback_table;
    CUDA_CHECK(cudaMallocAsync(&lookback_table, nblk_x * sizeof(lookback<To>),
                               cuda::getActiveStream()));
    CUDA_CHECK(cudaMemsetAsync(lookback_table, 0, nblk_x * sizeof(lookback<To>),
                               cuda::getActiveStream()));

    int *block_counter;
    CUDA_CHECK(cudaMallocAsync(&block_counter, sizeof(int),
                               cuda::getActiveStream()));
    CUDA_CHECK(cudaMemsetAsync(block_counter, 0, sizeof(int),
                               cuda::getActiveStream()));

    /*
    auto scan1Pass =
        common::getKernel("cuda::scan1Pass", std::array{scan_first_cuh_src},
                          TemplateArgs(TemplateTypename<To>(), TemplateTypename<Ti>(),
                          TemplateArg(op)));
    */


    //EnqueueArgsjqArgs(blocks, threads_x, getActiveStream());
    CUDA_LAUNCH((scan1Pass<To, Ti, op>), blocks, threads_x, out, in, inclusive_scan,
                lookback_table, block_counter, nblk_x, nblk_y);
    POST_LAUNCH_CHECK();
#endif
}

}  // namespace kernel
}  // namespace cuda
