/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <ops.hpp>
#include <backend.hpp>
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "config.hpp"
#include <memory.hpp>
#include <type_traits>

#include <cub/device/device_reduce.cuh>

using std::unique_ptr;

const static int FULL_MASK = 0xFFFFFFFF;

namespace cuda
{
namespace kernel
{

    template<typename Ti, typename Tk, typename To, af_op_t op, uint dim, uint DIMY>
    __global__
    static void reduce_dim_kernel_by_key(Param<To> out,
                                  CParam <Tk> key,
                                  CParam <Ti> in,
                                  uint blocks_x, uint blocks_y, uint offset_dim,
                                  bool change_nan, To nanval)
    {
        /*
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;
        const uint tid  = tidy * THREADS_X + tidx;

        const uint zid = blockIdx.x / blocks_x;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint xid = blockIdx_x * blockDim.x + tidx;

        __shared__ To s_val[THREADS_X * DIMY];

        const uint wid = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
        const uint blockIdx_y = (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y) * wid;
        const uint yid = blockIdx_y; // yid  of output. updated for input later.

        uint ids[4] = {xid, yid, zid, wid};

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        To * const optr = out.ptr + ids[3] * out.strides[3] +
                                    ids[2] * out.strides[2] +
                                    ids[1] * out.strides[1] + ids[0];

        const uint blockIdx_dim = ids[dim];
        ids[dim] = ids[dim] * blockDim.y + tidy;

        const Ti * iptr = in.ptr + ids[3] * in.strides[3] +
                                   ids[2] * in.strides[2] +
                                   ids[1] * in.strides[1] + ids[0];

        const uint id_dim_in = ids[dim];
        const uint istride_dim = in.strides[dim];

        bool is_valid =
            (ids[0] < in.dims[0]) &&
            (ids[1] < in.dims[1]) &&
            (ids[2] < in.dims[2]) &&
            (ids[3] < in.dims[3]);

        Transform<Ti, To, op> transform;
        Binary<To, op> reduce;
        To out_val = reduce.init();
        for (int id = id_dim_in; is_valid && (id < in.dims[dim]); id += offset_dim * blockDim.y) {
            To in_val = transform(*iptr);
            if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval;
            out_val = reduce(in_val, out_val);
            iptr = iptr + offset_dim * blockDim.y * istride_dim;
        }

        s_val[tid] = out_val;

        To *s_ptr = s_val + tid;
        __syncthreads();

        if (DIMY == 8) {
            if (tidy < 4) *s_ptr = reduce(*s_ptr, s_ptr[THREADS_X * 4]);
            __syncthreads();
        }

        if (DIMY >= 4) {
            if (tidy < 2) *s_ptr = reduce(*s_ptr, s_ptr[THREADS_X * 2]);
            __syncthreads();
        }

        if (DIMY >= 2) {
            if (tidy < 1) *s_ptr = reduce(*s_ptr, s_ptr[THREADS_X * 1]);
            __syncthreads();
        }

        if (tidy == 0 && is_valid &&
            (blockIdx_dim < out.dims[dim])) {
            *optr = *s_ptr;
        }
        */
    }

    template<typename Ti, typename Tk, typename To, af_op_t op, int dim>
    void reduce_dim_launcher_by_key(Param<To> out, CParam<Ti> in, CParam<Tk> key,
                             const uint threads_y, const dim_t blocks_dim[4],
                             bool change_nan, double nanval)
    {
        /*
         * TODO: throw not yet implemented error
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_dim[0] * blocks_dim[2],
                    blocks_dim[1] * blocks_dim[3]);

        const int maxBlocksY = cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
        blocks.z = divup(blocks.y, maxBlocksY);
        blocks.y = divup(blocks.y, blocks.z);

        switch (threads_y) {
        case 8:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 8>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        case 4:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 4>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        case 2:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 2>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        case 1:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 1>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        }

        POST_LAUNCH_CHECK();
        */
    }

    //swaps key and value according to key ordering and target direction
    template<typename Ti, typename Tk>
    __device__ int kv_swap(Tk &k, Ti &v, int mask, int dir)
    {
        Tk ky = __shfl_xor_sync(FULL_MASK, k, mask);
        Ti vy = __shfl_xor_sync(FULL_MASK, v, mask);
        bool swap = (k < ky == dir) && (k != ky);
        k = swap ? ky : k;
        v = swap ? vy : v;
    }

    //specialization for cfloat
    template<typename Tk>
    __device__ int kv_swap(Tk &k, cuda::cfloat &v, int mask, int dir)
    {
        Tk   ky;
        cuda::cfloat vy;
        //int   ky = __shfl_xor_sync(FULL_MASK, k, mask);
        //float vy = __shfl_xor_sync(FULL_MASK, v, mask);
        bool swap = (k < ky == dir) && (k != ky);
        k = swap ? ky : k;
        v = swap ? vy : v;
    }

    //specialization for cdouble
    template<typename Tk>
    __device__ int kv_swap(Tk &k, cuda::cdouble &v, int mask, int dir)
    {
        Tk   ky;
        cuda::cdouble vy;
        //int   ky = __shfl_xor_sync(FULL_MASK, k, mask);
        //float vy = __shfl_xor_sync(FULL_MASK, v, mask);
        bool swap = (k < ky == dir) && (k != ky);
        k = swap ? ky : k;
        v = swap ? vy : v;
    }


    //__shfl_down_sync wrapper
    template<typename T>
    __device__ T shfl_down_sync(unsigned mask, T var, int delta) {
        return __shfl_down_sync(mask, var, delta);
    }
    //specialization for cfloat
    template<> inline
    __device__ cuda::cfloat shfl_down_sync(unsigned mask, cuda::cfloat var, int delta) {
        cuda::cfloat res = { __shfl_down_sync(mask, var.x, delta), 
                             __shfl_down_sync(mask, var.y, delta)};
        return res;
    }
    //specialization for cdouble
    template<> inline
    __device__ cuda::cdouble shfl_down_sync(unsigned mask, cuda::cdouble var, int delta) {
        cuda::cdouble res = { __shfl_down_sync(mask, var.x, delta), 
                              __shfl_down_sync(mask, var.y, delta)};
        return res;
    }

    __device__ __forceinline__ unsigned bfe(int x, int bf) {
        return (x >> bf) & 1;
    }

    //single-threaded binary search for key in list of values
    //used to find a thread's key position inside warp sized list in shared memory
    //TODO: template parameter bool? find first/last
    template<typename Tk, bool find_first_index>
    __device__  int findpos(const Tk &key, const Tk const *warpKeys, int n) {
        if(find_first_index) {
            int min=0, max=n-1, mid;
            while(min <= max) {
                mid = min + ((max - min) / 2);
                if(warpKeys[mid] < key) {
                    min = mid + 1;
                } else {
                    max = mid - 1;
                }
            }
            return min;
        } else {
            int min=0, max=n-1, mid;
            while(min <= max) {
                mid = min + ((max - min) / 2);
                if(warpKeys[mid] <= key) {
                    min = mid + 1;
                } else {
                    max = mid - 1;
                }
            }
            return max+1;
        }
    }

    //intermediate kernel used to test if further reductions are necessary
    //assumes sorted keys as input
    //assumes zeroed output flag before call
    template<typename Tk>
    __global__ void testReduction(int *needs_updates, const Tk *sorted_keys, const int n) {
        //TODO: set to 0 in kernel instead? removing memset dependency
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        Tk k;
        if(tid < n)
            k = sorted_keys[tid];

        int update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 1)) && (tid < n-1) && ((tid%32) < 31);

        int remaining_updates = __any_sync(0xFFFFFFFF, update_key);
        //needs updates will be set if any duplicate keys remain
        //TODO: atomics in shared before global?
        if(tid < n)
            atomicOr(needs_updates, remaining_updates);
    }

    const static int maxResPerWarp = 32; //assume dim 0, no NAN values

    //TODO change kernel params to match arrayfire api
    //TODO handle nans
    //TODO handle types
    //TODO handle reduction operator
    template<typename Ti, typename Tk, typename To, af_op_t op, uint DIMX>
    __global__
    static void reduce_first_by_key_kernel(int *n_reduced,
                                           Param<Tk>  reduced_keys,
                                           Param<To>  reduced_vals,
                                           CParam<Tk> keys,
                                           CParam<Ti> vals,
                                           int n, bool change_nan, To nanval) {

        /*
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;
        const uint tid  = tidy * blockDim.x + tidx;

        const uint zid = blockIdx.x / blocks_x;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint xid = blockIdx_x * blockDim.x * repeat + tidx;

        Binary<To, op> reduce;
        Transform<Ti, To, op> transform;

        __shared__ To s_val[THREADS_PER_BLOCK];

        const uint wid = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
        const uint blockIdx_y = (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y) * wid;
        const uint yid = blockIdx_y * blockDim.y + tidy;

        const Ti * const iptr = in.ptr + (wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1]);

        if (yid >= in.dims[1] ||
            zid >= in.dims[2] ||
            wid >= in.dims[3]) return;


        int lim = min((int)(xid + repeat * DIMX), in.dims[0]);

        To out_val = reduce.init();
        for (int id = xid; id < lim; id += DIMX) {
            To in_val = transform(iptr[id]);
            if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval;
            out_val = reduce(in_val, out_val);
        }

        s_val[tid] = out_val;

        */
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int laneid = tid % 32;

        //allocate and initialize shared memory
        const int nWarps = DIMX / 32;

        //printf("%d ", maxResPerWarp * nWarps);
        __shared__ int warpReduceSizes[nWarps];
        __shared__ int warpReduceFinalLocation[nWarps][maxResPerWarp];

        __shared__ Tk  warpReduceKeys[nWarps][maxResPerWarp];
        __shared__ Ti  warpReduceVals[nWarps][maxResPerWarp];

        __shared__ Tk  warpReduceKeysSmemFinal[nWarps * maxResPerWarp];
        __shared__ Ti  warpReduceValsSmemFinal[nWarps * maxResPerWarp];

        __shared__ int nReducedInSmem;
        __shared__ int reducedBlockSize;
        __shared__ int blockReductionsTotal;

        if(threadIdx.x == 0) {
            nReducedInSmem = 0;
            blockReductionsTotal = 0;
            reducedBlockSize = 0;
        }

        //load keys and values to threads
        Tk k;
        Ti v;
        if(tid < n) {
           k = keys.ptr[tid];
           v = vals.ptr[tid];
        } else {
           k = scalar<Tk>(INFINITY);
           v = scalar<Ti>(INFINITY);
        }
/*

        //__syncwarp();

        //test that warp needs to be sorted
        //should speedup cases where input doesn't contain many different keys or keys are already sorted
        char in_order = (k <= __shfl_down_sync(0xFFFFFFFF, k , 1));
        if(!__all_sync(0xFFFFFFFF, in_order)) {
            //sort each warp according to key
            //TODO: handle cases with non-complete warp
            //use shfl_sync mask?
            //2
            kv_swap(k, v, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0));
            //4
            kv_swap(k, v, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1));
            kv_swap(k, v, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));
            //8
            kv_swap(k, v, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2));
            kv_swap(k, v, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
            kv_swap(k, v, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
            //16
            kv_swap(k, v, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));
            kv_swap(k, v, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            kv_swap(k, v, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            kv_swap(k, v, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
            //32
            kv_swap(k, v, 0x10, bfe(laneid, 4));
            kv_swap(k, v, 0x08, bfe(laneid, 3));
            kv_swap(k, v, 0x04, bfe(laneid, 2));
            kv_swap(k, v, 0x02, bfe(laneid, 1));
            kv_swap(k, v, 0x01, bfe(laneid, 0));
        }

        //float s = v;
        //special case of single key per warp
        char all_eq = (k == __shfl_down_sync(0xFFFFFFFF, k , 1));
        if(__all_sync(0xFFFFFFFF, all_eq)) {
            v = v + shfl_down_sync(FULL_MASK, v , 1);
            v = v + shfl_down_sync(FULL_MASK, v , 2);
            v = v + shfl_down_sync(FULL_MASK, v , 4);
            v = v + shfl_down_sync(FULL_MASK, v , 8);
            v = v + shfl_down_sync(FULL_MASK, v , 16);
        } else {
            //preform reduction for each of the unique keys
            int update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 1)) && (laneid < 31);
            unsigned shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 1);
            Ti uval = shfl_down_sync(shflmask, v , 1); //primitive version for now
            v = update_key ?  (v + uval) : v;
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 2)) && (laneid < 30);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 2);
            uval = shfl_down_sync(shflmask, v , 2);
            v = update_key ?  (v + uval) : v;
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 4)) && (laneid < 28);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 4);
            uval = shfl_down_sync(shflmask, v , 4);
            v = update_key ?  (v + uval) : v;
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 8)) && (laneid < 24);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 8);
            uval = shfl_down_sync(shflmask, v , 8);
            v = update_key ?  (v + uval) : v;
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 16)) && (laneid < 16);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 16);
            uval = shfl_down_sync(shflmask, v , 16);
            v = update_key ?  (v + uval) : v;
            //__syncwarp();
        }

        //__syncthreads();//TMP
        */
        /*

        char unique_flag = ((k != __shfl_up_sync(0xFFFFFFFF, k , 1)) || (laneid == 0)) && (tid < n);
        char unique_id = unique_flag;
        const int warpid = threadIdx.x / 32;

        #pragma unroll
        for(int offset=1; offset<32; offset <<= 1) {
            char y = __shfl_up_sync(0xFFFFFFFF, unique_id, offset);
            if(laneid >= offset)
                unique_id += y;
        }

        //if warp is empty, will unique key be correct?
        if(laneid == 31) {
            //const int n_unique_keys = __popc(__ballot_sync(0xFFFFFFFF, unique_flag));
            warpReduceSizes[warpid] = unique_id;
        }


        if(unique_flag) {
            warpReduceKeys[warpid][unique_id-1] = k;
            warpReduceVals[warpid][unique_id-1] = v;
        }

        __syncthreads();

        //at this point, we have nWarps sorted lists in shared memory with each list's size located in the warpReduceSizes[] array
        //each thread should perform a binary search over each of the nWarps sorted lists to find its position in the final merged list

        int finalSMemPos = laneid;
        #pragma unroll
        for(int w=0; w<nWarps; ++w) {
            if(w != warpid) {
                if(warpid < w)
                    finalSMemPos += findpos<Tk, true>(warpReduceKeys[warpid][laneid], warpReduceKeys[w], warpReduceSizes[w]);
                else
                    finalSMemPos += findpos<Tk, false>(warpReduceKeys[warpid][laneid], warpReduceKeys[w], warpReduceSizes[w]);
            }
        }

        //from this point onwards, single warp should handle more than a single list, however
        //at the moment do primitive operation
        //multi-list per warp will require load balancing structure

        //only write unique keys from each warp
        if(laneid < warpReduceSizes[warpid] && tid < n) {
            warpReduceKeysSmemFinal[finalSMemPos] = warpReduceKeys[warpid][laneid];
            warpReduceValsSmemFinal[finalSMemPos] = warpReduceVals[warpid][laneid];
        }

        if(threadIdx.x == 0) {
            #pragma unroll
            for(int w=0; w<nWarps; ++w) {
                nReducedInSmem += warpReduceSizes[w];
            }
        }

        __syncthreads();
        //re-load keys and values to threads from first pass reduced shared memory
        if (threadIdx.x < nReducedInSmem) {
           k = warpReduceKeysSmemFinal[threadIdx.x];
           v = warpReduceValsSmemFinal[threadIdx.x];
        } else {
           k = scalar<Tk>(INFINITY);
           v = scalar<Ti>(INFINITY);
        }

        if(warpid * 32 > nReducedInSmem)
            return; //retire warp early
        //__syncwarp();

        //run second pass of reduction, should completely reduce shared memory (1024/32/32)
        //special case of single key per warp
        all_eq = (k == __shfl_down_sync(0xFFFFFFFF, k , 1));
        if(__all_sync(0xFFFFFFFF, all_eq)) {
            v += __shfl_down_sync(FULL_MASK, v , 1);
            v += __shfl_down_sync(FULL_MASK, v , 2);
            v += __shfl_down_sync(FULL_MASK, v , 4);
            v += __shfl_down_sync(FULL_MASK, v , 8);
            v += __shfl_down_sync(FULL_MASK, v , 16);
        } else {
            //perform reduction for each of the unique keys
            int update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 1)) && (laneid < 31);
            unsigned shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 1);
            float uval = __shfl_down_sync(shflmask, v , 1); //primitive version for now
            v += update_key ?  uval : scalar<Ti>(0);
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 2)) && (laneid < 30);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 2);
            uval = __shfl_down_sync(shflmask, v , 2);
            v += update_key ?  uval : scalar<Ti>(0);
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 4)) && (laneid < 28);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 4);
            uval = __shfl_down_sync(shflmask, v , 4);
            v += update_key ?  uval : scalar<Ti>(0);
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 8)) && (laneid < 24);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 8);
            uval = __shfl_down_sync(shflmask, v , 8);
            v += update_key ?  uval : scalar<Ti>(0);
            //__syncwarp();

            update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 16)) && (laneid < 16);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 16);
            uval = __shfl_down_sync(shflmask, v , 16);
            v += update_key ?  uval : scalar<Ti>(0);
            //__syncwarp();
        }

        unique_flag = ((k != __shfl_up_sync(0xFFFFFFFF, k , 1)) || (laneid == 0)) && (threadIdx.x < nReducedInSmem);
        //unique_flag = ((k != __shfl_up_sync(0xFFFFFFFF, k , 1)) || (laneid == 0));
        unique_id = unique_flag;

        #pragma unroll
        for(int offset=1; offset<32; offset <<= 1) {
            char y = __shfl_up_sync(0xFFFFFFFF, unique_id, offset);
            if(laneid >= offset)
                unique_id += y;
        }

        if(laneid == 31) {
            warpReduceSizes[warpid] = unique_id;
        }

        if(unique_flag) {
            warpReduceKeys[warpid][unique_id-1] = k;
            warpReduceVals[warpid][unique_id-1] = v;
        }
        __syncthreads();

        int warpSzScan = 0;
        if(warpid == 0 && laneid < nWarps) {
            warpSzScan = warpReduceSizes[laneid];
            int activemask = 0xFFFFFFFF >> (32 - nWarps);
            #pragma unroll
            for(int offset=1; offset<32; offset <<= 1) {
                char y = __shfl_up_sync(activemask, warpSzScan, offset);
                if(laneid >= offset)
                    warpSzScan += y;
            }
            //warpReduceSizes[laneid] = warpSzScan;
            if(laneid == 0) //todo: correct condition?
                reducedBlockSize = warpSzScan;
        }
        __syncthreads();

        //int reducedBlockSize = warpReduceSizes[nWarps - 1];
        //printf("bkredsz %d \n", reducedBlockSize);

        if(threadIdx.x == 0) {
            blockReductionsTotal = atomicAdd(n_reduced, reducedBlockSize);
        }

        int warpOffset = warpid > 0 ? warpReduceSizes[warpid - 1] : 0;

        if(laneid < warpReduceSizes[warpid] && threadIdx.x < nReducedInSmem && tid < n) {
            int warpOffset = warpid > 0 ? warpReduceSizes[warpid - 1] : 0;
            reduced_keys.ptr[blockReductionsTotal + warpOffset + laneid] = scalar<Tk>(warpReduceKeys[warpid][laneid]);
            reduced_vals.ptr[blockReductionsTotal + warpOffset + laneid] = scalar<To>(warpReduceVals[warpid][laneid]);
        }
        */
    }

    template<typename Ti, typename Tk, typename To, af_op_t op, int DIMX>
    int reduce_first_by_key_launcher(Param<Tk> keys_out, Param<To> vals_out, CParam<Tk> keys, CParam<Ti> vals, bool change_nan, double nanval)
    {
        auto n_reduced       = memAlloc<int>(1);
        CUDA_CHECK(cudaMemset(n_reduced.get(), 0, sizeof(int)));

        int nelems = keys.dims[0];
        const int numBlocks = divup(nelems, DIMX);
        printf("using %d blocks, each with %d threads to perform initial block-level reduction\n", numBlocks, DIMX);

        CUDA_LAUNCH((reduce_first_by_key_kernel<Ti, Tk, To, op, DIMX>), numBlocks, DIMX, n_reduced.get(), keys_out, vals_out, keys, vals, nelems, change_nan, scalar<To>(nanval));
        POST_LAUNCH_CHECK();

        int n_reduced_after_initial;
        CUDA_CHECK(cudaMemcpy(&n_reduced_after_initial, n_reduced.get(), sizeof(int), cudaMemcpyDeviceToHost));
        return n_reduced_after_initial;
    }

    template<typename Ti, typename Tk, typename To, af_op_t op>
    int test_reduction_launcher(Param<To> out, CParam<Ti> in, CParam<Tk> key,
                                const uint blocks_x, const uint blocks_y, const uint threads_x,
                                bool change_nan, double nanval)
    {
        /*
        //test if further reduction necessary
        CUDA(cudaMemset(needs_reduction, 0, sizeof(int)));
        numBlocks = divup(n_reduced_after_initial, numThreads);
        testReduction<<<numBlocks, numThreads>>>(needs_reduction, db_keys.Current(), n_reduced_after_initial);
        int another_reduction;
        CUDA(cudaMemcpy(&another_reduction, needs_reduction, sizeof(int), cudaMemcpyDeviceToHost));
        */
    }


    template<typename Ti, typename Tk, typename To, af_op_t op>
    void reduce_first_by_key(Param<Tk> keys_out, Param<To> vals_out, CParam<Tk> keys, CParam<Ti> vals, bool change_nan, double nanval)
    {
        printf("calling reduce_first_by_key\n");
        //printf("nelems to reduce: %d\n", nelems);
    }

    template<typename Ti, typename Tk, typename To, af_op_t op, int dim>
    void reduce_dim_by_key(Param<To> out, CParam<Ti> in, CParam<Tk> key, bool change_nan, double nanval)
    {
        AF_ERROR("Reduce dimension by key not yet implemented", AF_ERR_NOT_CONFIGURED);
    }

    template<typename Ti, typename Tk, typename To, af_op_t op>
    void reduce_by_key(Param<Tk> reduced_keys, Param<To> reduced_vals, CParam<Tk> keys, CParam<Ti> vals, int dim, bool change_nan, double nanval)
    {
        void   *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        int *d_num_runs_out;
        CUDA_CHECK(cudaMalloc(&d_num_runs_out, sizeof(int)));

        int n_items = vals.dims[dim]; //temp calculation

        CUDA_CHECK(cudaFree(d_num_runs_out));
        switch (dim) {
            case 0: return reduce_first_by_key<Ti, Tk, To, op   >(reduced_keys, reduced_vals, keys, vals, change_nan, nanval);
            //case 1: return reduce_dim_by_key  <Ti, Tk, To, op, 1>(out, in, key, change_nan, nanval);
            //case 2: return reduce_dim_by_key  <Ti, Tk, To, op, 2>(out, in, key, change_nan, nanval);
            //case 3: return reduce_dim_by_key  <Ti, Tk, To, op, 3>(out, in, key, change_nan, nanval);
        }
    }

}
}
