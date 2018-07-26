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

    //__shfl_up_sync wrapper
    template<typename T>
    __device__ T shfl_up_sync(unsigned mask, T var, int delta) {
        return __shfl_up_sync(mask, var, delta);
    }
    //specialization for cfloat
    template<> inline
    __device__ cuda::cfloat shfl_up_sync(unsigned mask, cuda::cfloat var, int delta) {
        cuda::cfloat res = { __shfl_up_sync(mask, var.x, delta), 
                             __shfl_up_sync(mask, var.y, delta)};
        return res;
    }
    //specialization for cdouble
    template<> inline
    __device__ cuda::cdouble shfl_up_sync(unsigned mask, cuda::cdouble var, int delta) {
        cuda::cdouble res = { __shfl_up_sync(mask, var.x, delta), 
                              __shfl_up_sync(mask, var.y, delta)};
        return res;
    }


    // Reduces keys across block boundaries
    template<typename Tk, typename To, af_op_t op>
    __global__ void final_boundary_reduce(int *reduced_block_sizes, Param<Tk> keys, Param<To> vals, const int n) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        Binary<To, op> reduce;

        if(tid == ((blockIdx.x + 1) * blockDim.x) - 1 && blockIdx.x < gridDim.x - 1) {
            Tk k0 = keys.ptr[tid];
            Tk k1 = keys.ptr[tid + 1];
            if(k0 == k1) {
                To v0 = vals.ptr[tid];
                To v1 = vals.ptr[tid + 1];
                vals.ptr[tid + 1] = reduce(v0, v1);
                reduced_block_sizes[blockIdx.x] = blockDim.x - 1;
            } else {
                reduced_block_sizes[blockIdx.x] = blockDim.x;
            }
        }

        //if last block, set block size to difference between n and block boundary
        if(threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
            reduced_block_sizes[blockIdx.x] = n - (blockIdx.x * blockDim.x);
        }
    }

    // Tests if data needs further reduction, including across block boundaries
    template<typename Tk>
    __global__ void test_needs_reduction(int *needs_another_reduction, int *needs_block_boundary_reduced, CParam<Tk> keys_in, const int n) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        Tk k;

        if(tid < n) {
            k = keys_in.ptr[tid];
        }

        int update_key = (k == __shfl_down_sync(0xFFFFFFFF, k , 1)) && (tid < (n-1)) && ((threadIdx.x % 32) < 31);
        int remaining_updates = __any_sync(0xFFFFFFFF, update_key);

        __syncthreads();

        //TODO: single per warp? change to assignment rather than atomicOr
        if(remaining_updates)
            atomicOr(needs_another_reduction, remaining_updates);

        //check across warp boundaries
        if((tid + 1) < n) {
            k = keys_in.ptr[tid+1];
        }

        update_key        = (k == __shfl_down_sync(0xFFFFFFFF, k , 1)) && ((tid+1) < (n-1)) && ((threadIdx.x % 32) < 31);
        remaining_updates = __any_sync(0xFFFFFFFF, update_key);

        //TODO: single per warp? change to assignment rather than atomicOr
        if(remaining_updates)
            atomicOr(needs_another_reduction, remaining_updates);

        //last thread in each block checks if any inter-block keys need further reduction
        if(tid == ((blockIdx.x + 1) * blockDim.x) - 1 && blockIdx.x < gridDim.x - 1) {
            int k0 = keys_in.ptr[tid];
            int k1 = keys_in.ptr[tid + 1];
            if(k0 == k1) {
                atomicOr(needs_block_boundary_reduced, 1);
            }
        }
    }

    // Compacts "incomplete" block-sized chunks of data in global memory
    template<typename Tk, typename To>
    __global__ void compact(int* reduced_block_sizes, Param<Tk> keys_out, Param<To> vals_out, CParam<Tk> keys_in, CParam<To> vals_in) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        Tk k;
        To v;

        // reduced_block_sizes should have inclusive sum of block sizes
        int nwrite   = (blockIdx.x == 0) ? reduced_block_sizes[0] : reduced_block_sizes[blockIdx.x] - reduced_block_sizes[blockIdx.x - 1];
        int writeloc = (blockIdx.x == 0) ? 0 : reduced_block_sizes[blockIdx.x - 1];

        k = keys_in.ptr[tid];
        v = vals_in.ptr[tid];

        if(threadIdx.x < nwrite) {
            keys_out.ptr[writeloc + threadIdx.x] = k;
            vals_out.ptr[writeloc + threadIdx.x] = v;
        }
    }

    const static int maxResPerWarp = 32; //assume dim 0, no NAN values

    // Reduces each block by key
    template<typename Ti, typename Tk, typename To, af_op_t op, uint DIMX>
    __global__
    static void reduce_blocks_by_key(int *reduced_block_sizes,
                                     Param<Tk>  reduced_keys,
                                     Param<To>  reduced_vals,
                                     CParam<Tk> keys,
                                     CParam<Ti> vals,
                                     int n, bool change_nan, To nanval) {

        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int laneid = tid % 32;

        const int nWarps = DIMX / 32;

        //
        // Allocate and initialize shared memory

        __shared__ int warpReduceSizes[nWarps]; //number of reduced elements in each warp

        __shared__ Tk  warpReduceKeys[nWarps][maxResPerWarp]; //reduced key segments for each warp
        __shared__ To  warpReduceVals[nWarps][maxResPerWarp]; //reduced values for each warp corresponding to each key segment

        // space to hold left/right-most keys of each reduced warp to check if reduction should happen accros boundaries
        __shared__ Tk  warpReduceLeftBoundaryKeys[nWarps];
        __shared__ Tk  warpReduceRightBoundaryKeys[nWarps];

        // space to hold left/right-most values of each reduced warp to check if reduction should happen accros boundaries
        __shared__ To  warpReduceLeftBoundaryVals[nWarps];
        __shared__ To  warpReduceRightBoundaryVals[nWarps];

        // space to compact and finalize all reductions within block
        __shared__ Tk  warpReduceKeysSmemFinal[nWarps * maxResPerWarp];
        __shared__ To  warpReduceValsSmemFinal[nWarps * maxResPerWarp];

        //
        // will hold final number of reduced elements in block
        __shared__ int reducedBlockSize;

        if(threadIdx.x == 0) {
            reducedBlockSize = 0;
        }
        __syncthreads();


        Binary<To, op> reduce;
        Transform<Ti, To, op> transform;

        // load keys and values to threads
        Tk k;
        To v;
        if(tid < n) {
            k = keys.ptr[tid];
            v = transform(vals.ptr[tid]);
            if (change_nan) v = !IS_NAN(v) ? v : nanval;
        }

        Tk eq_check = (k != shfl_up_sync(0xFFFFFFFF, k , 1));
        // mark threads containing unique keys
        char unique_flag = (eq_check || (laneid == 0)) && (tid < n);

        // scan unique flags to enumerate unique keys
        char unique_id = unique_flag;
        #pragma unroll
        for(int offset=1; offset<32; offset <<= 1) {
            char y = shfl_up_sync(0xFFFFFFFF, unique_id, offset);
            if(laneid >= offset)
                unique_id += y;
        }

        //
        // Reduce each warp by key
        char all_eq = (k == shfl_down_sync(0xFFFFFFFF, k , 1));
        if(__all_sync(0xFFFFFFFF, all_eq)) { // check special case of single key per warp
            v = reduce(v, shfl_down_sync(FULL_MASK, v , 1));
            v = reduce(v, shfl_down_sync(FULL_MASK, v , 2));
            v = reduce(v, shfl_down_sync(FULL_MASK, v , 4));
            v = reduce(v, shfl_down_sync(FULL_MASK, v , 8));
            v = reduce(v, shfl_down_sync(FULL_MASK, v , 16));
        } else {
            //preform reduction for each of the unique keys
            int eq_check = (unique_id == shfl_down_sync(0xFFFFFFFF, unique_id , 1));
            int update_key =  eq_check && (laneid < 31) && ((tid + 1) < n); //checks if this thread should perform a reduction
            unsigned shflmask = __ballot_sync(0xFFFFFFFF, update_key); //obtains mask of all threads that should be reduced
            shflmask |= (shflmask << 1); //shifts mask to include source threads that should participate in _shfl
            To uval = shfl_down_sync(shflmask, v , 1); //shfls data from neighboring threads
            v = reduce(v, (update_key ?  uval : Binary<To, op>::init())); //update if thread requires it

            eq_check = (unique_id == shfl_down_sync(0xFFFFFFFF, unique_id , 2));
            update_key = eq_check && (laneid < 30) && update_key && ((tid + 2) < n);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 2);
            uval = shfl_down_sync(shflmask, v , 2);
            v = reduce(v, (update_key ?  uval : Binary<To, op>::init()));

            eq_check = (unique_id == shfl_down_sync(0xFFFFFFFF, unique_id , 4));
            update_key = eq_check && (laneid < 28) && update_key && ((tid + 4) < n);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 4);
            uval = shfl_down_sync(shflmask, v , 4);
            v = reduce(v, (update_key ?  uval : Binary<To, op>::init()));

            eq_check = (unique_id == shfl_down_sync(0xFFFFFFFF, unique_id , 8));
            update_key = eq_check && (laneid < 24) && update_key && ((tid + 8) < n);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 8);
            uval = shfl_down_sync(shflmask, v , 8);
            v = reduce(v, (update_key ?  uval : Binary<To, op>::init()));

            eq_check = (unique_id == shfl_down_sync(0xFFFFFFFF, unique_id , 16));
            update_key = eq_check && (laneid < 16) && update_key && ((tid + 16) < n);
            shflmask = __ballot_sync(0xFFFFFFFF, update_key);
            shflmask |= (shflmask << 16);
            uval = shfl_down_sync(shflmask, v , 16);
            v = reduce(v, (update_key ?  uval : Binary<To, op>::init()));
        }


        const int warpid = threadIdx.x / 32;

        //last thread in warp has reduced warp size due to scan^
        if(laneid == 31) {
            warpReduceSizes[warpid] = unique_id;
        }

        //write left boundary values for each warp
        if(unique_flag && unique_id == 1) {
            warpReduceLeftBoundaryKeys[warpid] = k;
            warpReduceLeftBoundaryVals[warpid] = v;
        }

        //write right boundary values for each warp
        if(unique_flag && unique_id == warpReduceSizes[warpid]) {
            warpReduceRightBoundaryKeys[warpid] = k;
            warpReduceRightBoundaryVals[warpid] = v;
        }

        __syncthreads();

        // if rightmost thread, check next warp's kv,
        // invalidate self and change warpReduceSizes since first thread of next warp will update same key
        //TODO: what if extra empty warps???
        if(unique_flag && unique_id == warpReduceSizes[warpid] && warpid < nWarps - 1) {
            int tid_next_warp = (blockIdx.x * blockDim.x + (warpid + 1) * 32);
            // check within data range
            if( tid_next_warp < n &&  k == warpReduceLeftBoundaryKeys[warpid + 1]) {
                //disable writing from warps that need carry but aren't terminal
                if(warpReduceSizes[warpid] > 1 || warpid > 0) {
                    unique_flag = 0;
                }
            }
        }
        __syncthreads();

        // if leftmost thread, reduce carryover from previous warp(s) if needed
        if(unique_flag && unique_id == 1 && warpid > 0) {
            int test_wid = warpid - 1;
            while(test_wid >= 0 && k == warpReduceRightBoundaryKeys[test_wid]) {
                v = reduce(v, warpReduceRightBoundaryVals[test_wid]);
                --warpReduceSizes[test_wid];
                if(warpReduceSizes[test_wid] > 1)
                    break;

                --test_wid;
            }
        }

        if(unique_flag) {
            warpReduceKeys[warpid][unique_id-1] = k;
            warpReduceVals[warpid][unique_id-1] = v;
        }

        __syncthreads();

        // at this point, we have nWarps lists in shared memory with each list's size located in the warpReduceSizes[] array
        // perform warp-scan to determine each warp's write location
        int warpSzScan = 0;
        if(warpid == 0 && laneid < nWarps) {
            warpSzScan = warpReduceSizes[laneid];
            int activemask = 0xFFFFFFFF >> (32 - nWarps);
            #pragma unroll
            for(int offset=1; offset < 32; offset <<= 1) {
                char y = __shfl_up_sync(activemask, warpSzScan, offset);
                if(laneid >= offset)
                    warpSzScan += y;
            }
            warpReduceSizes[laneid] = warpSzScan;
            //final thread has final reduced size of block
            if(laneid == nWarps-1)
                reducedBlockSize = warpSzScan;
        }
        __syncthreads();

        // write reduced block size to global memory
        if(threadIdx.x == 0) {
            reduced_block_sizes[blockIdx.x] = reducedBlockSize;
        }


        // compact reduced keys and values before writing to global memory
        if(warpid > 0) {
            int wsz = warpReduceSizes[warpid] - warpReduceSizes[warpid - 1];
            if(laneid < wsz) {
                int warpOffset = warpReduceSizes[warpid - 1];
                warpReduceKeysSmemFinal[warpOffset + laneid] = warpReduceKeys[warpid][laneid];
                warpReduceValsSmemFinal[warpOffset + laneid] = warpReduceVals[warpid][laneid];
            }
        } else {
            int wsz = warpReduceSizes[warpid];
            if(laneid < wsz) {
                warpReduceKeysSmemFinal[laneid] = warpReduceKeys[0][laneid];
                warpReduceValsSmemFinal[laneid] = warpReduceVals[0][laneid];
            }
        }
        __syncthreads();

        //write reduced keys/values per-block
        if(threadIdx.x < reducedBlockSize) {
            reduced_keys.ptr[(blockIdx.x * blockDim.x) + threadIdx.x] = warpReduceKeysSmemFinal[threadIdx.x];
            reduced_vals.ptr[(blockIdx.x * blockDim.x) + threadIdx.x] = warpReduceValsSmemFinal[threadIdx.x];
        }

    }

    template<typename Ti, typename Tk, typename To, af_op_t op, int DIMX>
    int reduce_first_by_key_launcher(Param<Tk> keys_out, Param<To> vals_out, CParam<Tk> keys, CParam<Ti> vals, bool change_nan, double nanval) {

    }

    template<typename Ti, typename Tk, typename To, af_op_t op>
    int test_reduction_launcher(Param<To> out, CParam<Ti> in, CParam<Tk> key,
                                const uint blocks_x, const uint blocks_y, const uint threads_x,
                                bool change_nan, double nanval)
    {
    }


    template<typename Ti, typename Tk, typename To, af_op_t op>
    void reduce_first_by_key(Param<Tk> keys_out, Param<To> vals_out, CParam<Tk> keys, CParam<Ti> vals, bool change_nan, double nanval)
    {
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
