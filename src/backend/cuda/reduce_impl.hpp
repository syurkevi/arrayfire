/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>

#undef _GLIBCXX_USE_INT128
#include <reduce.hpp>
#include <complex>
#include <kernel/reduce.hpp>
#include <kernel/reduce_by_key.hpp>
#include <err_cuda.hpp>
#include <set.hpp>
#include <cub/device/device_scan.cuh>

using std::swap;
using af::dim4;

namespace cuda
{
    template<af_op_t op, typename Ti, typename To>
    Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan, double nanval)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;
        Array<To> out = createEmptyArray<To>(odims);
        kernel::reduce<Ti, To, op>(out, in, dim, change_nan, nanval);
        return out;
    }

    template<af_op_t op, typename Ti, typename Tk, typename To>
    void reduce_by_key_dim(Array<Tk> &keys_out,     Array<To> &vals_out,
                           const Array<Tk> &keys, const Array<Ti> &vals,
                           int dim, bool change_nan, double nanval)
    { }

    template<af_op_t op, typename Ti, typename Tk, typename To>
    void reduce_by_key_first(Array<Tk> &keys_out,     Array<To> &vals_out,
                             const Array<Tk> &keys, const Array<Ti> &vals,
                             bool change_nan, double nanval)
    {
        dim4 idims = keys.dims();
        dim4 odims = vals.dims();

        //allocate space for output and temporary working arrays
        Array<Tk> reduced_keys   = createEmptyArray<Tk>(odims);
        Array<To> reduced_vals   = createEmptyArray<To>(odims);

        Array<Tk> t_reduced_keys = createEmptyArray<Tk>(odims);
        Array<To> t_reduced_vals = createEmptyArray<To>(odims);

        //flags determining more reduction is necessary
        auto needs_another_reduction        = memAlloc<int>(1);
        auto needs_block_boundary_reduction = memAlloc<int>(1);

        //reset flags
        CUDA_CHECK(cudaMemsetAsync(needs_another_reduction.get(), 0, sizeof(int), getActiveStream()));
        CUDA_CHECK(cudaMemsetAsync(needs_block_boundary_reduction.get(), 0, sizeof(int), getActiveStream()));

        int nelems = idims[0];

        const unsigned int numThreads = 128;
        int numBlocks = divup(nelems, numThreads);

        auto reduced_block_sizes = memAlloc<int>(numBlocks);

        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, reduced_block_sizes.get(), reduced_block_sizes.get(), numBlocks);
        auto d_temp_storage = memAlloc<char>(temp_storage_bytes);

        int n_reduced_host = nelems;
        int needs_another_reduction_host;
        int needs_block_boundary_reduction_host;

        bool first_pass = true;
        do {
            numBlocks = divup(n_reduced_host, numThreads);

            if(first_pass) {
                CUDA_LAUNCH((kernel::reduce_blocks_by_key<Ti, Tk, To, op, numThreads>), numBlocks, numThreads,
                             reduced_block_sizes.get(), reduced_keys, reduced_vals, keys, vals, nelems, change_nan, scalar<To>(nanval));
                POST_LAUNCH_CHECK();
                first_pass = false;
            } else {
                CUDA_LAUNCH((kernel::reduce_blocks_by_key<To, Tk, To, op, numThreads>), numBlocks, numThreads, reduced_block_sizes.get(), reduced_keys, reduced_vals, t_reduced_keys, t_reduced_vals, n_reduced_host, change_nan, scalar<To>(nanval));
                POST_LAUNCH_CHECK();
            }

            cub::DeviceScan::InclusiveSum((void*)d_temp_storage.get(), temp_storage_bytes, reduced_block_sizes.get(), reduced_block_sizes.get(), numBlocks);

            CUDA_LAUNCH((kernel::compact<Tk, To>), numBlocks, numThreads, reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals, reduced_keys, reduced_vals);
            POST_LAUNCH_CHECK();

            CUDA_CHECK(cudaMemcpy(&n_reduced_host, reduced_block_sizes.get() + (numBlocks - 1), sizeof(int), cudaMemcpyDeviceToHost));

            //reset flags
            CUDA_CHECK(cudaMemsetAsync(needs_another_reduction.get(), 0, sizeof(int), getActiveStream()));
            CUDA_CHECK(cudaMemsetAsync(needs_block_boundary_reduction.get(), 0, sizeof(int), getActiveStream()));

            numBlocks = divup(n_reduced_host, numThreads);
            CUDA_LAUNCH((kernel::test_needs_reduction<Tk>), numBlocks, numThreads,
                    needs_another_reduction.get(), needs_block_boundary_reduction.get(), t_reduced_keys, n_reduced_host);
            POST_LAUNCH_CHECK();

            CUDA_CHECK(cudaMemcpy(&needs_another_reduction_host, needs_another_reduction.get(), sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&needs_block_boundary_reduction_host, needs_block_boundary_reduction.get(), sizeof(int), cudaMemcpyDeviceToHost));

            if(needs_block_boundary_reduction_host && !needs_another_reduction_host) {
                CUDA_LAUNCH((kernel::final_boundary_reduce<Tk, To, op>), numBlocks, numThreads, reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals, n_reduced_host);
                POST_LAUNCH_CHECK();

                cub::DeviceScan::InclusiveSum((void*)d_temp_storage.get(), temp_storage_bytes, reduced_block_sizes.get(), reduced_block_sizes.get(), numBlocks);

                CUDA_CHECK(cudaMemcpy(&n_reduced_host, reduced_block_sizes.get() + (numBlocks - 1), sizeof(int), cudaMemcpyDeviceToHost));

                CUDA_LAUNCH((kernel::compact<Tk, To>), numBlocks, numThreads, reduced_block_sizes.get(), reduced_keys, reduced_vals, t_reduced_keys, t_reduced_vals);
                POST_LAUNCH_CHECK();

                swap(t_reduced_keys, reduced_keys);
                swap(t_reduced_vals, reduced_vals);
            }
        } while(needs_another_reduction_host || needs_block_boundary_reduction_host);

        odims[0] = n_reduced_host;
        std::vector<af_seq> index;
        for(int i=0; i<odims.ndims(); ++i) {
            af_seq s = { 0.0, (double)odims[i] - 1 , 1.0 };
            index.push_back(s);
        }

        keys_out = createSubArray<Tk>(t_reduced_keys, index, true);
        vals_out = createSubArray<To>(t_reduced_vals, index, true);
    }

    template<af_op_t op, typename Ti, typename Tk, typename To>
    void reduce_by_key(Array<Tk> &keys_out,     Array<To> &vals_out,
                       const Array<Tk> &keys, const Array<Ti> &vals,
                       const int dim, bool change_nan, double nanval)
    {
        if(dim == 0) {
            reduce_by_key_first<op, Ti, Tk, To>(keys_out, vals_out, keys, vals, change_nan, nanval);
        } else {
            reduce_by_key_dim<op, Ti, Tk, To>(keys_out, vals_out, keys, vals, dim, change_nan, nanval);
        }
    }

    template<af_op_t op, typename Ti, typename To>
    To reduce_all(const Array<Ti> &in, bool change_nan, double nanval)
    {
        return kernel::reduce_all<Ti, To, op>(in, change_nan, nanval);
    }
}

#define INSTANTIATE(Op, Ti, To)                                         \
    template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim, \
                                          bool change_nan, double nanval); \
    template void reduce_by_key<Op, Ti, int, To>(Array<int> &keys_out, Array<To> &vals_out, const Array<int> &keys, const Array<Ti> &vals, const int dim, \
                                          bool change_nan, double nanval); \
    template To reduce_all<Op, Ti, To>(const Array<Ti> &in, bool change_nan, double nanval);
