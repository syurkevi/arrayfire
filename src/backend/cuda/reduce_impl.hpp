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
    void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out, const Array<Tk> &keys, const Array<Ti> &vals, const int dim, bool change_nan, double nanval)
    {
        //TODO: use unique set to determine output size ahead of time?
        //Array<Tk> fkey = keys;
        //fkey.modDims(dim4(keys.elements()));
        //Array<Tk> unique_keys = setUnique(fkey, is_sorted);
        //dim4 udims = unique_keys.dims();
        //dim4 odims = vals.dims();
        //odims[dim] = udims[0];

        dim4 odims = vals.dims();

        Array<Tk> temp_keys    = createEmptyArray<Tk>(odims);
        Array<Tk> reduced_keys = createEmptyArray<Tk>(odims);
        Array<To> temp_vals    = createEmptyArray<To>(odims);
        Array<To> reduced_vals = createEmptyArray<To>(odims);

        cub::DoubleBuffer<Tk> db_keys((Tk*)getDevicePtr(reduced_keys), (Tk*)getDevicePtr(temp_keys));
        cub::DoubleBuffer<To> db_vals((To*)getDevicePtr(reduced_vals), (To*)getDevicePtr(temp_vals));


        auto needs_reduction = memAlloc<int>(1);

        int n_reduced_after_initial;
        n_reduced_after_initial= kernel::reduce_first_by_key_launcher<Ti, Tk, To, op, 128>(reduced_keys, reduced_vals, keys, vals, change_nan, nanval);

        printf("n_reduced_after_initial: %d\n", n_reduced_after_initial);

        /*
        //prep temporary working memory
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, db_keys, db_vals, n_reduced_after_initial, 0, sizeof(int)*8, (cudaStream_t)0, false);

        void *d_temp_storage;
        size_t temp_storage_bytes = 0;

        CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        //perform sort by key
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, db_keys, db_vals, n_reduced_after_initial, 0, sizeof(int)*8, (cudaStream_t)0, false);

        //TODO: correct launch invocation
        bool needs_another_reduction = test_reduction_launcher();

        int n_reduced_after_iteration = n_reduced_after_initial;
        while(another_reduction) {

            //TODO: correct launch invocation
            n_reduced_after_iteration = reduce_first_by_key_launcher<>();
            //reduce_by_key_block<<<numBlocks, numThreads>>>(n_reduced, db_keys.Alternate(), db_vals.Alternate(), db_keys.Current(), db_vals.Current(), n_reduced_after_iteration);

            db_keys.selector = !db_keys.selector;
            db_vals.selector = !db_vals.selector;

            //TODO: correct launch invocation
            bool needs_another_reduction = test_reduction_launcher();
        }

        //TODO: prep final output pointer
        if(db_keys.Current() != *keys_out) {
            CUDA(cudaFree(*keys_out));
            *keys_out = db_keys.Current();
        }
        if(db_vals.Current() != *vals_out) {
            CUDA(cudaFree(*vals_out));
            *vals_out = db_vals.Current();
        }

        CUDA_CHECK(cudaMemcpy(nreduced, n_reduced, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(n_reduced));
        */
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
