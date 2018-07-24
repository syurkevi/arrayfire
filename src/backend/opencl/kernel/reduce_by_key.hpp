/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <algorithm>
#include <string>
#include <mutex>
#include <map>
#include <kernel_headers/reduce_blocks_by_key_first.hpp>
#include <kernel_headers/reduce_by_key_compact.hpp>
#include <kernel_headers/reduce_by_key_needs_reduction.hpp>
#include <kernel_headers/reduce_by_key_boundary.hpp>
#include <kernel_headers/ops.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include <cache.hpp>
#include "names.hpp"
#include "config.hpp"
#include <memory.hpp>
#include <memory>

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace compute = boost::compute;

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using std::unique_ptr;

namespace opencl
{

namespace kernel
{

    template<typename Ti, typename Tk, typename To, af_op_t op>
    void reduce_by_key_dim(Param keys_out, Param vals_out, Param keys, Param vals, bool change_nan, double nanval, int dim)
    { }

    template<typename Ti, typename Tk, typename To, af_op_t op>
    void launch_reduce_blocks_by_key(cl::Buffer *reduced_block_sizes,
                                     Param keys_out, Param vals_out,
                                     const Param keys, const Param vals,
                                     int change_nan, double nanval, const int n, const uint threads_x)
    {
        std::string ref_name =
            std::string("reduce_blocks_by_key_0_") +
            std::string(dtype_traits<Ti>::getName()) +
            std::string("_") +
            std::string(dtype_traits<Tk>::getName()) +
            std::string("_") +
            std::string(dtype_traits<To>::getName()) +
            std::string("_") +
            std::to_string(op) +
            std::string("_") +
            std::to_string(threads_x);

        int device = getActiveDeviceId();

        kc_entry_t entry = kernelCache(device, ref_name);

        if (entry.prog==0 && entry.ker==0) {

            Binary<To, op> reduce;
            ToNumStr<To> toNumStr;

            std::ostringstream options;
            options << " -D To=" << dtype_traits<To>::getName()
                    << " -D Tk=" << dtype_traits<Tk>::getName()
                    << " -D Ti=" << dtype_traits<Ti>::getName()
                    << " -D T=To"
                    << " -D DIMX=" << threads_x
                    << " -D init=" << toNumStr(reduce.init())
                    << " -D " << binOpName<op>()
                    << " -D CPLX=" << af::iscplx<Ti>();

            if (std::is_same<Ti, double>::value ||
                std::is_same<Ti, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }

            const char *ker_strs[] = {ops_cl, reduce_blocks_by_key_first_cl};
            const int   ker_lens[] = {ops_cl_len, reduce_blocks_by_key_first_cl_len};
            Program prog;
            buildProgram(prog, 2, ker_strs, ker_lens, options.str());

            entry.prog = new Program(prog);
            entry.ker = new Kernel(*entry.prog, "reduce_blocks_by_key_first");

            addKernelToCache(device, ref_name, entry);
        }

        int numBlocks = divup(n, threads_x);

        NDRange local(threads_x);
        NDRange global(threads_x * numBlocks);

        auto reduceOp = KernelFunctor<Buffer,
                                      Buffer, KParam,
                                      Buffer, KParam,
                                      Buffer, KParam,
                                      Buffer, KParam,
                                      int, To, int>(*entry.ker);

        reduceOp(EnqueueArgs(getQueue(), global, local),
                 *reduced_block_sizes,
                 *keys_out.data, keys_out.info,
                 *vals_out.data, vals_out.info,
                         *keys.data, keys.info,
                         *vals.data, vals.info,
                 change_nan, scalar<To>(nanval), n);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename Tk, typename To, af_op_t op>
    void launch_final_boundary_reduce(cl::Buffer *reduced_block_sizes,
                                      Param keys_out, Param vals_out,
                                      const int n,const int numBlocks, const int threads_x)
    {
        std::string ref_name =
            std::string("final_boundary_reduce") +
            std::string(dtype_traits<Tk>::getName()) +
            std::string("_") +
            std::string(dtype_traits<To>::getName()) +
            std::string("_") +
            std::to_string(op) +
            std::string("_") +
            std::to_string(threads_x);

        int device = getActiveDeviceId();

        kc_entry_t entry = kernelCache(device, ref_name);

        if (entry.prog==0 && entry.ker==0) {

            Binary<To, op> reduce;
            ToNumStr<To> toNumStr;

            std::ostringstream options;
            options << " -D To=" << dtype_traits<To>::getName()
                    << " -D Tk=" << dtype_traits<Tk>::getName()
                    << " -D T=To"
                    << " -D DIMX=" << threads_x
                    << " -D init=" << toNumStr(reduce.init())
                    << " -D " << binOpName<op>()
                    << " -D CPLX=" << af::iscplx<To>();

            if (std::is_same<To, double>::value ||
                std::is_same<To, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }

            const char *ker_strs[] = {ops_cl, reduce_by_key_boundary_cl};
            const int   ker_lens[] = {ops_cl_len, reduce_by_key_boundary_cl_len};
            Program prog;
            buildProgram(prog, 2, ker_strs, ker_lens, options.str());

            entry.prog = new Program(prog);
            entry.ker = new Kernel(*entry.prog, "final_boundary_reduce");

            addKernelToCache(device, ref_name, entry);
        }

        NDRange local(threads_x);
        NDRange global(threads_x * numBlocks);

        auto reduceOp = KernelFunctor<Buffer,
                                      Buffer, KParam,
                                      Buffer, KParam,
                                      int>(*entry.ker);

        reduceOp(EnqueueArgs(getQueue(), global, local),
                 *reduced_block_sizes,
                 *keys_out.data, keys_out.info,
                 *vals_out.data, vals_out.info, n);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename Tk, typename To>
    void launch_compact(cl::Buffer *reduced_block_sizes,
                        Param keys_out, Param vals_out,
                        const Param keys, const Param vals,
                        const int numBlocks, const int threads_x)
    {
        std::string ref_name =
            std::string("compact_") +
            std::string(dtype_traits<Tk>::getName()) +
            std::string("_") +
            std::string(dtype_traits<To>::getName()) +
            std::string("_") +
            std::to_string(threads_x);

        int device = getActiveDeviceId();

        kc_entry_t entry = kernelCache(device, ref_name);

        if (entry.prog==0 && entry.ker==0) {

            ToNumStr<To> toNumStr;

            std::ostringstream options;
            options << " -D To=" << dtype_traits<To>::getName()
                    << " -D Tk=" << dtype_traits<Tk>::getName()
                    << " -D T=To"
                    << " -D DIMX=" << threads_x
                    << " -D CPLX=" << af::iscplx<To>();

            if (std::is_same<To, double>::value ||
                std::is_same<To, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }

            const char *ker_strs[] = {ops_cl, reduce_by_key_compact_cl};
            const int   ker_lens[] = {ops_cl_len, reduce_by_key_compact_cl_len};
            Program prog;
            buildProgram(prog, 2, ker_strs, ker_lens, options.str());

            entry.prog = new Program(prog);
            entry.ker = new Kernel(*entry.prog, "compact");

            addKernelToCache(device, ref_name, entry);
        }

        NDRange local(threads_x);
        NDRange global(threads_x * numBlocks);

        auto reduceOp = KernelFunctor<Buffer,
                                      Buffer, KParam,
                                      Buffer, KParam,
                                      Buffer, KParam,
                                      Buffer, KParam>(*entry.ker);

        reduceOp(EnqueueArgs(getQueue(), global, local),
                 *reduced_block_sizes,
                 *keys_out.data, keys_out.info,
                 *vals_out.data, vals_out.info,
                         *keys.data, keys.info,
                         *vals.data, vals.info);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename Tk>
    void launch_test_needs_reduction(cl::Buffer needs_reduction, cl::Buffer needs_boundary,
                        const Param keys, const int n, const int numBlocks, const int threads_x)
    {
        std::string ref_name =
            std::string("test_needs_reduction_") +
            std::string(dtype_traits<Tk>::getName()) +
            std::string("_") +
            std::to_string(threads_x);

        int device = getActiveDeviceId();

        kc_entry_t entry = kernelCache(device, ref_name);

        if (entry.prog==0 && entry.ker==0) {

            std::ostringstream options;
            options << " -D Tk=" << dtype_traits<Tk>::getName()
                    << " -D DIMX=" << threads_x;

            const char *ker_strs[] = {ops_cl, reduce_by_key_needs_reduction_cl};
            const int   ker_lens[] = {ops_cl_len, reduce_by_key_needs_reduction_cl_len};
            Program prog;
            buildProgram(prog, 2, ker_strs, ker_lens, options.str());

            entry.prog = new Program(prog);
            entry.ker = new Kernel(*entry.prog, "test_needs_reduction");

            addKernelToCache(device, ref_name, entry);
        }

        NDRange local(threads_x);
        NDRange global(threads_x * numBlocks);

        auto reduceOp = KernelFunctor<Buffer,
                                      Buffer,
                                      Buffer, KParam,
                                      int>(*entry.ker);

        reduceOp(EnqueueArgs(getQueue(), global, local),
                 needs_reduction, needs_boundary, *keys.data, keys.info, n);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename Ti, typename Tk, typename To, af_op_t op >
    int reduce_by_key_first(Param   keys_out, Param vals_out,
                            const Param keys, const Param vals,
                            bool change_nan, double nanval)
    {
        dim4 idims = dim4(keys.info.dims[0], keys.info.dims[1], keys.info.dims[2], keys.info.dims[3]);
        dim4 odims = dim4(vals.info.dims[0], vals.info.dims[1], vals.info.dims[2], vals.info.dims[3]);

        Array<Tk> t_reduced_keys = createEmptyArray<Tk>(odims);
        Array<To> t_reduced_vals = createEmptyArray<To>(odims);

        //flags determining more reduction is necessary
        auto needs_another_reduction        = memAlloc<int>(1);
        auto needs_block_boundary_reduction = memAlloc<int>(1);

        ////reset flags
        getQueue().enqueueFillBuffer<int>(*needs_another_reduction.get(), 0, 0, sizeof(int));
        getQueue().enqueueFillBuffer<int>(*needs_block_boundary_reduction.get(), 0, 0, sizeof(int));

        int nelems = idims[0];

        const unsigned int numThreads = 128;
        int numBlocks = divup(nelems, numThreads);

        auto reduced_block_sizes = memAlloc<int>(numBlocks);

        launch_reduce_blocks_by_key<Ti, Tk, To, op>(reduced_block_sizes.get(),
                                                    keys_out, vals_out,
                                                    keys, vals,
                                                    change_nan, nanval, nelems, numThreads);

        compute::command_queue c_queue(getQueue()());
        compute::buffer val_buf((*reduced_block_sizes.get())());

        compute::inclusive_scan(compute::make_buffer_iterator<int>(val_buf),
                                compute::make_buffer_iterator<int>(val_buf, numBlocks),
                                compute::make_buffer_iterator<int>(val_buf), c_queue);

        launch_compact<Tk, To>(reduced_block_sizes.get(),
                               t_reduced_keys, t_reduced_vals,
                               keys_out, vals_out,
                               numBlocks, numThreads);

        int n_reduced_host;
        getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true, (numBlocks - 1) * sizeof(int), sizeof(int), &n_reduced_host);
        printf("reduced_block_sizes %d\n", n_reduced_host);

        numBlocks = divup(n_reduced_host, numThreads);

        launch_test_needs_reduction<Tk>(*needs_another_reduction.get(), *needs_block_boundary_reduction.get(),
                                        t_reduced_keys, n_reduced_host, numBlocks, numThreads);

        int needs_another_reduction_host;
        getQueue().enqueueReadBuffer(*needs_another_reduction.get(), true, 0, sizeof(int), &needs_another_reduction_host);

        int needs_block_boundary_reduction_host;
        getQueue().enqueueReadBuffer(*needs_block_boundary_reduction.get(), true, 0, sizeof(int), &needs_block_boundary_reduction_host);

        printf("needs reduction?%d needs bbred?%d\n", needs_another_reduction_host, needs_block_boundary_reduction_host);
        //TODO: single do while
        while(needs_another_reduction_host || needs_block_boundary_reduction_host) {
            needs_block_boundary_reduction_host = 0;
            needs_another_reduction_host = 0;
            numBlocks = divup(n_reduced_host, numThreads);

            launch_reduce_blocks_by_key<To, Tk, To, op>(reduced_block_sizes.get(),
                                                        keys_out, vals_out,
                                                        t_reduced_keys, t_reduced_vals,
                                                        change_nan, nanval, n_reduced_host, numThreads);

            compute::inclusive_scan(compute::make_buffer_iterator<int>(val_buf),
                                    compute::make_buffer_iterator<int>(val_buf, numBlocks),
                                    compute::make_buffer_iterator<int>(val_buf), c_queue);

            launch_compact<Tk, To>(reduced_block_sizes.get(),
                                   t_reduced_keys, t_reduced_vals,
                                   keys_out, vals_out,
                                   numBlocks, numThreads);

            getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true, (numBlocks - 1) * sizeof(int), sizeof(int), &n_reduced_host);
            printf("reduced_block_sizes %d\n", n_reduced_host);

            //reset flags
            getQueue().enqueueFillBuffer<int>(*needs_another_reduction.get(), 0, 0, sizeof(int));
            getQueue().enqueueFillBuffer<int>(*needs_block_boundary_reduction.get(), 0, 0, sizeof(int));

            numBlocks = divup(n_reduced_host, numThreads);

            launch_test_needs_reduction<Tk>(*needs_another_reduction.get(), *needs_block_boundary_reduction.get(),
                                            t_reduced_keys, n_reduced_host, numBlocks, numThreads);

            getQueue().enqueueReadBuffer(*needs_another_reduction.get(), true, 0, sizeof(int), &needs_another_reduction_host);
            getQueue().enqueueReadBuffer(*needs_block_boundary_reduction.get(), true, 0, sizeof(int), &needs_block_boundary_reduction_host);

            if(needs_block_boundary_reduction_host) {
                launch_final_boundary_reduce<Tk, To, op>(reduced_block_sizes.get(),
                                                         t_reduced_keys, t_reduced_vals,
                                                         n_reduced_host, numBlocks, numThreads);

                compute::inclusive_scan(compute::make_buffer_iterator<int>(val_buf),
                                        compute::make_buffer_iterator<int>(val_buf, numBlocks),
                                        compute::make_buffer_iterator<int>(val_buf), c_queue);

                getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true, (numBlocks - 1) * sizeof(int), sizeof(int), &n_reduced_host);

                launch_compact<Tk, To>(reduced_block_sizes.get(),
                                       t_reduced_keys, t_reduced_vals,
                                       keys_out, vals_out,
                                       numBlocks, numThreads);

                //std::swap(t_reduced_keys.getData(), keys_out.data);
                //std::swap(t_reduced_vals.getData(), vals_out.data);
            }
        }

        return n_reduced_host;
    }

    template<af_op_t op, typename Ti, typename Tk, typename To>
    void reduce_by_key(Array<Tk> &keys_out,     Array<To> &vals_out,
                       const Array<Tk> &keys, const Array<Ti> &vals,
                       int dim, bool change_nan, double nanval)
    {
        if (dim == 0) {
            dim4 idims = keys.dims();
            dim4 odims = vals.dims();

            //allocate space for output arrays
            Array<Tk> reduced_keys = createEmptyArray<Tk>(odims);
            Array<To> reduced_vals = createEmptyArray<To>(odims);

            int n_reduced = reduce_by_key_first<Ti, Tk, To, op>(reduced_keys, reduced_vals, keys, vals, change_nan, nanval);

            odims[0] = n_reduced;
            std::vector<af_seq> index;
            for(int i=0; i<odims.ndims(); ++i) {
                af_seq s = { 0.0, (double)odims[i] - 1 , 1.0 };
                index.push_back(s);
            }

            keys_out = createSubArray<Tk>(reduced_keys, index, true);
            vals_out = createSubArray<To>(reduced_vals, index, true);

        } else {
            reduce_by_key_dim  <Ti, Tk, To, op>(keys_out, vals_out, keys, vals, change_nan, nanval, dim);
        }
    }

}

}
