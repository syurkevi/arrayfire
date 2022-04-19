/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/Binary.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/ops.hpp>
#include <kernel_headers/scan_first.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {

template<typename Ti, typename To, af_op_t op>
static opencl::Kernel getScanFirstKernel(const std::string key,
                                         const bool isFinalPass,
                                         const uint threads_x,
                                         const bool inclusiveScan) {
    using std::string;
    using std::vector;

    const uint threads_y       = THREADS_PER_GROUP / threads_x;
    const uint SHARED_MEM_SIZE = THREADS_PER_GROUP;
    ToNumStr<To> toNumStr;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ti>(),   TemplateTypename<To>(),
        TemplateArg(isFinalPass), TemplateArg(op),
        TemplateArg(threads_x),   TemplateArg(inclusiveScan),
    };
    vector<string> compileOpts = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(DIMY, threads_y),
        DefineKeyFromStr(binOpName<op>()),
        DefineValue(SHARED_MEM_SIZE),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyValue(CPLX, af::iscplx<Ti>()),
        DefineKeyValue(IS_FINAL_PASS, (isFinalPass ? 1 : 0)),
        DefineKeyValue(INCLUSIVE_SCAN, inclusiveScan),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    return common::getKernel(key, std::array{ops_cl_src, scan_first_cl_src},
                             tmpltArgs, compileOpts);
}

template<typename Ti, typename To, af_op_t op>
static void scanFirstLauncher(Param &out, Param &tmp, const Param &in,
                              const bool isFinalPass, const uint groups_x,
                              const uint groups_y, const uint threads_x,
                              const bool inclusiveScan = true) {
    //printf("launcher\n");
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto scan = getScanFirstKernel<Ti, To, op>("scanFirst", isFinalPass,
                                               threads_x, inclusiveScan);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    scan(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *tmp.data,
         tmp.info, *in.data, in.info, groups_x, groups_y, lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
static void bcastFirstLauncher(Param &out, Param &tmp, const bool isFinalPass,
                               const uint groups_x, const uint groups_y,
                               const uint threads_x, const bool inclusiveScan) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto bcast = getScanFirstKernel<Ti, To, op>("bcastFirst", isFinalPass,
                                                threads_x, inclusiveScan);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    bcast(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
          *tmp.data, tmp.info, groups_x, groups_y, lim);
    CL_DEBUG_FINISH(getQueue());
}

struct lookback {
    unsigned long localsum;
    unsigned long inclusive_prefix;
    bool local_complete;
    bool prefix_complete;
};

template<typename Ti, typename To, af_op_t op>
static void scanFirst(Param &out, const Param &in,
                      const bool inclusiveScan = true) {
#if 0
    //printf("scanFirst\n");
    uint threads_x = nextpow2(std::max(32u, (uint)out.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(out.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(out.info.dims[1], threads_y);

    if (groups_x == 1) {
        scanFirstLauncher<Ti, To, op>(out, out, in, true, groups_x, groups_y,
                                      threads_x, inclusiveScan);

    } else {
        Param tmp           = out;
        tmp.info.dims[0]    = groups_x;
        tmp.info.strides[0] = 1;
        for (int k = 1; k < 4; k++) {
            tmp.info.strides[k] =
                tmp.info.strides[k - 1] * tmp.info.dims[k - 1];
        }

        int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];

        tmp.data = bufferAlloc(tmp_elements * sizeof(To));

        scanFirstLauncher<Ti, To, op>(out, tmp, in, false, groups_x, groups_y,
                                      threads_x, inclusiveScan);

        if (op == af_notzero_t) {
            scanFirstLauncher<To, To, af_add_t>(tmp, tmp, tmp, true, 1,
                                                groups_y, threads_x, true);
        } else {
            scanFirstLauncher<To, To, op>(tmp, tmp, tmp, true, 1, groups_y,
                                          threads_x, true);
        }

        bcastFirstLauncher<To, To, op>(out, tmp, true, groups_x, groups_y,
                                       threads_x, inclusiveScan);

        bufferFree(tmp.data);
    }
#else
    //const int threads_x = 32;
    //uint threads_x = nextpow2(std::max(32u, (uint)out.info.dims[0]));
    //threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_x = 256;
    //uint threads_y = THREADS_PER_GROUP / threads_x;
    uint threads_y = 1;

    int nblk_x = divup(out.info.dims[0], threads_x);
    int nblk_y = out.info.dims[1];

    cl::NDRange local(threads_x, 1, 1);

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ti>(),
        TemplateTypename<To>(),
        TemplateArg(op),
        TemplateArg(inclusiveScan)
    };

    //TODO: remove extra params once kernel works
    ToNumStr<To> toNumStr;
    const uint SHARED_MEM_SIZE = THREADS_PER_GROUP;
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(DIMY, threads_y),
        DefineKeyFromStr(binOpName<op>()),
        DefineValue(SHARED_MEM_SIZE),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyValue(CPLX, af::iscplx<Ti>()),
        DefineKeyValue(IS_FINAL_PASS, 1),
        DefineKeyValue(INCLUSIVE_SCAN, inclusiveScan),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());


    auto scanFirst = common::getKernel("scan1Pass", {ops_cl_src, scan_first_cl_src},
                                       tmpltArgs, compileOpts);

    // create and initialize lookback table to 0
    auto lookback_table = memAlloc<char>(nblk_x * sizeof(lookback));
    getQueue().enqueueFillBuffer(*lookback_table, 0, 0, nblk_x * sizeof(lookback));

    auto block_counter = memAlloc<uint>(1);
    getQueue().enqueueFillBuffer(*block_counter, 0, 0, sizeof(uint));

    cl::NDRange global(local[0] * nblk_x * out.info.dims[2],
                       local[1] * nblk_y * out.info.dims[3], 1);

    scanFirst(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, *lookback_table, *block_counter, nblk_x, nblk_y);

    CL_DEBUG_FINISH(getQueue());
#endif
}

}  // namespace kernel
}  // namespace opencl
