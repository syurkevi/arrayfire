/*******************************************************
 * Copyright (c) 2022 ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>
#include <kernel/config.hpp>
// #include <kernel/names.hpp>
// #include <kernel/scan_first.hpp>
// #include <kernel_headers/where.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>


namespace oneapi {
namespace kernel {

template<typename T, int dimensions>
using local_accessor =
    sycl::accessor<T, dimensions, sycl::access::mode::read_write,
                   sycl::access::target::local>;

template<typename T>
class transposeKernel {
   public:
    transposeKernel(sycl::accessor<T> oData, const KParam out,
                    const sycl::accessor<T> iData, const KParam in,
                    const int blocksPerMatX, const int blocksPerMatY,
                    const bool conjugate, const bool IS32MULTIPLE,
                    local_accessor<T, 1> shrdMem, sycl::stream debugStream)
        : oData_(oData)
        , out_(out)
        , iData_(iData)
        , in_(in)
        , blocksPerMatX_(blocksPerMatX)
        , blocksPerMatY_(blocksPerMatY)
        , conjugate_(conjugate)
        , IS32MULTIPLE_(IS32MULTIPLE)
        , shrdMem_(shrdMem)
        , debugStream_(debugStream) {}
    void operator()(sycl::nd_item<2> it) const {
    }
   private:
    sycl::accessor<T> oData_;
    KParam out_;
    sycl::accessor<T> iData_;
    KParam in_;
    int blocksPerMatX_;
    int blocksPerMatY_;
    sycl::stream debugStream_;
    bool conjugate_;
    bool IS32MULTIPLE_;
    local_accessor<T, 1> shrdMem_;
};

template<typename T>
static void where(Param<uint> &out, Param<T> &in) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    // uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
    // uint groups_y = divup(in.info.dims[1], threads_y);

    // Param rtmp;
    // Param otmp;

    // rtmp.info.dims[0] = groups_x;
    // otmp.info.dims[0] = in.info.dims[0];

    // rtmp.info.strides[0] = 1;
    // otmp.info.strides[0] = 1;

    // rtmp.info.offset = 0;
    // otmp.info.offset = 0;

    // for (int k = 1; k < 4; k++) {
    //     rtmp.info.dims[k]    = in.info.dims[k];
    //     rtmp.info.strides[k] = rtmp.info.strides[k - 1] * rtmp.info.dims[k - 1];

    //     otmp.info.dims[k]    = in.info.dims[k];
    //     otmp.info.strides[k] = otmp.info.strides[k - 1] * otmp.info.dims[k - 1];
    // }

    // int rtmp_elements = rtmp.info.strides[3] * rtmp.info.dims[3];
    // rtmp.data         = bufferAlloc(rtmp_elements * sizeof(uint));

    // int otmp_elements = otmp.info.strides[3] * otmp.info.dims[3];
    // otmp.data         = bufferAlloc(otmp_elements * sizeof(uint));

    // scanFirstLauncher<T, uint, af_notzero_t>(otmp, rtmp, in, false, groups_x,
    //                                          groups_y, threads_x);

    // // Linearize the dimensions and perform scan
    // Param ltmp        = rtmp;
    // ltmp.info.offset  = 0;
    // ltmp.info.dims[0] = rtmp_elements;
    // for (int k = 1; k < 4; k++) {
    //     ltmp.info.dims[k]    = 1;
    //     ltmp.info.strides[k] = rtmp_elements;
    // }

    // scanFirst<uint, uint, af_add_t>(ltmp, ltmp);

    // // Get output size and allocate output
    // uint total;
    // getQueue().enqueueReadBuffer(*rtmp.data, CL_TRUE,
    //                              sizeof(uint) * (rtmp_elements - 1),
    //                              sizeof(uint), &total);

    // out.data = bufferAlloc(total * sizeof(uint));

    // out.info.dims[0]    = total;
    // out.info.strides[0] = 1;
    // for (int k = 1; k < 4; k++) {
    //     out.info.dims[k]    = 1;
    //     out.info.strides[k] = total;
    // }

    // if (total > 0)
    //     get_out_idx<T>(out.data, otmp, rtmp, in, threads_x, groups_x, groups_y);

    // bufferFree(rtmp.data);
    // bufferFree(otmp.data);
}

}  // namespace kernel
}  // namespace oneapi
