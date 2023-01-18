/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>  // common utility header for setting up kernels
                                // has the divup macro
#include <debug_oneapi.hpp>  // for debug only related oneapi validations

namespace arrayfire {
namespace oneapi {

namespace kernel {

// for oneapi create functor object for kernel
// canonical lambda function can also work for simpler kernels 
template<typename T>
class exampleKernel {
public:
    exampleKernel(sycl::accessor<T> cData, KParam cInfo,
                  sycl::accessor<T> aData, KParam aInfo,
                  sycl::accessor<T> bData, KParam bInfo,
                  af_someenum_t p)
        : cData_(cData) , cInfo_(cInfo)
        , aData_(aData) , aInfo_(aInfo)
        , bData_(bData) , bInfo_(bInfo)
        , p_(p) {} // copy input data in constructor

    // functor method called by handler during kernel invocation
    void operator()(sycl::nd_item<2> it) const {

        // get current thread global identifiers along required dimensions
        int i = it.get_global_id(0);
        int j = it.get_global_id(1);

        if (i < aInfo_.dims[0] && j < aInfo_.dims[1]) {
            // if needed use strides array to compute linear index of arrays
            int src1Idx = i + j * aInfo_.strides[1];
            int src2Idx = i + j * bInfo_.strides[1];
            int dstIdx  = i + j * cInfo_.strides[1];

            // kernel algorithm goes here
            cData_[dstIdx] = aData_[src1Idx] + bData_[src2Idx];
        }
    }

private:
    // data required for kernel computation
    // as private members of function object
    sycl::accessor<T> cData_;
    KParam cInfo_;
    sycl::accessor<T> aData_;
    KParam aInfo_;
    sycl::accessor<T> bData_;
    KParam bInfo_;
    af_someenum_t p_;
};

static const unsigned TX = 16;  // kernel launch config values
static const unsigned TY = 16;  // kernel launch config values

template<typename T>  // oneapi kernel wrapper function
void exampleFunc(Param<T> c, const Param<T> a, const Param<T> b, const af_someenum_t p) {

    auto local = sycl::range{TX, TY}; // set your oneapi work group parameters

    const int blk_x = divup(c.info.dims[0], local[0]);
    const int blk_y = divup(c.info.dims[1], local[1]);

    // configure global launch parameters
    auto global = sycl::range{blk_x * local[0],
                              blk_y * local[1]};

    getQueue().submit([&](sycl::handler &h) {
        // set up necessary accessors for kernel here
        // can be used to specify read/write buffer semantics
        // for task-graph optimizations
        auto c_acc = c.data->get_access(h);
        auto a_acc = a.data->get_access(h);
        auto b_acc = b.data->get_access(h);
        // also place to set up local memory allocation

        // executes kernel across launch configuration 
        h.parallel_for(
            sycl::nd_range{global, local},
            exampleKernel<T>(c_acc, c.info, a_acc, a.info, b_acc, b.info, p));
    });

    ONEAPI_DEBUG_FINISH(getQueue()); // Macro for post kernel launch checks
                                // these checks are done ONLY IN DEBUG mode
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
