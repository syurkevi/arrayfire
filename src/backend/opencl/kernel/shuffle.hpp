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
#include <common/kernel_cache.hpp>
#include <common/VarPhilox.hpp>

#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <shuffle.hpp>
#include <traits.hpp>
#include <af/defines.h>

#include <limits>
#include <random>
#include <string>
#include <vector>

#include <kernel_headers/shuffle.hpp>

namespace opencl {
namespace kernel {

static const int THREADS = 256;

struct lookback {
    unsigned long localsum;
    unsigned long inclusive_prefix;
    bool local_complete;
    bool prefix_complete;
};

template<typename T>
void shuffle(Param out, const Param in, Param block_counter, const uintl seed, uintl counter) {
    uint nBits = common::getVarPhiloxCipherBits(in.info.dims[0]);
    // number of elements to generate required
    uint capacity = (1 << nBits);

    uint64_t left_side_bits = nBits / 2;
    uint64_t left_side_mask = (1ull << left_side_bits) - 1;

    uint64_t right_side_bits = nBits - left_side_bits;
    uint64_t right_side_mask = (1ull << right_side_bits) - 1;

    const int threads = 32;

    int nblk_x = divup(capacity, threads);
    int nblk_y = out.info.dims[1];

    cl::NDRange local(threads, 1, 1);

    std::vector<TemplateArg> targs   = {TemplateTypename<T>()};
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto shuffleOp1 =
        common::getKernel("shufflePhilox1Pass", {shuffle_cl_src}, targs, options);

    auto shuffleOp2 =
        common::getKernel("shufflePhilox2Pass", {shuffle_cl_src}, targs, options);

    // create and initialize lookback table to 0
    auto lookback_table = memAlloc<char>(nblk_x * sizeof(lookback));
    getQueue().enqueueFillBuffer(*lookback_table, 0, 0, nblk_x * sizeof(lookback));

    cl::NDRange global1(local[0] * nblk_x, 1, 1);

    shuffleOp1(cl::EnqueueArgs(getQueue(), global1, local),
               *out.data, out.info,
               *in.data, in.info,
               *lookback_table,
               *block_counter.data,
               seed, counter,
               nBits, left_side_bits, left_side_mask, right_side_bits, right_side_mask);

    cl::NDRange global2(local[0] * nblk_x * out.info.dims[2],
                        local[1] * nblk_y * out.info.dims[3], 1);

    shuffleOp2(cl::EnqueueArgs(getQueue(), global2, local), *out.data, out.info,
               *in.data, in.info, *lookback_table, seed, counter,
               nBits, left_side_bits, left_side_mask, right_side_bits, right_side_mask,
               nblk_x, nblk_y);

    CL_DEBUG_FINISH(getQueue());
    counter += capacity;
}

}  // namespace kernel
}  // namespace opencl
