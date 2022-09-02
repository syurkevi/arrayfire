/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <CL/sycl.hpp>
#include <kernel/KParam.hpp>

namespace oneapi {

struct Param {
    sycl::Buffer* data;
    KParam info;
    Param& operator=(const Param& other) = default;
    Param(const Param& other)            = default;
    Param(Param&& other)                 = default;

    // AF_DEPRECATED("Use Array<T>")
    Param();
    // AF_DEPRECATED("Use Array<T>")
    Param(sycl::Buffer* data_, KParam info_);
    ~Param() = default;
};

// AF_DEPRECATED("Use Array<T>")
Param makeParam(sycl::Buffer& mem, int off, const int dims[4],
                const int strides[4]);
}  // namespace oneapi
