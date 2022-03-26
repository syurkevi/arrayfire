/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <backend.hpp>
#include <af/defines.h>

namespace cpu {

template<typename T>
Array<T> shuffle(const Array<T> &in, const uintl &seed, uintl &counter);

}  // namespace cpu
