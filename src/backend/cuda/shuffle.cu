/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/shuffle.hpp>
#include <af/dim4.hpp>
#include <cassert>

using common::half;

namespace cuda {

template<typename T>
Array<T> shuffle(const Array<T> &in,  const int dim) {
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::shuffle<T>(out, in, dim);
    return out;
}

#define INSTANTIATE(T)                                   \
    template Array<T> shuffle(                           \
        const Array<T> &in,  const int dim);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cuda
