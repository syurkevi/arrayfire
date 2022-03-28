/*******************************************************
 * Copyright (c) 2022, ArrayFire
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
Array<T> shuffle(const Array<T> &in, const uintl &seed, uintl &counter) {
    Array<T> out = createEmptyArray<T>(in.dims());
    Array<int> block_counter = createValueArray<int>(af::dim4(1), 0);
    kernel::shuffle<T>(out, in, block_counter, seed, counter);
    return out;
}

#define INSTANTIATE_SHUFFLE(T)                                   \
    template Array<T> shuffle(const Array<T> &in,                \
            const uintl &seed, uintl &counter);

INSTANTIATE_SHUFFLE(float)
//INSTANTIATE_SHUFFLE(double)
//INSTANTIATE_SHUFFLE(int)
//INSTANTIATE_SHUFFLE(uint)
//INSTANTIATE_SHUFFLE(intl)
//INSTANTIATE_SHUFFLE(uintl)
//INSTANTIATE_SHUFFLE(char)
//INSTANTIATE_SHUFFLE(uchar)
//INSTANTIATE_SHUFFLE(short)
//INSTANTIATE_SHUFFLE(ushort)
//INSTANTIATE_SHUFFLE(half)
//INSTANTIATE_SHUFFLE(cfloat)
//INSTANTIATE_SHUFFLE(cdouble)

}  // namespace cuda
