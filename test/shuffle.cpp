/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;

TEST(Shuffle, Basic1D) {
    array in = range(af::dim4(10, 1, 1, 1));
    array out = af::shuffle(in);
}

TEST(Shuffle, Basic2D) {

    array in = range(af::dim4(16536, 4, 1, 1));
    array out = af::shuffle(in);
}

TEST(Shuffle, Basic3D) {

    array in = range(af::dim4(10, 2, 3, 1));
    array out = af::shuffle(in);
}

TEST(Shuffle, Basic4D) {

    //TODO: tests
    array in = range(af::dim4(10, 4, 5, 6));
    array out = af::shuffle(in);
}

TEST(Shuffle, SeedResetEqual) {
    af::setSeed(1);
    array in = range(af::dim4(10, 1, 1, 1));
    array before_seed = af::shuffle(in);

    af::setSeed(1);

    array after_seed = af::shuffle(in);
    ASSERT_ARRAYS_EQ(before_seed, after_seed);
}
