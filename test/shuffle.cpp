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
    array in = range(af::dim4(8192, 1, 1, 1));


    array inn = in.copy();
    af::array tmp = af::randu(8192);
    af::array val, idx;
    af::sort(val, idx, tmp);
    af::array shuffled = inn(idx); // use original(idx, span) for rows

    array out = af::shuffle(in);
    //af_print(out)
}

TEST(Shuffle, Basic2D) {

    array in = range(af::dim4(10, 4, 1, 1));
    af_print(in);
    array out = af::shuffle(in);
    af_print(out);

    af::setSeed(0);
    //af::setDefaultRandomEngineType(AF_RANDOM_ENGINE_MERSENNE_GP11213);

    out = af::shuffle(in);
    af_print(out);
}

TEST(Shuffle, SeedResetEqual) {
    af::setSeed(1);
    array in = range(af::dim4(10, 1, 1, 1));
    array before_seed = af::shuffle(in);
    af_print(in);
    af_print(before_seed);

    af::setSeed(1);

    array after_seed = af::shuffle(in);
    af_print(in);
    af_print(after_seed);
    ASSERT_ARRAYS_EQ(before_seed, after_seed);
}
