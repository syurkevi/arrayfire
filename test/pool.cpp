/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <af/ml.h>

#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <string>
#include <vector>

using af::abs;
using af::pool2;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::randu;
using af::seq;
using af::span;
using af::sum;

using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

///TODO: c api tests
///TODO: cpp api tests
///TODO: type tests
///TODO: {invalid,} argument tests

TEST(Pool2, SNIPPET_pool2) {

    //! [ex_ml_pool2]

    // constant input data
    // {{1 2 3 4},
    //  {1 2 3 4},
    //  {1 2 3 4},
    //  {1 2 3 4}},
    float input_vals[16] = {1, 1, 1, 1,
                            2, 2, 2, 2,
                            3, 3, 3, 3,
                            4, 4, 4, 4};
    array input(4, 4, input_vals);

    const int win_sz  = 2;
    const int padding = 1;
    const int stride  = 2;
    array pooled = pool2(input,
                         win_sz, win_sz,
                         padding, padding,
                         stride, stride);
    af_print(pooled);
    // pooled == {{
    //              }}

    //! [ex_ml_pool2]

    //float expected_pooled[4] = {1.5, 1.5,
                                //2.5, 2.5};

    //array pooled_gold(2, 2, expected_pooled);
    //ASSERT_ARRAYS_NEAR(pooled, pooled_gold, 1e-5);

}
