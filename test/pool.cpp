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

// Documentation example
TEST(Pool2, SNIPPET_pool2) {
    //! [ex_ml_pool2]

    // input data that will be pooled
    // {{1 1  3 3},
    //  {2 1  3 4},
    //
    //  {2 2  4 4},
    //  {3 2  4 5}}

    float input_vals[16] = {1, 2, 2, 3,
                            1, 1, 2, 2,
                            3, 3, 4, 4,
                            3, 4, 4, 5};
    array input(4, 4, input_vals);

    const int win_sz  = 2;
    const int padding = 0;
    const int stride  = 2;

    array pooled = pool2(input,
                         win_sz, win_sz,
                         padding, padding,
                         stride, stride);

    // pooled == {{ 2, 4
    //              3, 5 }}

    // back-propagate gradient back to pooled values
    array incoming_gradient = af::constant(1, pooled.dims());

    array gradient = pool2Gradient(incoming_gradient,
                                   input, pooled,
                                   win_sz, win_sz,
                                   padding, padding,
                                   stride, stride);
    // gradient == {{0 0  0 0},
    //              {1 0  0 1},
    //
    //              {0 0  0 0},
    //              {1 0  0 1}}

    //! [ex_ml_pool2]

    float expected_pooled[4] = { 2, 3,
                                 4, 5 };

    float expected_gradient[16] = { 0, 1, 0, 1,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 1, 0, 1 };

    array pooled_gold(2, 2, expected_pooled);
    ASSERT_ARRAYS_NEAR(pooled, pooled_gold, 1e-5);

    array gradient_gold(4, 4, expected_gradient);
    ASSERT_ARRAYS_NEAR(gradient, gradient_gold, 1e-5);

}

TEST(Pool2, Pool2ForwardC) {
    float h_in[] = { 1, 2, 1, 2, 1, 2, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     3, 1, 1, 3, 1, 1, 3,
                     1, 1, 1, 1, 1, 1, 1,
                     1, 4, 1, 4, 1, 4, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     5, 1, 5, 1, 6, 1, 2 };

    af_array input = 0;
    dim4 idims(7, 7);
    ASSERT_SUCCESS(af_create_array(&input, h_in, idims.ndims(), idims.get(), (af_dtype) dtype_traits<float>::af_type));

    const int win_sz  = 3;
    const int padding = 1;
    const int stride  = 3;
    af_array pooled = 0;
    ASSERT_SUCCESS(af_pool2(&pooled, input,
                   win_sz, win_sz,
                   padding, padding,
                   stride, stride, AF_POOLING_MAX));

    float h_out[] = { 2, 2, 2,
                      4, 4, 4,
                      5, 6, 2 };

    dim4 pdims(3, 3);
    af_array gold_pooled = 0;
    ASSERT_SUCCESS(af_create_array(&gold_pooled, h_out, pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<float>::af_type));
    ASSERT_ARRAYS_NEAR(pooled, gold_pooled, 1E-5);

    if(input  != 0) af_release_array(input);
    if(pooled != 0) af_release_array(pooled);
    if(gold_pooled  != 0) af_release_array(gold_pooled);
}

TEST(Pool2, Pool2BackwardC) {
    float h_in[] = { 1, 2, 1, 2, 1, 2, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     3, 1, 1, 3, 1, 1, 3,
                     1, 1, 1, 1, 1, 1, 1,
                     1, 4, 1, 4, 1, 4, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     5, 1, 5, 1, 6, 1, 2 };

    af_array input = 0;
    dim4 idims(7, 7);
    ASSERT_SUCCESS(af_create_array(&input, h_in, idims.ndims(), idims.get(), (af_dtype) dtype_traits<float>::af_type));

    float h_out[] = { 2, 2, 2,
                      4, 4, 4,
                      5, 6, 2 };

    dim4 pdims(3, 3);
    af_array pooled = 0;
    ASSERT_SUCCESS(af_create_array(&pooled, h_out, pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<float>::af_type));

    af_array incoming_gradient = 0;
    ASSERT_SUCCESS(af_range(&incoming_gradient, pdims.ndims(), pdims.get(), 0, (af_dtype) dtype_traits<float>::af_type));

    const int win_sz  = 3;
    const int padding = 1;
    const int stride  = 3;

    af_array gradient = 0;
    ASSERT_SUCCESS(af_pool2Gradient(&gradient,
                                    incoming_gradient, input,
                                    pooled, win_sz, win_sz,
                                    padding, padding,
                                    stride, stride, AF_POOLING_MAX));


    float h_pool_grad[] = { 0, 0, 0, 1, 0, 2, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 1, 0, 2, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 1, 0, 2 };

    af_array gold_gradient = 0;
    ASSERT_SUCCESS(af_create_array(&gold_gradient, h_pool_grad, idims.ndims(), idims.get(), (af_dtype) dtype_traits<float>::af_type));

    ASSERT_ARRAYS_NEAR(gradient, gold_gradient, 1E-5);

    if(input  != 0) af_release_array(input);
    if(pooled != 0) af_release_array(pooled);
    if(incoming_gradient != 0) af_release_array(incoming_gradient);
    if(gradient != 0) af_release_array(gradient);
    if(gold_gradient != 0) af_release_array(gold_gradient);
}

template<typename T>
class Pool2 : public ::testing::Test
{ };

// Create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, short, ushort, uchar> TestTypes;

// Register the type list
TYPED_TEST_CASE(Pool2, TestTypes);

// CPP
template<typename T>
void Pool2ForwardTyped(int batch_c=1, int batch_n=1) {

    if (noDoubleTests<T>()) return;

    T h_in[] = { 1, 2, 1, 2, 1, 2, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 3, 1, 1, 3, 1, 1, 3,
                 1, 1, 1, 1, 1, 1, 1,
                 1, 4, 1, 4, 1, 4, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 5, 1, 5, 1, 6, 1, 2 };

    af_dtype ta = (af_dtype)dtype_traits<T>::af_type;
    array input(7, 7, h_in);
    input = tile(input, dim4(1, 1, batch_c, batch_n));

    const int win_sz  = 3;
    const int padding = 1;
    const int stride  = 3;
    array pooled = pool2(input,
                         win_sz, win_sz,
                         padding, padding,
                         stride, stride);

    T h_out[] = { 2, 2, 2,
                  4, 4, 4,
                  5, 6, 2 };

    array gold_pooled(3, 3, h_out);
    gold_pooled = tile(gold_pooled, dim4(1, 1, batch_c, batch_n));

    ASSERT_ARRAYS_NEAR(pooled, gold_pooled, 1E-5);
}

template<typename T>
void Pool2BackwardTyped(int batch_c=1, int batch_n=1) {
    T h_in[] = { 1, 2, 1, 2, 1, 2, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 3, 1, 1, 3, 1, 1, 3,
                 1, 1, 1, 1, 1, 1, 1,
                 1, 4, 1, 4, 1, 4, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 5, 1, 5, 1, 6, 1, 2 };

    array input(7, 7, h_in);
    input = tile(input, dim4(1, 1, batch_c, batch_n));

    T h_out[] = { 2, 2, 2,
                  4, 4, 4,
                  5, 6, 2 };

    array pooled(3, 3, h_out);
    pooled = tile(pooled, dim4(1, 1, batch_c, batch_n));
    array incoming_gradient = range(pooled.dims()) + 1 + range(pooled.dims(), 1);

    af_dtype ta = (af_dtype)dtype_traits<T>::af_type;
    incoming_gradient = incoming_gradient.as(ta);

    const int win_sz  = 3;
    const int padding = 1;
    const int stride  = 3;
    array gradient = pool2Gradient(incoming_gradient,
                                   input, pooled,
                                   win_sz, win_sz,
                                   padding, padding,
                                   stride, stride);
    T h_pool_grad[] = { 0, 1, 0, 2, 0, 3, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 2, 0, 3, 0, 4, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        3, 0, 0, 0, 4, 0, 5 };

    array pool_grad_gold(7, 7, h_pool_grad);
    pool_grad_gold = tile(pool_grad_gold, dim4(1, 1, batch_c, batch_n));
    ASSERT_ARRAYS_NEAR(gradient, pool_grad_gold, 1E-5);
}

TYPED_TEST(Pool2, Pool2Forward) {
    Pool2ForwardTyped<TypeParam>();
}

TYPED_TEST(Pool2, Pool2ForwardBatch) {
    const int c = 3;
    const int n = 32;
    Pool2ForwardTyped<TypeParam>(c, n);
}

TYPED_TEST(Pool2, Pool2Backward) {
    Pool2BackwardTyped<TypeParam>();
}

TYPED_TEST(Pool2, Pool2BackwardBatch) {
    const int c = 3;
    const int n = 32;
    Pool2BackwardTyped<TypeParam>(c, n);
}

TEST(Pool2, InvalidVectorPool2) {
    try
    {
        array input = af::range(7);

        const int win_sz  = 3;
        const int padding = 1;
        const int stride  = 3;
        array pooled = pool2(input,
                             win_sz, win_sz,
                             padding, padding,
                             stride, stride);
        FAIL() << "Expected af::exception\n";
    } catch (af::exception &ex) {
        SUCCEED();
    } catch(...) {
        FAIL() << "Expected af::exception\n";
    }
}

TEST(Pool2, InvalidGradientSizePool2) {
    try
    {
        array input    = af::range(dim4(7, 7));
        array output   = af::range(dim4(3, 3));
        array gradient = af::range(dim4(7, 7));

        const int win_sz  = 3;
        const int padding = 1;
        const int stride  = 3;
        array ograd = pool2Gradient(gradient,
                                    input, output,
                                    win_sz, win_sz,
                                    padding, padding,
                                    stride, stride);
        FAIL() << "Expected af::exception\n";
    } catch (af::exception &ex) {
        SUCCEED();
    } catch(...) {
        FAIL() << "Expected af::exception\n";
    }
}
