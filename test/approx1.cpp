/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/array.h>
#include <af/blas.h>
#include <af/complex.h>
#include <af/constants.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/exception.h>
#include <af/gfor.h>
#include <af/index.h>
#include <af/random.h>
#include <af/signal.h>
#include <af/traits.hpp>

#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <complex>
#include <string>
#include <vector>

using af::abs;
using af::approx1;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::randu;
using af::reorder;
using af::seq;
using af::span;
using af::sum;

using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Approx1 : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

// Create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// Register the type list
TYPED_TEST_CASE(Approx1, TestTypes);

template<typename T>
void approx1Test(string pTestFile, const unsigned resultIdx,
                 const af_interp_type method, bool isSubRef = false,
                 const vector<af_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray   = 0;
    af_array posArray  = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (af_dtype)dtype_traits<BT>::af_type));

    ASSERT_SUCCESS(af_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                             << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

    if (inArray != 0) af_release_array(inArray);
    if (posArray != 0) af_release_array(posArray);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

TYPED_TEST(Approx1, Approx1Nearest) {
    approx1Test<TypeParam>(string(TEST_DIR "/approx/approx1.test"), 0,
                           AF_INTERP_NEAREST);
}

TYPED_TEST(Approx1, Approx1Linear) {
    approx1Test<TypeParam>(string(TEST_DIR "/approx/approx1.test"), 1,
                           AF_INTERP_LINEAR);
}

template<typename T>
void approx1CubicTest(string pTestFile, const unsigned resultIdx,
                      const af_interp_type method, bool isSubRef = false,
                      const vector<af_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray   = 0;
    af_array posArray  = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (af_dtype)dtype_traits<BT>::af_type));
    ASSERT_SUCCESS(af_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;

    float max = real(outData[0]), min = real(outData[0]);
    for (int i = 1; i < (int)nElems; ++i) {
        min = (real(outData[i]) < min) ? real(outData[i]) : min;
        max = (real(outData[i]) > max) ? real(outData[i]) : max;
    }
    float range = max - min;
    ASSERT_GT(range, 0.f);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        double integral;
        // Test that control points are exact
        if ((std::modf(in[1][elIter], &integral) < 0.001) ||
            (std::modf(in[1][elIter], &integral) > 0.999)) {
            ret = abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                                 << outData[elIter] << "at: " << elIter << endl;
        } else {
            // Match intermediate values within a threshold
            ret =
                abs(tests[resultIdx][elIter] - outData[elIter]) < 0.035 * range;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                                 << outData[elIter] << "at: " << elIter << endl;
        }
    }

    // Delete
    delete[] outData;

    if (inArray != 0) af_release_array(inArray);
    if (posArray != 0) af_release_array(posArray);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

TYPED_TEST(Approx1, Approx1Cubic) {
    approx1CubicTest<TypeParam>(string(TEST_DIR "/approx/approx1_cubic.test"),
                                0, AF_INTERP_CUBIC_SPLINE);
}

///////////////////////////////////////////////////////////////////////////////
// Test Argument Failure Cases
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void approx1ArgsTest(string pTestFile, const af_interp_type method,
                     const af_err err) {
    SUPPORTED_TYPE_CHECK(T);
    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray  = 0;
    af_array posArray = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(),
                                   idims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (af_dtype)dtype_traits<BT>::af_type));

    ASSERT_EQ(err, af_approx1(&outArray, inArray, posArray, method, 0));

    if (inArray != 0) af_release_array(inArray);
    if (posArray != 0) af_release_array(posArray);
    if (outArray != 0) af_release_array(outArray);
}

TYPED_TEST(Approx1, Approx1NearestArgsPos2D) {
    approx1ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx1_pos2d.test"),
                               AF_INTERP_NEAREST, AF_ERR_SIZE);
}
TYPED_TEST(Approx1, Approx1LinearArgsPos2D) {
    approx1ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx1_pos2d.test"),
                               AF_INTERP_LINEAR, AF_ERR_SIZE);
}
TYPED_TEST(Approx1, Approx1ArgsInterpBilinear) {
    approx1ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx1.test"),
                               AF_INTERP_BILINEAR, AF_ERR_ARG);
}

template<typename T>
void approx1ArgsTestPrecision(string pTestFile, const unsigned,
                              const af_interp_type method) {
    SUPPORTED_TYPE_CHECK(T);
    vector<dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray  = 0;
    af_array posArray = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(),
                                   idims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    if ((af_dtype)dtype_traits<T>::af_type == c32 ||
        (af_dtype)dtype_traits<T>::af_type == c64) {
        ASSERT_EQ(AF_ERR_ARG,
                  af_approx1(&outArray, inArray, posArray, method, 0));
    } else {
        ASSERT_SUCCESS(af_approx1(&outArray, inArray, posArray, method, 0));
    }

    if (inArray != 0) af_release_array(inArray);
    if (posArray != 0) af_release_array(posArray);
    if (outArray != 0) af_release_array(outArray);
}

TYPED_TEST(Approx1, Approx1NearestArgsPrecision) {
    approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR "/approx/approx1.test"),
                                        0, AF_INTERP_NEAREST);
}

TYPED_TEST(Approx1, Approx1LinearArgsPrecision) {
    approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR "/approx/approx1.test"),
                                        1, AF_INTERP_LINEAR);
}

TYPED_TEST(Approx1, Approx1CubicArgsPrecision) {
    approx1ArgsTestPrecision<TypeParam>(
        string(TEST_DIR "/approx/approx1_cubic.test"), 2,
        AF_INTERP_CUBIC_SPLINE);
}

//////////////////////////////////////// CPP //////////////////////////////////
//
TEST(Approx1, CPP) {
    const unsigned resultIdx = 1;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR "/approx/approx1.test"),
                                numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    array input(idims, &(in[0].front()));
    array pos(pdims, &(in[1].front()));
    const af_interp_type method = AF_INTERP_LINEAR;
    array output                = approx1(input, pos, method, 0);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                             << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx1, CPPNearestBatch) {
    array input = randu(600, 10);
    array pos   = input.dims(0) * randu(100, 10);

    array outBatch = approx1(input, pos, AF_INTERP_NEAREST);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) =
            approx1(input(span, i), pos(span, i), AF_INTERP_NEAREST);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) =
            approx1(input(span, i), pos(span, i), AF_INTERP_NEAREST);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPLinearBatch) {
    array input = iota(dim4(10000, 20), c32);
    array pos   = input.dims(0) * randu(10000, 20);

    array outBatch = approx1(input, pos, AF_INTERP_LINEAR);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) =
            approx1(input(span, i), pos(span, i), AF_INTERP_LINEAR);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) =
            approx1(input(span, i), pos(span, i), AF_INTERP_LINEAR);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPCubicBatch) {
    array input = iota(dim4(10000, 20), c32);
    array pos   = input.dims(0) * randu(10000, 20);

    array outBatch = approx1(input, pos, AF_INTERP_CUBIC_SPLINE);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) =
            approx1(input(span, i), pos(span, i), AF_INTERP_CUBIC_SPLINE);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) =
            approx1(input(span, i), pos(span, i), AF_INTERP_CUBIC_SPLINE);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPNearestMaxDims) {
    const size_t largeDim = 65535 * 32 + 1;
    array input           = randu(1, largeDim);
    array pos             = input.dims(0) * randu(1, largeDim);
    array out             = approx1(input, pos, AF_INTERP_NEAREST);

    input = randu(1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    out   = approx1(input, pos, AF_INTERP_NEAREST);

    input = randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    out   = approx1(input, pos, AF_INTERP_NEAREST);

    SUCCEED();
}

TEST(Approx1, CPPLinearMaxDims) {
    const size_t largeDim = 65535 * 32 + 1;
    array input           = iota(dim4(1, largeDim), c32);
    array pos             = input.dims(0) * randu(1, largeDim);
    array outBatch        = approx1(input, pos, AF_INTERP_LINEAR);

    input    = iota(dim4(1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_LINEAR);

    input    = iota(dim4(1, 1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_LINEAR);

    SUCCEED();
}

TEST(Approx1, CPPCubicMaxDims) {
    const size_t largeDim = 65535 * 32 + 1;
    array input           = iota(dim4(1, largeDim), c32);
    array pos             = input.dims(0) * randu(1, largeDim);
    array outBatch        = approx1(input, pos, AF_INTERP_CUBIC);

    input    = iota(dim4(1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_CUBIC);

    input    = iota(dim4(1, 1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_CUBIC);

    SUCCEED();
}

TEST(Approx1, OtherDimLinear) {
    int start = 0;
    int stop  = 10000;
    int step  = 100;
    int num   = 1000;
    array xi  = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array yi  = 4 * xi - 3;
    array xo  = af::round(step * randu(num, 2, 2, 2));
    array yo  = 4 * xo - 3;
    for (int d = 1; d < 4; d++) {
        dim4 rdims(0, 1, 2, 3);
        rdims[0] = d;
        rdims[d] = 0;

        array yi_reordered =
            reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered =
            reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = approx1(yi_reordered, xo_reordered, d, start, step,
                                     AF_INTERP_LINEAR);
        array res =
            reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx1, OtherDimCubic) {
    float start = 0;
    float stop  = 100;
    float step  = 0.01;
    int num     = 1000;
    array xi    = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    array yi    = af::sin(xi);
    array xo    = af::round(step * af::randu(num, 2, 2, 2));
    array yo    = af::sin(xo);
    for (int d = 1; d < 4; d++) {
        dim4 rdims(0, 1, 2, 3);
        rdims[0] = d;
        rdims[d] = 0;

        array yi_reordered =
            reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered =
            reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = approx1(yi_reordered, xo_reordered, d, start, step,
                                     AF_INTERP_CUBIC);
        array res =
            reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx1, CPPUsage) {
    //! [ex_signal_approx1]

    // Input data array.
    float input_vals[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), input_vals);
    // [3 1 1 1]
    //     10.0000
    //     20.0000
    //     30.0000

    // Array of positions to be found along the first dimension.
    float pv[5] = {0.0f, 0.5, 1.0f, 1.5, 2.0f};
    array pos(dim4(5, 1), pv);
    // [5 1 1 1]
    //     0.0000
    //     0.5000
    //     1.0000
    //     1.5000
    //     2.0000

    // Perform interpolation across dimension 0.
    array interp = approx1(in, pos);
    // [5 1 1 1]
    //     10.0000
    //     15.0000
    //     20.0000
    //     25.0000
    //     30.0000

    //! [ex_signal_approx1]

    float civ[5] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f};
    array interp_gold(dim4(5, 1), civ);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPUniformUsage) {
    //! [ex_signal_approx1_uniform]

    float input_vals[9] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
                           60.0f, 70.0f, 80.0f, 90.0f};
    array in(dim4(3, 3), input_vals);
    // [3 3 1 1]
    //     10.0000    40.0000    70.0000
    //     20.0000    50.0000    80.0000
    //     30.0000    60.0000    90.0000

    // Array of positions to be found along the interpolation
    // dimension, `interp_dim`.
    float pv[5] = {0.0f, 0.5, 1.0f, 1.5f, 2.0f};
    array pos(dim4(5, 1), pv);
    // [5 1 1 1]
    //     0.0000
    //     0.5000
    //     1.0000
    //     1.5000
    //     2.0000

    // Define range of indices with which the input values will
    // correspond along the interpolation dimension.
    const double idx_start = 0.0;
    const double idx_step  = 1.0;

    // Perform interpolation across dimension 0.
    int interp_dim         = 0;
    array col_major_interp = approx1(in, pos, interp_dim, idx_start, idx_step);
    // [5 3 1 1]
    //     10.0000    40.0000    70.0000
    //     15.0000    45.0000    75.0000
    //     20.0000    50.0000    80.0000
    //     25.0000    55.0000    85.0000
    //     30.0000    60.0000    90.0000

    // Perform interpolation across dimension 1.
    interp_dim = 1;
    array row_major_interp =
        approx1(in, transpose(pos), interp_dim, idx_start, idx_step);
    // [3 5 1 1]
    //     10.0000    25.0000    40.0000    55.0000    70.0000
    //     20.0000    35.0000    50.0000    65.0000    80.0000
    //     30.0000    45.0000    60.0000    75.0000    90.0000

    //! [ex_signal_approx1_uniform]

    float civ[15] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 40.0f, 45.0f, 50.0f,
                     55.0f, 60.0f, 70.0f, 75.0f, 80.0f, 85.0f, 90.0f};
    array interp_gold_col(dim4(5, 3), civ);
    ASSERT_ARRAYS_EQ(col_major_interp, interp_gold_col);

    float riv[15] = {10.0f, 20.0f, 30.0f, 25.0f, 35.0f, 45.0f, 40.0f, 50.0f,
                     60.0f, 55.0f, 65.0f, 75.0f, 70.0f, 80.0f, 90.0f};
    array interp_gold_row(dim4(3, 5), riv);
    ASSERT_ARRAYS_EQ(row_major_interp, interp_gold_row);
}

TEST(Approx1, CPPDecimalStepRescaleGrid) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.f, 0.25f, 0.5f, 0.75f, 1.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 0.5;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPRepeatPos) {
    float inv[9] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
                    60.0f, 70.0f, 80.0f, 90.0f};
    array in(dim4(3, 3), inv);
    float pv[5] = {0.0f, 0.5f, 0.5f, 1.5f, 1.5f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[15] = {10.0f, 15.0f, 15.0f, 25.0f, 25.0f, 40.0f, 45.0f, 45.0f,
                    55.0f, 55.0f, 70.0f, 75.0f, 75.0f, 85.0f, 85.0f};
    array interp_gold(dim4(5, 3), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPNonMonotonicPos) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.5f, 1.0f, 1.5f, 0.0f, 2.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {15.0f, 20.0f, 25.0f, 10.0f, 30.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPMismatchingIndexingDim) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[4] = {0.0f, 0.5f, 1.0f, 2.0f};
    array pos(dim4(1, 4), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim          = 1;
    const float off_grid          = -1.0;
    array interp = approx1(in, pos, interp_dim, interp_grid_start,
                           interp_grid_step, AF_INTERP_LINEAR, off_grid);

    float iv[12] = {10.0f, 20.0f, 30.0f, -1.0f, -1.0f, -1.0f,
                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    array interp_gold(dim4(3, 4), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPNegativeGridStart) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = -1;
    const double interp_grid_step = 1;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {20.0f, 25.0f, 30.0f, 0.0f, 0.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPInterpolateBackwards) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    array pos(dim4(3, 1), pv);

    const int interp_grid_start   = in.elements() - 1;
    const double interp_grid_step = -1;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {30.0f, 25.0f, 20.0f, 15.0f, 10.0f};
    array interp_gold(dim4(3, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPStartOffGridAndNegativeStep) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.0f, -0.5f, -1.0f, -1.5f, -2.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = -1;
    const double interp_grid_step = -1;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {0.0f, 0.0f, 10.0f, 15.0f, 20.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPUniformInvalidStepSize) {
    try {
        float inv[3] = {10.0f, 20.0f, 30.0f};
        array in(dim4(3, 1), inv);
        float pv[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
        array pos(dim4(5, 1), pv);

        const int interp_grid_start   = 0;
        const double interp_grid_step = 0;
        const int interp_dim          = 0;
        array interp =
            approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);
        FAIL() << "Expected af::exception\n";
    } catch (af::exception& ex) { SUCCEED(); } catch (...) {
        FAIL() << "Expected af::exception\n";
    }
}

// Unless the sampling grid specifications - begin, step - are
// specified by the user, ArrayFire will assume a regular grid with a
// starting index of 0 and a step value of 1.
TEST(Approx1, CPPInfCheck) {
    array sampled(seq(0.0, 5.0, 0.5));
    sampled(0) = af::Inf;
    seq xo(0.0, 2.0, 0.25);
    array interp           = approx1(sampled, xo);
    array interp_augmented = join(1, xo, interp);

    float goldv[9] = {static_cast<float>(af::Inf),
                      static_cast<float>(af::Inf),
                      static_cast<float>(af::Inf),
                      static_cast<float>(af::Inf),
                      0.5f,
                      0.625f,
                      0.75f,
                      0.875f,
                      1.0f};
    array gold(dim4(9, 1), goldv);
    interp(af::isInf(interp)) = 0;
    gold(af::isInf(gold))     = 0;
    ASSERT_ARRAYS_EQ(interp, gold);
}

TEST(Approx1, CPPUniformInfCheck) {
    array sampled(seq(10.0, 50.0, 10.0));
    sampled(0) = af::Inf;
    seq xo(0.0, 8.0, 2.0);
    array interp   = approx1(sampled, xo, 0, 0, 2);
    float goldv[5] = {static_cast<float>(af::Inf), 20.0f, 30.0f, 40.0f, 50.0f};
    array gold(dim4(5, 1), goldv);
    interp(af::isInf(interp)) = 0;
    gold(af::isInf(gold))     = 0;
    ASSERT_ARRAYS_EQ(interp, gold);
}

TEST(Approx1, CPPEmptyPos) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    array pos;
    array interp = approx1(in, pos);
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interp.isempty());
}

TEST(Approx1, CPPEmptyInput) {
    array in;
    float pv[3] = {0.0f, 1.0f, 2.0f};
    array pos(dim4(3, 1), pv);

    array interp = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(interp.isempty());
}

TEST(Approx1, CPPEmptyPosAndInput) {
    array in;
    array pos;
    array interp = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interp.isempty());
}

template<typename T>
class Approx1V2 : public ::testing::Test {
   protected:
    typedef typename dtype_traits<T>::base_type BT;

    vector<T> h_gold_cast;
    vector<T> h_in_cast;
    vector<BT> h_pos_cast;

    dim4 gold_dims;
    dim4 in_dims;
    dim4 pos_dims;

    af_array gold;
    af_array in;
    af_array pos;

    Approx1V2() : gold(0), in(0), pos(0) {}

    void SetUp() {}

    void releaseArrays() {
        if (pos != 0) { ASSERT_SUCCESS(af_release_array(pos)); }
        if (in != 0) { ASSERT_SUCCESS(af_release_array(in)); }
        if (gold != 0) { ASSERT_SUCCESS(af_release_array(gold)); }
    }

    void TearDown() { releaseArrays(); }

    void setTestData(float* h_gold, dim4 gold_dims, float* h_in, dim4 in_dims,
                     float* h_pos, dim4 pos_dims) {
        releaseArrays();

        gold = 0;
        in   = 0;
        pos  = 0;

        this->gold_dims = gold_dims;
        this->in_dims   = in_dims;
        this->pos_dims  = pos_dims;

        for (int i = 0; i < gold_dims.elements(); ++i) {
            h_gold_cast.push_back(static_cast<T>(h_gold[i]));
        }
        for (int i = 0; i < in_dims.elements(); ++i) {
            h_in_cast.push_back(static_cast<T>(h_in[i]));
        }
        for (int i = 0; i < pos_dims.elements(); ++i) {
            h_pos_cast.push_back(static_cast<BT>(h_pos[i]));
        }

        ASSERT_SUCCESS(af_create_array(&gold, &h_gold_cast.front(),
                                       gold_dims.ndims(), gold_dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_create_array(&in, &h_in_cast.front(), in_dims.ndims(),
                                       in_dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_create_array(&pos, &h_pos_cast.front(),
                                       pos_dims.ndims(), pos_dims.get(),
                                       (af_dtype)dtype_traits<BT>::af_type));
    }

    void testSpclOutArray(TestOutputArrayType out_array_type) {
        SUPPORTED_TYPE_CHECK(T);

        af_array out = 0;
        TestOutputArrayInfo metadata(out_array_type);
        genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                           (af_dtype)dtype_traits<T>::af_type, &metadata);

        ASSERT_SUCCESS(af_approx1_v2(&out, in, pos, AF_INTERP_LINEAR, 0));
        ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);
    }

    void testSpclOutArrayUniform(TestOutputArrayType out_array_type) {
        SUPPORTED_TYPE_CHECK(T);

        af_array out = 0;
        TestOutputArrayInfo metadata(out_array_type);
        genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                           (af_dtype)dtype_traits<T>::af_type, &metadata);

        ASSERT_SUCCESS(af_approx1_uniform_v2(&out, in, pos, 0, 0.0, 1.0,
                                             AF_INTERP_LINEAR, 0.f));
        ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);
    }
};

TYPED_TEST_CASE(Approx1V2, TestTypes);

class SimpleTestData {
   public:
    static const int h_gold_size = 15;
    static const int h_in_size   = 9;
    static const int h_pos_size  = 5;

    vector<float> h_gold;
    vector<float> h_in;
    vector<float> h_pos;

    dim4 gold_dims;
    dim4 in_dims;
    dim4 pos_dims;

    SimpleTestData() : gold_dims(5, 3), in_dims(3, 3), pos_dims(5) {
        float gold_arr[h_gold_size] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f,
                                       40.0f, 45.0f, 50.0f, 55.0f, 60.0f,
                                       70.0f, 75.0f, 80.0f, 85.0f, 90.0f};

        float in_arr[h_in_size] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
                                   60.0f, 70.0f, 80.0f, 90.0f};

        float pos_arr[h_pos_size] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};

        h_gold.assign(gold_arr, gold_arr + h_gold_size);
        h_in.assign(in_arr, in_arr + h_in_size);
        h_pos.assign(pos_arr, pos_arr + h_pos_size);
    }
};

template<typename T>
class Approx1V2Simple : public Approx1V2<T> {
   protected:
    void SetUp() {
        SimpleTestData data;
        this->setTestData(&data.h_gold.front(), data.gold_dims,
                          &data.h_in.front(), data.in_dims, &data.h_pos.front(),
                          data.pos_dims);
    }
};

TYPED_TEST_CASE(Approx1V2Simple, TestTypes);

TYPED_TEST(Approx1V2Simple, UseNullOutputArray) {
    this->testSpclOutArray(NULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UseFullExistingOutputArray) {
    this->testSpclOutArray(FULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UseExistingOutputSubArray) {
    this->testSpclOutArray(SUB_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UseReorderedOutputArray) {
    this->testSpclOutArray(REORDERED_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseNullOutputArray) {
    this->testSpclOutArrayUniform(NULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseFullExistingOutputArray) {
    this->testSpclOutArrayUniform(FULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseExistingOutputSubArray) {
    this->testSpclOutArrayUniform(SUB_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseReorderedOutputArray) {
    this->testSpclOutArrayUniform(REORDERED_ARRAY);
}

class Approx1NullArgs : public ::testing::Test {
   protected:
    af_array out;
    af_array in;
    af_array pos;

    Approx1NullArgs() : out(0), in(0), pos(0) {}

    void SetUp() {
        SimpleTestData data;

        ASSERT_SUCCESS(af_create_array(&in, &data.h_in.front(),
                                       data.in_dims.ndims(), data.in_dims.get(),
                                       f32));
        ASSERT_SUCCESS(af_create_array(&pos, &data.h_pos.front(),
                                       data.pos_dims.ndims(),
                                       data.pos_dims.get(), f32));
    }

    void TearDown() {
        if (pos != 0) { ASSERT_SUCCESS(af_release_array(pos)); }
        if (in != 0) { ASSERT_SUCCESS(af_release_array(in)); }
    }
};

TEST_F(Approx1NullArgs, NullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG,
              af_approx1(out_ptr, this->in, this->pos, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, NullInputArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx1(&this->out, 0, this->pos, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, NullPosArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx1(&this->out, this->in, 0, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2NullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG, af_approx1_v2(out_ptr, this->in, this->pos,
                                        AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2NullInputArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx1_v2(&this->out, 0, this->pos, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2NullPosArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx1_v2(&this->out, this->in, 0, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, UniformNullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG, af_approx1_uniform(out_ptr, this->in, this->pos, 0,
                                             0.0, 1.0, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, UniformNullInputArray) {
    ASSERT_EQ(AF_ERR_ARG, af_approx1_uniform(&this->out, 0, this->pos, 0, 0.0,
                                             1.0, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, UniformNullPosArray) {
    ASSERT_EQ(AF_ERR_ARG, af_approx1_uniform(&this->out, this->in, 0, 0, 0.0,
                                             1.0, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2UniformNullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG,
              af_approx1_uniform_v2(out_ptr, this->in, this->pos, 0, 0.0, 1.0,
                                    AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2UniformNullInputArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx1_uniform_v2(&this->out, 0, this->pos, 0, 0.0, 1.0,
                                    AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2UniformNullPosArray) {
    ASSERT_EQ(AF_ERR_ARG, af_approx1_uniform_v2(&this->out, this->in, 0, 0, 0.0,
                                                1.0, AF_INTERP_LINEAR, 0.f));
}

#include <af/graphics.h>
TEST(Approx1, PCHIPCubicSimpleCos) {
    float h_x[11] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_y[11] = { 1, 0.5403, -0.41615, -0.98999, -0.65364, 0.28366, 0.96017, 0.7539, -0.1455, -0.91113, -0.83907};
    float h_xnew[200] = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9,
        1.95, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95,
        3, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4,
        4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5, 5.05,
        5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75, 5.8, 5.85, 5.9, 5.95, 6, 6.05, 6.1,
        6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55, 6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7, 7.05, 7.1, 7.15,
        7.2, 7.25, 7.3, 7.35, 7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8, 8.05, 8.1, 8.15, 8.2,
        8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 8.6, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95, 9, 9.05, 9.1, 9.15, 9.2, 9.25,
        9.3, 9.35, 9.4, 9.45, 9.5, 9.55, 9.6, 9.65, 9.7, 9.75, 9.8, 9.85, 9.9, 9.95};

    float h_pchip_cubic_gold[200] = {1, 0.98861, 0.9756, 0.96105, 0.94501, 0.92756, 0.90876, 0.88867, 0.86737, 0.84491,
                                     0.82135, 0.79678, 0.77125, 0.74482, 0.71756, 0.68955, 0.66083, 0.63148, 0.60157,
                                     0.57115, 0.5403, 0.50705, 0.46968, 0.42862, 0.3843, 0.33716, 0.28762, 0.23612,
                                     0.18308, 0.12894, 0.074124, 0.01907, -0.035794, -0.090037, -0.14323, -0.19494,
                                     -0.24473, -0.29218, -0.33685, -0.37832, -0.41615, -0.45268, -0.49032, -0.52875,
                                     -0.56764, -0.60668, -0.64554, -0.6839, -0.72143, -0.75782, -0.79273, -0.82586,
                                     -0.85686, -0.88543, -0.91123, -0.93395, -0.95327, -0.96885, -0.98038, -0.98754,
                                     -0.98999, -0.98873, -0.98503, -0.97903, -0.97085, -0.96064, -0.94853, -0.93464,
                                     -0.91912, -0.9021, -0.8837, -0.86406, -0.84333, -0.82162, -0.79907, -0.77581,
                                     -0.75199, -0.72773, -0.70316, -0.67842, -0.65364, -0.62638, -0.59437, -0.55808,
                                     -0.51794, -0.47441, -0.42792, -0.37892, -0.32786, -0.27519, -0.22134, -0.16676,
                                     -0.1119, -0.057211, -0.0031253, 0.049906, 0.10144, 0.15102, 0.19822, 0.24258,
                                     0.28366, 0.32403, 0.36626, 0.40992, 0.45461, 0.49987, 0.54531, 0.59047, 0.63495,
                                     0.67832, 0.72015, 0.76001, 0.79748, 0.83214, 0.86355, 0.8913, 0.91496, 0.9341,
                                     0.9483, 0.95713, 0.96017, 0.95947, 0.95741, 0.95406, 0.94946, 0.94367, 0.93676,
                                     0.92877, 0.91978, 0.90983, 0.89898, 0.8873, 0.87483, 0.86164, 0.84779, 0.83332,
                                     0.81831, 0.8028, 0.78686, 0.77054, 0.7539, 0.7342, 0.70898, 0.67871, 0.64388,
                                     0.60495, 0.56241, 0.51673, 0.4684, 0.41787, 0.36565, 0.31219, 0.25798, 0.2035,
                                     0.14922, 0.095618, 0.043174, -0.0076377, -0.056338, -0.10245, -0.1455, -0.18838,
                                     -0.23394, -0.28165, -0.331, -0.38145, -0.43247, -0.48353, -0.53411, -0.58368,
                                     -0.63171, -0.67767, -0.72103, -0.76128, -0.79786, -0.83027, -0.85797, -0.88044,
                                     -0.89714, -0.90754, -0.91113, -0.91112, -0.91106, -0.91089, -0.91055, -0.91,
                                     -0.90918, -0.90804, -0.90652, -0.90456, -0.90212, -0.89914, -0.89557, -0.89134,
                                     -0.88641, -0.88073, -0.87424, -0.86688, -0.8586, -0.84935 };
    af::array x(11, h_x);
    af::array y(11, h_y);
    af::array xnew(200, h_xnew);
    af::array pchip_gold(200, h_pchip_cubic_gold);

    //af_print(x);
    //af_print(y);
    //af_print(xnew);
    //af_print(pchip_gold);

    af::array af_ynew = af::approx1(y, xnew, AF_INTERP_CUBIC);
    af::Window w(512, 512, "cubic?");
    do {
        w.plot(xnew, pchip_gold);
        w.plot(xnew, af_ynew);
        //w.show();
    } while(!w.close());

}

TEST(Approx1, CubicSplineSimpleCos) {
    float h_x[11] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_y[11] = { 1, 0.5403, -0.41615, -0.98999, -0.65364, 0.28366, 0.96017, 0.7539, -0.1455, -0.91113, -0.83907};
    float h_xnew[200] = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9,
        1.95, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95,
        3, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4,
        4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5, 5.05,
        5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75, 5.8, 5.85, 5.9, 5.95, 6, 6.05, 6.1,
        6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55, 6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7, 7.05, 7.1, 7.15,
        7.2, 7.25, 7.3, 7.35, 7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8, 8.05, 8.1, 8.15, 8.2,
        8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 8.6, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95, 9, 9.05, 9.1, 9.15, 9.2, 9.25,
        9.3, 9.35, 9.4, 9.45, 9.5, 9.55, 9.6, 9.65, 9.7, 9.75, 9.8, 9.85, 9.9, 9.95};

    float h_cubic_spline_gold[200] = {
            1, 1.0031, 1.0027, 0.99906, 0.99218, 0.98221, 0.96926, 0.95345, 0.9349, 0.91372, 0.89003, 0.86394, 0.83556,
            0.80503, 0.77244, 0.73791, 0.70157, 0.66352, 0.62388, 0.58277, 0.5403, 0.49659, 0.45176, 0.40591, 0.35917,
            0.31165, 0.26346, 0.21473, 0.16556, 0.11608, 0.066389, 0.016617, -0.033126, -0.082724, -0.13206, -0.18102,
            -0.22949, -0.27736, -0.3245, -0.3708, -0.41615, -0.46043, -0.50356, -0.54546, -0.58605, -0.62524, -0.66295,
            -0.6991, -0.73362, -0.7664, -0.79739, -0.82648, -0.85361, -0.87869, -0.90164, -0.92237, -0.94081, -0.95688,
            -0.97048, -0.98155, -0.98999, -0.99575, -0.99886, -0.99936, -0.99729, -0.9927, -0.98565, -0.97617, -0.96431,
            -0.95013, -0.93366, -0.91496, -0.89406, -0.87102, -0.84588, -0.8187, -0.78951, -0.75836, -0.7253, -0.69038,
            -0.65364, -0.61515, -0.57501, -0.53335, -0.49031, -0.446, -0.40056, -0.35411, -0.30678, -0.25869, -0.20998,
            -0.16077, -0.11119, -0.061357, -0.011408, 0.038533, 0.088338, 0.13788, 0.18703, 0.23567, 0.28366, 0.33089,
            0.37726, 0.42267, 0.46703, 0.51026, 0.55227, 0.59295, 0.63222, 0.66999, 0.70617, 0.74066, 0.77338, 0.80422,
            0.83311, 0.85995, 0.88464, 0.9071, 0.92724, 0.94496, 0.96017, 0.9728, 0.98286, 0.99037, 0.99537, 0.99786,
            0.99789, 0.99548, 0.99065, 0.98344, 0.97386, 0.96194, 0.94772, 0.93121, 0.91244, 0.89144, 0.86824, 0.84286,
            0.81532, 0.78566, 0.7539, 0.72009, 0.68432, 0.64675, 0.60748, 0.56665, 0.5244, 0.48084, 0.4361, 0.39032,
            0.34363, 0.29614, 0.248, 0.19932, 0.15024, 0.10089, 0.051389, 0.0018721, -0.047534, -0.096701, -0.1455,
            -0.19381, -0.24154, -0.28862, -0.33496, -0.38049, -0.42512, -0.46877, -0.51137, -0.55284, -0.59308,
            -0.63203, -0.66961, -0.70573, -0.74032, -0.77329, -0.80457, -0.83407, -0.86172, -0.88743, -0.91113,
            -0.93273, -0.95217, -0.96934, -0.98418, -0.99661, -1.0065, -1.0139, -1.0186, -1.0206, -1.0198, -1.016,
            -1.0093, -0.99956, -0.98667, -0.97057, -0.95118, -0.92841, -0.90219, -0.87244 };

    af::array x(11, h_x);
    af::array y(11, h_y);
    af::array xnew(200, h_xnew);
    af::array spline_gold(200, h_cubic_spline_gold);
    //af_print(x);
    //af_print(y);
    //af_print(xnew);
    //af_print(pchip_gold);

    af::array af_ynew = af::approx1(y, xnew, AF_INTERP_CUBIC_SPLINE);
    af::Window w(512, 512, "cubic?");
    do {
        w.plot(xnew, spline_gold);
        w.plot(xnew, af_ynew);
        w.show();
    } while(!w.close());

}
