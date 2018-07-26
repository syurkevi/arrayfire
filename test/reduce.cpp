/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>
#include <algorithm>

using std::vector;
using std::complex;
using std::string;
using std::cout;
using std::endl;
using af::array;
using af::cfloat;
using af::cdouble;
using af::dim4;
using af::freeHost;


template<typename T>
class Reduce : public ::testing::Test
{
};

typedef ::testing::Types<float, double, cfloat, cdouble, uint, int, intl, uintl, uchar, short, ushort> TestTypes;
TYPED_TEST_CASE(Reduce, TestTypes);

typedef af_err (*reduceFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, reduceFunc af_reduce>
void reduceTest(string pTestFile, int off = 0, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

    vector<dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    dim4 dims       = numDims[0];

    vector<Ti> in(data[0].begin(), data[0].end());

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
        ASSERT_EQ(AF_SUCCESS, af_release_array(tempArray));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
    }

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {

        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_reduce(&outArray, inArray, d + off));

        af_dtype t;
        af_get_type(&t, outArray);

        // Get result
        vector<To> outData(dims.elements());
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)&outData.front(), outArray));

        size_t nElems = currGoldBar.size();
        if(std::equal(currGoldBar.begin(), currGoldBar.end(), outData.begin()) == false)
        {
            for (size_t elIter = 0; elIter < nElems; ++elIter) {

                EXPECT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                                << " for dim " << d + off << endl;
            }
            af_print_array(outArray);
            for(int i = 0; i < (int)nElems; i++) {
                cout << currGoldBar[i] << ", ";
            }

            cout << endl;
            for(int i = 0; i < (int)nElems; i++) {
                cout << outData[i] << ", ";
            }
            FAIL();
        }

        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

template<typename T,reduceFunc OP>
struct promote_type {
    typedef T type;
};

// char and uchar are promoted to int for sum and product
template<> struct promote_type<uchar , af_sum>       { typedef uint type; };
template<> struct promote_type<char  , af_sum>       { typedef uint type; };
template<> struct promote_type<short , af_sum>       { typedef int  type; };
template<> struct promote_type<ushort, af_sum>       { typedef uint type; };
template<> struct promote_type<uchar , af_product>   { typedef uint type; };
template<> struct promote_type<char  , af_product>   { typedef uint type; };
template<> struct promote_type<short , af_product>   { typedef int  type; };
template<> struct promote_type<ushort, af_product>   { typedef uint type; };

#define REDUCE_TESTS(FN)                                                                    \
    TYPED_TEST(Reduce,Test_##FN)                                                            \
    {                                                                                       \
        reduceTest<TypeParam, typename promote_type<TypeParam, af_##FN>::type, af_##FN>(    \
            string(TEST_DIR"/reduce/"#FN".test")                                            \
            );                                                                              \
    }                                                                                       \

REDUCE_TESTS(sum);
REDUCE_TESTS(min);
REDUCE_TESTS(max);

#undef REDUCE_TESTS
#define REDUCE_TESTS(FN, OT)                        \
    TYPED_TEST(Reduce,Test_##FN)                    \
    {                                               \
        reduceTest<TypeParam, OT, af_##FN>(         \
            string(TEST_DIR"/reduce/"#FN".test")    \
            );                                      \
    }                                               \

REDUCE_TESTS(any_true, unsigned char);
REDUCE_TESTS(all_true, unsigned char);
REDUCE_TESTS(count, unsigned);

#undef REDUCE_TESTS

TEST(Reduce,Test_Reduce_Big0)
{
    if (noDoubleTests<int>()) return;

    reduceTest<int, int, af_sum>(
        string(TEST_DIR"/reduce/big0.test"),
        0
        );
}

/*
TEST(Reduce,Test_Reduce_Big1)
{
    if (noDoubleTests<int>()) return;

    reduceTest<int, int, af_sum>(
        string(TEST_DIR"/reduce/big1.test"),
        1
        );
}
*/

/////////////////////////////////// CPP //////////////////////////////////
//
typedef af::array (*ReductionOp)(const af::array&, const int);

using af::NaN;
using af::allTrue;
using af::anyTrue;
using af::constant;
using af::count;
using af::iota;
using af::max;
using af::min;
using af::product;
using af::randu;
using af::round;
using af::seq;
using af::span;
using af::sum;

template<typename Ti, typename To, ReductionOp reduce>
void cppReduceTest(string pTestFile)
{
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

    vector<dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    dim4 dims       = numDims[0];

    vector<Ti> in(data[0].begin(), data[0].end());

    array input(dims, &in.front());

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {

        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        array output = reduce(input, d);

        // Get result
        vector<To> outData(dims.elements());
        output.host((void*)&outData.front());

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                            << " for dim " << d << endl;
        }
    }
}

TEST(Reduce, Test_Sum_Scalar_MaxDim)
{
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(sum<float>(A, 1), largeDim);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(sum<float>(A, 2), largeDim);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(sum<float>(A, 3), largeDim);
}

TEST(Reduce, Test_Min_Scalar_MaxDim)
{
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A = iota(dim4(1, largeDim, 1, 1));
    ASSERT_EQ(min(A, 1).scalar<float>(), 0.f);
    A = iota(dim4(1, 1, largeDim, 1));
    ASSERT_EQ(min(A, 2).scalar<float>(), 0.f);
    A = iota(dim4(1, 1, 1, largeDim));
    ASSERT_EQ(min(A, 3).scalar<float>(), 0.f);
}

TEST(Reduce, Test_Max_Scalar_MaxDim)
{
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A = iota(dim4(1, largeDim, 1, 1));
    ASSERT_EQ(max(A, 1).scalar<float>(), largeDim - 1);
    A = iota(dim4(1, 1, largeDim, 1));
    ASSERT_EQ(max(A, 2).scalar<float>(), largeDim - 1);
    A = iota(dim4(1, 1, 1, largeDim));
    ASSERT_EQ(max(A, 3).scalar<float>(), largeDim - 1);
}

TEST(Reduce, Test_anyTrue_Scalar_MaxDim)
{
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(anyTrue(A, 1).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(anyTrue(A, 2).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(anyTrue(A, 3).scalar<char>(), 1);
}

TEST(Reduce, Test_allTrue_Scalar_MaxDim)
{
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(allTrue(A, 1).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(allTrue(A, 2).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(allTrue(A, 3).scalar<char>(), 1);
}

TEST(Reduce, Test_count_Scalar_MaxDim)
{
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(count(A, 1).scalar<unsigned int>(), largeDim);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(count(A, 2).scalar<unsigned int>(), largeDim);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(count(A, 3).scalar<unsigned int>(), largeDim);
}

#define CPP_REDUCE_TESTS(FN, FNAME, Ti, To)         \
    TEST(Reduce, Test_##FN##_CPP)                   \
    {                                               \
        cppReduceTest<Ti, To, FN>(                  \
            string(TEST_DIR"/reduce/"#FNAME".test") \
            );                                      \
    }

CPP_REDUCE_TESTS(sum, sum, float, float);
CPP_REDUCE_TESTS(min, min, float, float);
CPP_REDUCE_TESTS(max, max, float, float);
CPP_REDUCE_TESTS(anyTrue, any_true, float, unsigned char);
CPP_REDUCE_TESTS(allTrue, all_true, float, unsigned char);
CPP_REDUCE_TESTS(count, count, float, unsigned);

//
// Reduce By Key tests
//

// ReduceByKey helper methods
//TODO: move to test data
std::tuple<af::array, af::array> prepKeyValuePairs_10000Test() {
    af::array keys, vals;
    vals = af::constant(1, 1000000);
    keys = af::constant(0, 1000000);
    //vals = af::constant(1, 3);
    //keys = af::constant(0, 3);
    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> prepKeyValuePairs_10000_2Test() {
    af::array keys, vals;
    vals = af::constant(1, 1000000);
    //keys = af::constant(0, 100000);
    //vals = af::constant(1, 3);
    //keys = af::constant(0, 3);
    //keys = af::flat(af::range(af::dim4(2, 100000/2), 1, s32));
    keys = af::flat(af::range(af::dim4(1000000/2, 2), 1, s32));
    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> prepKeyValuePairs_10Random(const int n) {
    af::array keys, vals;
    vals = af::randn(n);
    keys = af::flat(af::range(af::dim4(n/10, 10), -1, s32));

    //shufle keys
    int len = keys.dims(0);
    af::array tmp = af::randu(len, 1);
    af::array val, idx;
    af::sort(val, idx, tmp);
    keys = keys(idx);
    //assert(keys.dims(0) == vals.dims(0));
    return std::make_tuple(keys.as(s32), vals);
}

std::tuple<af::array, af::array> prepKeyValuePairs_10Contiguous(const int n) {
    af::array keys, vals;
    vals = af::randn(n);
    keys = af::flat(af::range(af::dim4(10, n/10), 1, s32));
    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> prepKeyValuePairs_500Random(const int n) {
    af::array keys, vals;
    vals = af::randn(n);
    keys = af::flat(af::range(af::dim4(n/500, 500), -1, s32));

    //shufle keys
    int len = keys.dims(0);
    af::array tmp = af::randu(len, 1);
    af::array val, idx;
    af::sort(val, idx, tmp);
    keys = keys(idx);
    return std::make_tuple(keys.as(s32), vals);
}

std::tuple<af::array, af::array> prepKeyValuePairs_500Contiguous(const int n) {
    af::array keys, vals;
    vals = af::randn(n);
    keys = af::flat(af::range(af::dim4(500, n/500), 1, s32));
    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest0_KeyVals() {
    const int testSz = 8;
    const int   h_keys[testSz] = { 0, 2, 2, 9, 5, 5, 5, 8 };
    const float h_vals[testSz] = { 0, 7, 1, 6, 2, 5, 3, 4 };
    af::array keys(testSz, h_keys);
    af::array vals(testSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest0_Gold() {
    const int resSz = 5;
    const int   h_keys[resSz] = { 0, 2, 9,  5, 8 };
    const float h_vals[resSz] = { 0, 8, 6, 10, 4 };
    af::array keys(resSz, h_keys);
    af::array vals(resSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest1_KeyVals() {
    const int testSz = 64;
    const int   h_keys[testSz] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float h_vals[testSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    af::array keys(testSz, h_keys);
    af::array vals(testSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest1_Gold() {
    const int resSz = 1;
    const int   h_keys[resSz] = {  0 };
    const float h_vals[resSz] = { 64 };
    af::array keys(resSz, h_keys);
    af::array vals(resSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest2_KeyVals() {
    const int testSz = 64;
    const int   h_keys[testSz] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    const float h_vals[testSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    af::array keys(testSz, h_keys);
    af::array vals(testSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest2_Gold() {
    const int resSz = 32;
    const int   h_keys[resSz] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,};
    const float h_vals[resSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 33};
    af::array keys(resSz, h_keys);
    af::array vals(resSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest3_KeyVals() {
    const int testSz = 256;
    const int   h_keys[testSz] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    const float h_vals[testSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    af::array keys(testSz, h_keys);
    af::array vals(testSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest3_Gold() {
    const int resSz = 32;
    const int   h_keys[resSz] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2};
    const float h_vals[resSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 225};
    af::array keys(resSz, h_keys);
    af::array vals(resSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest4_KeyVals() {
    const int testSz = 32;
    const int   h_keys[testSz] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float h_vals[testSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    af::array keys(testSz, h_keys);
    af::array vals(testSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest4_Gold() {
    const int resSz = 1;
    const int   h_keys[resSz] = { 0 };
    const float h_vals[resSz] = { 32 };
    af::array keys(resSz, h_keys);
    af::array vals(resSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest5_KeyVals() {
    const int testSz = 32;
    const int   h_keys[testSz] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    const float h_vals[testSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    af::array keys(testSz, h_keys);
    af::array vals(testSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest5_Gold() {
    const int resSz = 32;
    const int   h_keys[resSz] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    const float h_vals[resSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    af::array keys(resSz, h_keys);
    af::array vals(resSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest6_KeyVals() {
    const int testSz = 128;
    const int   h_keys[testSz] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float h_vals[testSz] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    af::array keys(testSz, h_keys);
    af::array vals(testSz, h_vals);

    return std::make_tuple(keys, vals);
}

std::tuple<af::array, af::array> getTest6_Gold() {
    const int resSz = 1;
    const int   h_keys[resSz] = {   0 };
    const float h_vals[resSz] = { 128 };
    af::array keys(resSz, h_keys);
    af::array vals(resSz, h_vals);

    return std::make_tuple(keys, vals);
}


// ReduceByKey tests
TEST(ReduceByKey, ReduceByKeySimple) {
    af::array keys, vals;
    std::tie(keys,vals) = getTest0_KeyVals();

    af::array keyResGold, valsReducedGold;
    std::tie(keyResGold, valsReducedGold) = getTest0_Gold();

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>(valsReducedGold == valsReduced));
}

TEST(ReduceByKey, FullwarpSingelval) {
    af::array keys, vals;
    std::tie(keys,vals) = getTest4_KeyVals();

    af::array keyResGold, valsReducedGold;
    std::tie(keyResGold, valsReducedGold) = getTest4_Gold();

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>(valsReducedGold == valsReduced));
}

TEST(ReduceByKey, FullwarpUniquevals) {
    af::array keys, vals;
    std::tie(keys,vals) = getTest5_KeyVals();

    af::array keyResGold, valsReducedGold;
    std::tie(keyResGold, valsReducedGold) = getTest5_Gold();

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>(valsReducedGold == valsReduced));
}

TEST(ReduceByKey, WarpszReduceFull) {
    af::array keys, vals;
    std::tie(keys,vals) = getTest1_KeyVals();

    af::array keyResGold, valsReducedGold;
    std::tie(keyResGold, valsReducedGold) = getTest1_Gold();

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>(valsReducedGold == valsReduced));
}

TEST(ReduceByKey, WarpszReduceFullMultipleWarpsSinglevals) {
    af::array keys, vals;
    std::tie(keys,vals) = getTest6_KeyVals();

    af::array keyResGold, valsReducedGold;
    std::tie(keyResGold, valsReducedGold) = getTest6_Gold();

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>(valsReducedGold == valsReduced));
}

TEST(ReduceByKey, WarpszReduceBoundary) {
    af::array keys, vals;
    std::tie(keys,vals) = getTest2_KeyVals();

    af::array keyResGold, valsReducedGold;
    std::tie(keyResGold, valsReducedGold) = getTest2_Gold();

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>(valsReducedGold == valsReduced));
}

TEST(ReduceByKey, BlockSizeReduce) {
    af::array keys, vals;
    std::tie(keys,vals) = getTest3_KeyVals();

    af::array keyResGold, valsReducedGold;
    std::tie(keyResGold, valsReducedGold) = getTest3_Gold();

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>(valsReducedGold == valsReduced));

}

TEST(ReduceByKey, MultiBlockReduceSingleval) {
    af::array keys = af::constant(0, 1024 * 1024, s32);
    af::array vals = af::constant(1, 1024 * 1024, f32);

    af::array keyResGold = af::constant(0, 1);
    af::array valsReducedGold = af::constant(1024 * 1024, 1, f32);

    af::array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals);

    ASSERT_TRUE(af::allTrue<bool>(keyResGold == keyRes));
    ASSERT_TRUE(af::allTrue<bool>((valsReducedGold - valsReduced) < 0.0001));
}


void reduce_by_key_test(std::string test_fn) {
    vector<dim4> numDims;
    vector<vector<float> > data;
    vector<vector<float> > tests;
    readTests<float,float,float> (test_fn, numDims, data, tests);

    for(int t=0; t<numDims.size()/2; ++t) {
        dim4 kdim = numDims[t*2];
        dim4 vdim = numDims[t*2 + 1];

        vector<int> in_keys(data[t*2].begin(), data[t*2].end());
        vector<float> in_vals(data[t*2 + 1].begin(), data[t*2 + 1].end());

        af_array inKeys   = 0;
        af_array inVals   = 0;
        af_array outKeys  = 0;
        af_array outVals  = 0;
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inKeys, &in_keys.front(), kdim.ndims(), kdim.get(), (af_dtype) af::dtype_traits<int>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inVals, &in_vals.front(), vdim.ndims(), vdim.get(), (af_dtype) af::dtype_traits<float>::af_type));

        vector<int>   currGoldKeys(tests[t*2].begin(), tests[t*2].end());
        vector<float> currGoldVals(tests[t*2 + 1].begin(), tests[t*2 + 1].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_sum_by_key(&outKeys, &outVals, inKeys, inVals, 0));

        dim_t ok0, ok1, ok2, ok3;
        dim_t ov0, ov1, ov2, ov3;
        af_get_dims(&ok0, &ok1, &ok2, &ok3, outKeys);
        af_get_dims(&ov0, &ov1, &ov2, &ov3, outVals);

        // Get result
        vector<int>   outKeysVec(ok0 * ok1 * ok2 * ok3);
        vector<float> outValsVec(ov0 * ov1 * ov2 * ov3);

        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)&outKeysVec.front(), outKeys));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)&outValsVec.front(), outVals));

        size_t nElems = currGoldKeys.size();
        if(std::equal(currGoldKeys.begin(), currGoldKeys.end(), outKeysVec.begin()) == false)
        {
            for (size_t elIter = 0; elIter < nElems; ++elIter) {

                EXPECT_NEAR(currGoldKeys[elIter], outKeysVec[elIter], 1e-4) << "at: " << elIter
                                                                << endl;
                EXPECT_NEAR(currGoldVals[elIter], outValsVec[elIter], 1e-4) << "at: " << elIter
                                                                << endl;
            }
            for(int i = 0; i < (int)nElems; i++) {
                cout << currGoldKeys[i] << ":" << currGoldVals[i] << ", ";
            }

            for(int i = 0; i < (int)nElems; i++) {
                cout << outKeysVec[i] << ":" << outValsVec[i] << ", ";
            }
            FAIL();
        }

        ASSERT_EQ(AF_SUCCESS, af_release_array(outKeys));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outVals));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inKeys));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inVals));
    }
}
TEST(ReduceByKey, MultiBlockReduceContig10) {
    reduce_by_key_test(string(TEST_DIR"/reduce/test_contig10_by_key.test"));
}

TEST(ReduceByKey, MultiBlockReduceRandom10) {
    reduce_by_key_test(string(TEST_DIR"/reduce/test_random10_by_key.test"));
}

TEST(ReduceByKey, MultiBlockReduceContig500) {
    reduce_by_key_test(string(TEST_DIR"/reduce/test_contig500_by_key.test"));
}

TEST(ReduceByKey, MultiBlockReduceByKeyRandom500) {
    reduce_by_key_test(string(TEST_DIR"/reduce/test_random500_by_key.test"));
}

TEST(ReduceByKey, productReduceByKey)
{
    const static int testSz = 8;
    const int   testKeys[testSz] = { 0, 2, 2, 9, 5, 5, 5, 8 };
    const float testVals[testSz] = { 0, 7, 1, 6, 2, 5, 3, 4 };

    af::array keys(testSz, testKeys);
    af::array vals(testSz, testVals);

    af::array reduced_keys, reduced_vals;
    productByKey(reduced_keys, reduced_vals, keys, vals, 0, 1);

    const int goldSz = 5;
    const float gold_reduce[goldSz] = { 0, 7, 6, 30, 4 };
    float *h_a = reduced_vals.host<float>();

    for (int i = 0; i < goldSz; i++) {
        ASSERT_EQ(gold_reduce[i], h_a[i]);
    }

}

TEST(ReduceByKey, minReduceByKey)
{
    const static int testSz = 8;
    const int   testKeys[testSz] = { 0, 2, 2, 9, 5, 5, 5, 8 };
    const float testVals[testSz] = { 0, 7, 1, 6, 2, 5, 3, 4 };

    af::array keys(testSz, testKeys);
    af::array vals(testSz, testVals);

    af::array reduced_keys, reduced_vals;
    minByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const float gold_reduce[goldSz] = { 0, 1, 6, 2, 4 };
    float *h_a = reduced_vals.host<float>();

    for (int i = 0; i < goldSz; i++) {
        ASSERT_EQ(gold_reduce[i], h_a[i]);
    }

}

TEST(ReduceByKey, maxReduceByKey)
{
    const static int testSz = 8;
    const int   testKeys[testSz] = { 0, 2, 2, 9, 5, 5, 5, 8 };
    const float testVals[testSz] = { 0, 7, 1, 6, 2, 5, 3, 4 };

    af::array keys(testSz, testKeys);
    af::array vals(testSz, testVals);

    af::array reduced_keys, reduced_vals;
    maxByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const float gold_reduce[goldSz] = { 0, 7, 6, 5, 4 };
    float *h_a = reduced_vals.host<float>();

    for (int i = 0; i < goldSz; i++) {
        ASSERT_EQ(gold_reduce[i], h_a[i]);
    }

}

TEST(ReduceByKey, allTrueReduceByKey)
{
    const static int testSz = 8;
    const int   testKeys[testSz] = { 0, 2, 2, 9, 5, 5, 5, 8 };
    const float testVals[testSz] = { 0, 1, 1, 1, 0, 1, 1, 1 };

    af::array keys(testSz, testKeys);
    af::array vals(testSz, testVals);

    af::array reduced_keys, reduced_vals;
    allTrueByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const int gold_reduce[goldSz] = { 0, 1, 1, 0, 1 };
    char *h_a = reduced_vals.host<char>();

    for (int i = 0; i < goldSz; i++) {
        ASSERT_EQ(gold_reduce[i], h_a[i]);
    }

}

TEST(ReduceByKey, anyTrueReduceByKey)
{
    const static int testSz = 8;
    const int   testKeys[testSz] = { 0, 2, 2, 9, 5, 5, 8, 8 };
    const float testVals[testSz] = { 0, 1, 1, 1, 0, 1, 0, 0 };

    af::array keys(testSz, testKeys);
    af::array vals(testSz, testVals);

    af::array reduced_keys, reduced_vals;
    anyTrueByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const int gold_reduce[goldSz] = { 0, 1, 1, 1, 0 };
    char *h_a = reduced_vals.host<char>();


    for (int i = 0; i < goldSz; i++) {
        ASSERT_EQ(gold_reduce[i], h_a[i]);
    }

}

TEST(ReduceByKey, countReduceByKey)
{
    const static int testSz = 8;
    const int   testKeys[testSz] = { 0, 2, 2, 9, 5, 5, 5, 5 };
    const float testVals[testSz] = { 0, 1, 1, 1, 0, 1, 1, 1 };

    af::array keys(testSz, testKeys);
    af::array vals(testSz, testVals);

    af::array reduced_keys, reduced_vals;
    countByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 4;
    const int gold_reduce[goldSz] = { 0, 2, 1, 3 };
    unsigned *h_a = reduced_vals.host<unsigned>();

    for (int i = 0; i < goldSz; i++) {
        ASSERT_EQ(gold_reduce[i], h_a[i]);
    }

}

TEST(ReduceByKey, ReduceByKeyNans)
{
    const static int testSz = 8;
    const int   testKeys[testSz] = { 0, 2, 2, 9, 5, 5, 5, 8 };
    const float testVals[testSz] = { 0, 7, NAN, 6, 2, 5, 3, 4 };

    af::array keys(testSz, testKeys);
    af::array vals(testSz, testVals);

    af::array reduced_keys, reduced_vals;
    productByKey(reduced_keys, reduced_vals, keys, vals, 0, 1);

    const int goldSz = 5;
    const float gold_reduce[goldSz] = { 0, 7, 6, 30, 4 };
    float *h_a = reduced_vals.host<float>();

    for (int i = 0; i < goldSz; i++) {
        ASSERT_EQ(gold_reduce[i], h_a[i]);
    }
}

TEST(Reduce, Test_Product_Global)
{
    const int num = 100;
    array a = 1 + round(5 * randu(num, 1)) / 100;

    float res = product<float>(a);
    float *h_a = a.host<float>();
    float gold = 1;

    for (int i = 0; i < num; i++) {
        gold *= h_a[i];
    }

    ASSERT_NEAR(gold, res, 1e-3);
    freeHost(h_a);
}

TEST(Reduce, Test_Sum_Global)
{
    const int num = 10000;
    array a = round(2 * randu(num, 1));

    float res = sum<float>(a);
    float *h_a = a.host<float>();
    float gold = 0;

    for (int i = 0; i < num; i++) {
        gold += h_a[i];
    }

    ASSERT_EQ(gold, res);
    freeHost(h_a);
}

TEST(Reduce, Test_Count_Global)
{
    const int num = 10000;
    array a = round(2 * randu(num, 1));
    array b = a.as(b8);

    int res = count<int>(b);
    char *h_b = b.host<char>();
    int gold = 0;

    for (int i = 0; i < num; i++) {
        gold += h_b[i];
    }

    ASSERT_EQ(gold, res);
    freeHost(h_b);
}

TEST(Reduce, Test_min_Global)
{
    if (noDoubleTests<double>()) return;

    const int num = 10000;
    array a = randu(num, 1, f64);
    double res = min<double>(a);
    double *h_a = a.host<double>();
    double gold = std::numeric_limits<double>::max();

    if (noDoubleTests<double>()) return;

    for (int i = 0; i < num; i++) {
        gold = std::min(gold, h_a[i]);
    }

    ASSERT_EQ(gold, res);
    freeHost(h_a);
}

TEST(Reduce, Test_max_Global)
{
    const int num = 10000;
    array a = randu(num, 1);
    float res = max<float>(a);
    float *h_a = a.host<float>();
    float gold = -std::numeric_limits<float>::max();

    for (int i = 0; i < num; i++) {
        gold = std::max(gold, h_a[i]);
    }

    ASSERT_EQ(gold, res);
    freeHost(h_a);
}


template<typename T>
void typed_assert_eq(T lhs, T rhs, bool both = true)
{
    ASSERT_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<float>(float lhs, float rhs, bool both)
{
    ASSERT_FLOAT_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<double>(double lhs, double rhs, bool both)
{
    ASSERT_DOUBLE_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<cfloat>(cfloat lhs, cfloat rhs, bool both)
{
    ASSERT_FLOAT_EQ(real(lhs), real(rhs));
    if(both) {
        ASSERT_FLOAT_EQ(imag(lhs), imag(rhs));
    }
}

template<>
void typed_assert_eq<cdouble>(cdouble lhs, cdouble rhs, bool both)
{
    ASSERT_DOUBLE_EQ(real(lhs), real(rhs));
    if(both) {
        ASSERT_DOUBLE_EQ(imag(lhs), imag(rhs));
    }
}

TYPED_TEST(Reduce, Test_All_Global)
{
    if (noDoubleTests<TypeParam>()) return;

    // Input size test
    for(int i = 1; i < 1000; i+=100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam)true);
        array a(2, num/2, &h_vals.front());

        TypeParam res = allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)true, res, false);

        h_vals[3] = false;
        a = array(2, num/2, &h_vals.front());

        res = allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)false, res, false);
    }

    // false value location test
    const int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam)true);
    for(int i = 1; i < 10000; i+=100) {
        h_vals[i] = false;
        array a(2, num/2, &h_vals.front());

        TypeParam res = allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)false, res, false);

        h_vals[i] = true;
    }
}

TYPED_TEST(Reduce, Test_Any_Global)
{
    if (noDoubleTests<TypeParam>()) return;

    // Input size test
    for(int i = 1; i < 1000; i+=100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam)false);
        array a(2, num/2, &h_vals.front());

        TypeParam res = anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)false, res, false);

        h_vals[3] = true;
        a = array(2, num/2, &h_vals.front());

        res = anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)true, res, false);
    }

    // true value location test
    const int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam)false);
    for(int i = 1; i < 10000; i+=100) {
        h_vals[i] = true;
        array a(2, num/2, &h_vals.front());

        TypeParam res = anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)true, res, false);

        h_vals[i] = false;
    }
}

TEST(MinMax, MinMaxNaN)
{
    const int num = 10000;
    array A = randu(num);
    A(where(A < 0.25)) = NaN;

    float minval = min<float>(A);
    float maxval = max<float>(A);

    ASSERT_NE(std::isnan(minval), true);
    ASSERT_NE(std::isnan(maxval), true);

    float *h_A = A.host<float>();

    for (int i = 0; i < num; i++) {
        if (!std::isnan(h_A[i])) {
            ASSERT_LE(minval, h_A[i]);
            ASSERT_GE(maxval, h_A[i]);
        }
    }

    freeHost(h_A);
}

TEST(MinMax, MinCplxNaN)
{
    float real_wnan_data[] = {
        0.005f, NAN, -6.3f, NAN, -0.5f,
        NAN, NAN, 0.2f, -1205.4f, 8.9f
    };

    float imag_wnan_data[] = {
        NAN, NAN, -9.0f, -0.005f, -0.3f,
        0.007f, NAN, 0.1f, NAN, 4.5f
    };

    int rows = 5;
    int cols = 2;
    array real_wnan(rows, cols, real_wnan_data);
    array imag_wnan(rows, cols, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_min_real[] = { -0.5f, 0.2f };
    float gold_min_imag[] = { -0.3f, 0.1f };

    array min_val = af::min(a);

    vector< complex<float> > h_min_val(cols);
    min_val.host(&h_min_val[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_min_val[i].real(), gold_min_real[i]);
        ASSERT_FLOAT_EQ(h_min_val[i].imag(), gold_min_imag[i]);
    }
}

TEST(MinMax, MaxCplxNaN)
{
    // 4th element is unusually large to cover the case where
    //  one part holds the largest value among the array,
    //  and the other part is NaN.
    // There's a possibility where the NaN is turned into 0
    //  (since Binary<>::init() will initialize it to 0 in
    //  for complex max op) during the comparisons, and so its
    //  magnitude will determine that that element is the max,
    //  whereas it should have been ignored since its other
    //  part is NaN
    float real_wnan_data[] = {
        0.005f, NAN, -6.3f, NAN, -0.5f,
        NAN, NAN, 0.2f, -1205.4f, 8.9f
    };

    float imag_wnan_data[] = {
        NAN, NAN, -9.0f, -0.005f, -0.3f,
        0.007f, NAN, 0.1f, NAN, 4.5f
    };

    int rows = 5;
    int cols = 2;
    array real_wnan(rows, cols, real_wnan_data);
    array imag_wnan(rows, cols, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_max_real[] = { -6.3f, 8.9f };
    float gold_max_imag[] = { -9.0f, 4.5f };

    array max_val = af::max(a);

    vector< complex<float> > h_max_val(cols);
    max_val.host(&h_max_val[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_max_val[i].real(), gold_max_real[i]);
        ASSERT_FLOAT_EQ(h_max_val[i].imag(), gold_max_imag[i]);
    }
}

TEST(Count, NaN)
{
    const int num = 10000;
    array A = round(5 * randu(num));
    array B = A;

    A(where(A == 2)) = NaN;

    ASSERT_EQ(count<uint>(A), count<uint>(B));
}

TEST(Sum, NaN)
{
    const int num = 10000;
    array A = randu(num);
    A(where(A < 0.25)) = NaN;

    float res = sum<float>(A);

    ASSERT_EQ(std::isnan(res), true);

    res = sum<float>(A, 0);
    float *h_A = A.host<float>();

    float tmp = 0;
    for (int i = 0; i < num; i++) {
        tmp += std::isnan(h_A[i]) ? 0 : h_A[i];
    }

    ASSERT_NEAR(res/num, tmp/num, 1E-5);
    freeHost(h_A);
}

TEST(Product, NaN)
{
    const int num = 5;
    array A = randu(num);
    A(2) = NaN;

    float res = product<float>(A);

    ASSERT_EQ(std::isnan(res), true);

    res = product<float>(A, 1);
    float *h_A = A.host<float>();

    float tmp = 1;
    for (int i = 0; i < num; i++) {
        tmp *= std::isnan(h_A[i]) ? 1 : h_A[i];
    }

    ASSERT_NEAR(res/num, tmp/num, 1E-5);
    freeHost(h_A);
}

TEST(AnyAll, NaN)
{
    const int num = 10000;
    array A = (randu(num) > 0.5).as(f32);
    array B = A;

    B(where(B == 0)) = NaN;

    ASSERT_EQ(anyTrue<bool>(B), true);
    ASSERT_EQ(allTrue<bool>(B), true);
    ASSERT_EQ(anyTrue<bool>(A), true);
    ASSERT_EQ(allTrue<bool>(A), false);
}

TEST(MaxAll, IndexedSmall)
{
    const int num = 1000;
    const int st = 10;
    const int en = num - 100;
    array a = randu(num);
    float b = max<float>(a(seq(st, en)));

    vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) {
        res = std::max(res, ha[i]);
    }

    ASSERT_EQ(b, res);
}

TEST(MaxAll, IndexedBig)
{
    const int num = 100000;
    const int st = 1000;
    const int en = num - 1000;
    array a = randu(num);
    float b = max<float>(a(seq(st, en)));

    vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) {
        res = std::max(res, ha[i]);
    }

    ASSERT_EQ(b, res);
}

TEST(Reduce, KernelName)
{
    const int m = 64;
    const int n = 100;
    const int b = 5;

    array in = constant(0, m, n, b);
    for (int i = 0; i < b; i++) {
        array tmp = randu(m, n);
        in(span, span, i) = tmp;
        ASSERT_EQ(min<float>(in(span, span, i)),
                  min<float>(tmp));
    }
}

TEST(Reduce, AllSmallIndexed)
{
    const int len = 1000;
    array a = af::range(dim4(len, 2));
    array b = a(seq(len/2), span);
    ASSERT_EQ(max<float>(b), len/2-1);
}
