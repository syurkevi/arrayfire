/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>

#include <testHelpers.hpp>
#include <af/algorithm.h>
#include <af/arith.h>
#include <af/blas.h>
#include <af/defines.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/lapack.h>
#include <af/sparse.h>
#include <af/traits.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::deviceGC;
using af::dim4;
using af::dtype_traits;
using af::setDevice;
using af::sum;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;


template<typename T>
void sparseSolveTester(const int m, const int n, const int k, af_storage sstype, double eps,
                       int targetDevice = -1) {
    if (targetDevice >= 0) setDevice(targetDevice);

    deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

#if 1
    array A  = cpu_randu<T>(dim4(m, n));
    array X0 = cpu_randu<T>(dim4(n, k));
#else
    array A  = randu(m, n, (dtype)dtype_traits<T>::af_type);
    array X0 = randu(n, k, (dtype)dtype_traits<T>::af_type);
#endif
    A(A < T(0.2)) = T(0);
    array B0 = matmul(A, X0);

    array A_sparse = af::sparse(A, sstype);
    //af_print(A);
    //af_print(A_sparse);

    array Xsol = solve(A, B0);
    //af_print(Xsol);

    cout << A.dims() << " x ";
    cout << Xsol.dims() << "= ? " << endl;
    cout << B0.dims() <<endl;

    array X1 = solve(A_sparse, B0);
    //af_print(X1);

    array B1 = matmul(A, X1);

    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(abs(real(B0 - B1))) / (m * k),
        eps);
    ASSERT_NEAR(
        0,
        sum<typename dtype_traits<T>::base_type>(abs(imag(B0 - B1))) / (m * k),
        eps);
}

template<typename T>
void sparseSolveBatchTester(const int m, const int n, const int k, const int b, af_storage sstype,
                            double eps, int targetDevice = -1) {
    if (targetDevice >= 0) setDevice(targetDevice);

    deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

#if 1
    array A  = cpu_randu<T>(dim4(m, n, b));
    array X0 = cpu_randu<T>(dim4(n, k, b));
#else
    array A  = randu(m, n, b, (dtype)dtype_traits<T>::af_type);
    array X0 = randu(n, k, b, (dtype)dtype_traits<T>::af_type);
#endif

    A(A < T(0.2)) = T(0);
    array B0 = matmul(A, X0);

    try {
        af::sparse(A, sstype);
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
}

template<typename T>
void sparseSolveOverUnderDeterminedTester(const int m, const int n, const int k, af_storage sstype, double eps,
                                          int targetDevice = -1) {
    if (targetDevice >= 0) setDevice(targetDevice);

    deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

#if 1
    array A  = cpu_randu<T>(dim4(m, n));
    array X0 = cpu_randu<T>(dim4(n, k));
#else
    array A  = randu(m, n, (dtype)dtype_traits<T>::af_type);
    array X0 = randu(n, k, (dtype)dtype_traits<T>::af_type);
#endif
    A(A < T(0.2)) = T(0);
    array B0 = matmul(A, X0);

    array A_sparse = af::sparse(A, sstype);

    try {
        array X1 = solve(A_sparse, B0);
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_RUNTIME); }
}

template<typename T>
class SparseSolve : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_CASE(SparseSolve, TestTypes);

template<typename T>
double eps();

template<>
double eps<float>() {
    return 0.01f;
}

template<>
double eps<double>() {
    return 1e-5;
}

template<>
double eps<cfloat>() {
    return 0.015f;
}

template<>
double eps<cdouble>() {
    return 1e-5;
}

TYPED_TEST(SparseSolve, Square) {
    sparseSolveTester<TypeParam>(5, 5, 1, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, SquareMultipleOfTwo) {
    sparseSolveTester<TypeParam>(96, 96, 16, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, SquareLarge) {
    sparseSolveTester<TypeParam>(1000, 1000, 10, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, SquareMultipleOfTwoLarge) {
    sparseSolveTester<TypeParam>(2048, 2048, 32, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, SquareBatch) {
    sparseSolveBatchTester<TypeParam>(100, 100, 10, 10, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, SquareMultipleOfTwoBatch) {
    sparseSolveBatchTester<TypeParam>(96, 96, 16, 10, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, SquareLargeBatch) {
    sparseSolveBatchTester<TypeParam>(1000, 1000, 10, 10, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, SquareMultipleOfTwoLargeBatch) {
    sparseSolveBatchTester<TypeParam>(2048, 2048, 32, 10, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresUnderDetermined) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(80, 100, 20, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresUnderDeterminedMultipleOfTwo) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(96, 128, 40, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresUnderDeterminedLarge) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(800, 1000, 200, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresUnderDeterminedMultipleOfTwoLarge) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(1536, 2048, 400, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresOverDetermined) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(80, 60, 20, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresOverDeterminedMultipleOfTwo) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(96, 64, 1, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresOverDeterminedLarge) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(800, 600, 64, AF_STORAGE_CSR, eps<TypeParam>());
}

TYPED_TEST(SparseSolve, LeastSquaresOverDeterminedMultipleOfTwoLarge) {
    sparseSolveOverUnderDeterminedTester<TypeParam>(1536, 1024, 1, AF_STORAGE_CSR, eps<TypeParam>());
}

//currently unsupported
TEST(SparseSolve, SquareCOO) {
    deviceGC();
    if (noLAPACKTests()) return;

    array A  = cpu_randu<float>(dim4(10, 10));
    A(A < 0.2f) = 0.f;

    array X0 = cpu_randu<float>(dim4(10));
    array B0 = matmul(A, X0);

    array A_sparse = af::sparse(A, AF_STORAGE_COO);
    try {
        array X1 = solve(A_sparse, B0);
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_ARG); }
}

//currently unsupported
TEST(SparseSolve, SquareCSC) {
    deviceGC();

    if (noLAPACKTests()) return;

    array A     = cpu_randu<float>(dim4(10, 10));
    A(A < 0.2f) = 0.f;

    array X0 = cpu_randu<float>(dim4(10));
    array B0 = matmul(A, X0);

    try {
        //TODO: move outside once conversion is valid
        array A_sparse = af::sparse(A, AF_STORAGE_CSC);
        array X1 = solve(A_sparse, B0);
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_ARG); }
}
