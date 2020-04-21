/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <af/dim4.hpp>
#include <common/jit/ReduceNode.hpp>
#include <scalar.hpp>

#undef _GLIBCXX_USE_INT128
#include <complex>
#include <err_cuda.hpp>
#include <kernel/reduce.hpp>
#include <memory>
#include <reduce.hpp>
#include <string>

using af::dim4;
using std::swap;
namespace cuda {
template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int axis, bool change_nan,
                 double nanval) {
    dim4 odims    = in.dims();
    odims[axis]    = 1;
    Array<To> out = createEmptyArray<To>(odims);
    //if(axis == 0) {
        std::string name_str("Rd");
        name_str += shortname<To>(true);

        auto node = std::make_shared<common::ReduceNode>(
             getFullName<To>(), name_str.c_str(), in.getNode(), axis, op, in.dims());

        out = createNodeArray<To>(odims, common::Node_ptr(node));
        out.eval();
    ////} else {
        //kernel::reduce<Ti, To, op>(out, in, axis, change_nan, nanval);
    //}
    return out;
}

template<af_op_t op, typename Ti, typename To>
To reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    return kernel::reduce_all<Ti, To, op>(in, change_nan, nanval);
}
}  // namespace cuda

#define INSTANTIATE(Op, Ti, To)                                               \
    template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int axis,\
                                          bool change_nan, double nanval);    \
    template To reduce_all<Op, Ti, To>(const Array<Ti> &in, bool change_nan,  \
                                       double nanval);
