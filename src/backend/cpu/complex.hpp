/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cpu.hpp>
#include <jit/BinaryNode.hpp>
#include <jit/UnaryNode.hpp>
#include <optypes.hpp>
#include <af/dim4.hpp>
#include <complex>

namespace cpu {

template <typename To, typename Ti>
struct BinOp<To, Ti, af_cplx2_t> {
    void eval(jit::array<To> &out, const jit::array<Ti> &lhs,
              const jit::array<Ti> &rhs, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(lhs[i], rhs[i]); }
    }
};

template <typename To, typename Ti>
Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs,
               const af::dim4 &odims) {
    jit::Node_ptr lhs_node = lhs.getNode();
    jit::Node_ptr rhs_node = rhs.getNode();

    jit::BinaryNode<To, Ti, af_cplx2_t> *node =
        new jit::BinaryNode<To, Ti, af_cplx2_t>(lhs_node, rhs_node);

    return createNodeArray<To>(odims, jit::Node_ptr(node));
}

#define CPLX_UNARY_FN(op)                                                   \
    template <typename To, typename Ti>                                     \
    struct UnOp<To, Ti, af_##op##_t> {                                      \
        void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) { \
            for (int i = 0; i < lim; i++) { out[i] = std::op(in[i]); }      \
        }                                                                   \
    };

CPLX_UNARY_FN(real)
CPLX_UNARY_FN(imag)
CPLX_UNARY_FN(conj)
CPLX_UNARY_FN(abs)

template <typename To, typename Ti>
Array<To> real(const Array<Ti> &in) {
    jit::Node_ptr in_node = in.getNode();
    jit::UnaryNode<To, Ti, af_real_t> *node =
        new jit::UnaryNode<To, Ti, af_real_t>(in_node);

    return createNodeArray<To>(in.dims(),
                               jit::Node_ptr(static_cast<jit::Node *>(node)));
}

template <typename To, typename Ti>
Array<To> imag(const Array<Ti> &in) {
    jit::Node_ptr in_node = in.getNode();
    jit::UnaryNode<To, Ti, af_imag_t> *node =
        new jit::UnaryNode<To, Ti, af_imag_t>(in_node);

    return createNodeArray<To>(in.dims(),
                               jit::Node_ptr(static_cast<jit::Node *>(node)));
}

template <typename To, typename Ti>
Array<To> abs(const Array<Ti> &in) {
    jit::Node_ptr in_node = in.getNode();
    jit::UnaryNode<To, Ti, af_abs_t> *node =
        new jit::UnaryNode<To, Ti, af_abs_t>(in_node);

    return createNodeArray<To>(in.dims(),
                               jit::Node_ptr(static_cast<jit::Node *>(node)));
}

template <typename T>
Array<T> conj(const Array<T> &in) {
    jit::Node_ptr in_node = in.getNode();
    jit::UnaryNode<T, T, af_conj_t> *node =
        new jit::UnaryNode<T, T, af_conj_t>(in_node);

    return createNodeArray<T>(in.dims(),
                              jit::Node_ptr(static_cast<jit::Node *>(node)));
}
}  // namespace cpu
