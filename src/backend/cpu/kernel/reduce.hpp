/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>

namespace cpu
{
namespace kernel
{

template<af_op_t op, typename Ti, typename To, int D>
struct reduce_dim
{
    void operator()(Param<To> out, const dim_t outOffset,
                    CParam<Ti> in, const dim_t inOffset,
                    const int dim, bool change_nan, double nanval)
    {
        static const int D1 = D - 1;
        reduce_dim<op, Ti, To, D1> reduce_dim_next;

        const af::dim4 ostrides = out.strides();
        const af::dim4 istrides = in.strides();
        const af::dim4 odims    = out.dims();

        for (dim_t i = 0; i < odims[D1]; i++) {
            reduce_dim_next(out, outOffset + i * ostrides[D1],
                            in, inOffset + i * istrides[D1],
                            dim, change_nan, nanval);
        }
    }
};

template<af_op_t op, typename Ti, typename To>
struct reduce_dim<op, Ti, To, 0>
{

    Transform<Ti, To, op> transform;
    Binary<To, op> reduce;
    void operator()(Param<To> out, const dim_t outOffset,
                    CParam<Ti> in, const dim_t inOffset,
                    const int dim, bool change_nan, double nanval)
    {
        const af::dim4 istrides = in.strides();
        const af::dim4 idims    = in.dims();

        To * const outPtr = out.get() + outOffset;
        Ti const * const inPtr = in.get() + inOffset;
        dim_t stride = istrides[dim];

        To out_val = Binary<To, op>::init();
        for (dim_t i = 0; i < idims[dim]; i++) {
            To in_val = transform(inPtr[i * stride]);
            if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
            out_val = reduce(in_val, out_val);
        }

        *outPtr = out_val;
    }
};

template<af_op_t op, typename Ti, typename Tk, typename To, int D>
struct reduce_dim_by_key
{
    void operator()(Param<Tk> okeys, const dim_t okeysOffset,
                    Param<To> ovals, const dim_t ovalsOffset,
                    CParam<Tk> keys, const dim_t keysOffset,
                    CParam<Ti> vals, const dim_t valsOffset,
                    int *n_reduced, const int dim,  bool change_nan, double nanval)
    {
        AF_ERROR("Only 1-dimensional reduce_by_key is currently supported.", AF_ERR_NOT_SUPPORTED);
    }
};

template<af_op_t op, typename Ti, typename Tk, typename To>
struct reduce_dim_by_key<op, Ti, Tk, To, 0>
{

    Transform<Ti, To, op> transform;
    Binary<To, op> reduce;
    void operator()(Param<Tk> okeys, const dim_t okeysOffset,
                    Param<To> ovals, const dim_t ovalsOffset,
                    CParam<Tk> keys, const dim_t keysOffset,
                    CParam<Ti> vals, const dim_t valsOffset,
                    int *n_reduced, const int dim,  bool change_nan, double nanval)
    {
        // keys.dims() should== vals.dims()
        const af::dim4 istrides = keys.strides();
        const af::dim4 idims    = keys.dims();

        Tk const * const inKeysPtr = keys.get()  + keysOffset;
        Ti const * const inValsPtr = vals.get()  + valsOffset;
        Tk * const outKeysPtr      = okeys.get() + okeysOffset;
        To * const outValsPtr      = ovals.get() + ovalsOffset;


        //TODO: valid assumption? whatif empty? handle outside
        int nkeys = 0;
        Tk current_key = inKeysPtr[0];
        To out_val = reduce.init();

        for (dim_t i = 0; i < idims[0]; i++) {
            Tk keyval  = inKeysPtr[i];

            if(keyval == current_key) {
                To in_val = transform(inValsPtr[i]);
                if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
                out_val = reduce(in_val, out_val);

            } else {
                outKeysPtr[nkeys] = current_key;
                outValsPtr[nkeys] = out_val;

                current_key = keyval;
                out_val = transform(inValsPtr[i]);
                ++nkeys;
            }

            if(i == (idims[0] - 1)) {
                outKeysPtr[nkeys] = current_key;
                outValsPtr[nkeys] = out_val;
            }
        }

        *n_reduced = nkeys;
    }
};

}
}
