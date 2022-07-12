/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <approx.hpp>
#include <kernel/approx.hpp>
#include <platform.hpp>
#include <solve.hpp>
#include <blas.hpp>
#include <af/dim4.hpp>
#include <common/debug.hpp>

namespace cpu {

template<typename Ty, typename Tp>
void approx1(Array<Ty> &yo, const Array<Ty> &yi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const af_interp_type method, const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 1>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_LINEAR_COSINE:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 2>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_CUBIC_SPLINE:
            {
                unsigned minval = 4;
                Ty lder;
                Ty rder;

                size_t npts = yi.dims()[xdim];
                //SHOW(yi.dims(), xdim, npts);

                //TODO: create diff array
                Array<Ty> diffs = createEmptyArray<Ty>(af::dim4(npts));
                printf("xdim%d\n",xdim);
                getQueue().enqueue(kernel::calcDiffs<Ty>, diffs, yi, xdim);

                //TODO: create collocation matrix
                Array<Ty> collocM = createValueArray<Ty>(af::dim4(npts, npts), (Ty)0);
                getQueue().enqueue(kernel::fillCollocationMatrix<Ty>, collocM);

                //TODO: transform to upper triangular

                //TODO: transform rhs

                //TODO: solve Ax = b
                Array<Ty> ders = solve(collocM, diffs);
                // TODO: Array<T> solve(collocM, diffs, AF_MAT_UPPER);
                //getQueue().enqueue(kernel::pd<Ty>, ders);

                //TODO: use derivatives to figure out coefficients

                //TODO: pass to cubicInterpFunc to perform actual interp
                //

                //TODO: test:
                //      solve vs upper triangle for larger n
                //      solve vs sparse solve " "


                /*
                Array<Ty> ver = matmul(collocM,  ders, AF_MAT_NONE, AF_MAT_NONE);
                getQueue().enqueue(kernel::pd<Ty>, ver);
                */


                getQueue().enqueue(kernel::approx1<Ty, Tp>, yo, yi, xo, ders, xdim,
                                   xi_beg, xi_step, offGrid, method);
                break;
            }
        default: break;
    }
}

template<typename Ty, typename Tp>
void approx2(Array<Ty> &zo, const Array<Ty> &zi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Array<Tp> &yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const af_interp_type method,
             const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 1>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 2>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 3>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        default: break;
    }
}

#define INSTANTIATE(Ty, Tp)                                       \
    template void approx1<Ty, Tp>(                                \
        Array<Ty> & yo, const Array<Ty> &yi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const af_interp_type method, const float offGrid);        \
    template void approx2<Ty, Tp>(                                \
        Array<Ty> & zo, const Array<Ty> &zi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const Array<Tp> &yo, const int ydim, const Tp &yi_beg,    \
        const Tp &yi_step, const af_interp_type method, const float offGrid);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace cpu
