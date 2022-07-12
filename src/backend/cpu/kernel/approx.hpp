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
#include <math.hpp>
#include "interp.hpp"
#include <common/debug.hpp>

namespace cpu {
namespace kernel {

template<typename InT, typename LocT, int order>
void approx1(Param<InT> yo, CParam<InT> yi, CParam<LocT> xo, const int xdim,
             const LocT &xi_beg, const LocT &xi_step, const float offGrid,
             af_interp_type method) {
    InT *yo_ptr        = yo.get();
    const LocT *xo_ptr = xo.get();

    const af::dim4 yo_dims = yo.dims();
    const af::dim4 yi_dims = yi.dims();
    const af::dim4 xo_dims = xo.dims();

    const af::dim4 yo_strides = yo.strides();
    const af::dim4 yi_strides = yi.strides();
    const af::dim4 xo_strides = xo.strides();

    Interp1<InT, LocT, order> interp;
    bool is_xo_off[] = {xo_dims[0] > 1, xo_dims[1] > 1, xo_dims[2] > 1,
                        xo_dims[3] > 1};
    bool is_yi_off[] = {true, true, true, true};
    is_yi_off[xdim]  = false;

    for (dim_t idw = 0; idw < yo_dims[3]; idw++) {
        for (dim_t idz = 0; idz < yo_dims[2]; idz++) {
            dim_t yo_off_zw = idw * yo_strides[3] + idz * yo_strides[2];
            dim_t yi_off_zw = idw * yi_strides[3] * is_yi_off[3] +
                              idz * yi_strides[2] * is_yi_off[2];
            dim_t xo_off_zw = idw * xo_strides[3] * is_xo_off[3] +
                              idz * xo_strides[2] * is_xo_off[2];

            for (dim_t idy = 0; idy < yo_dims[1]; idy++) {
                dim_t yo_off = yo_off_zw + idy * yo_strides[1];
                dim_t yi_off = yi_off_zw + idy * yi_strides[1] * is_yi_off[1];
                dim_t xo_off = xo_off_zw + idy * xo_strides[1] * is_xo_off[1];

                //TODO: if yi_dims[xdim] == 2, lerp
                //TODO: if yi_dims[xdim] == 3, quadratic fit


                for (dim_t idx = 0; idx < yo_dims[0]; idx++) {
                    dim_t yi_idx = idx * is_yi_off[0];
                    const LocT x =
                        (xo_ptr[xo_off + idx * is_xo_off[0]] - xi_beg) /
                        xi_step;

                    // FIXME: Only cubic interpolation is doing clamping
                    // We need to make it consistent across all methods
                    // Not changing the behavior because tests will fail
                    bool clamp = order == 3;

                    if (x < 0 || yi_dims[xdim] < x + 1) {
                        yo_ptr[yo_off + idx] = scalar<InT>(offGrid);
                    } else {
                        if(method == AF_INTERP_CUBIC_SPLINE) {
                        } else {
                            interp(yo, yo_off + idx, yi, yi_off + yi_idx, x, method,
                                   1, clamp, xdim);
                        }
                    }
                }
            }
        }
    }
}

template<typename InT, typename LocT>
void approx1(Param<InT> yo, CParam<InT> yi, CParam<LocT> xo, CParam<InT> ders, const int xdim,
             const LocT &xi_beg, const LocT &xi_step, const float offGrid,
             af_interp_type method) {
    InT *yo_ptr        = yo.get();
    const LocT *xo_ptr = xo.get();

    const af::dim4 yo_dims = yo.dims();
    const af::dim4 yi_dims = yi.dims();
    const af::dim4 xo_dims = xo.dims();

    const af::dim4 yo_strides = yo.strides();
    const af::dim4 yi_strides = yi.strides();
    const af::dim4 xo_strides = xo.strides();

    bool is_xo_off[] = {xo_dims[0] > 1, xo_dims[1] > 1, xo_dims[2] > 1,
                        xo_dims[3] > 1};
    bool is_yi_off[] = {true, true, true, true};
    is_yi_off[xdim]  = false;

    for (dim_t idw = 0; idw < yo_dims[3]; idw++) {
        for (dim_t idz = 0; idz < yo_dims[2]; idz++) {
            dim_t yo_off_zw = idw * yo_strides[3] + idz * yo_strides[2];
            dim_t yi_off_zw = idw * yi_strides[3] * is_yi_off[3] +
                              idz * yi_strides[2] * is_yi_off[2];
            dim_t xo_off_zw = idw * xo_strides[3] * is_xo_off[3] +
                              idz * xo_strides[2] * is_xo_off[2];

            for (dim_t idy = 0; idy < yo_dims[1]; idy++) {
                dim_t yo_off = yo_off_zw + idy * yo_strides[1];
                dim_t yi_off = yi_off_zw + idy * yi_strides[1] * is_yi_off[1];
                dim_t xo_off = xo_off_zw + idy * xo_strides[1] * is_xo_off[1];

                //TODO: if yi_dims[xdim] == 2, lerp
                //TODO: if yi_dims[xdim] == 3, quadratic fit


                for (dim_t idx = 0; idx < yo_dims[0]; idx++) {
                    dim_t yi_idx = idx * is_yi_off[0];
                    const LocT x =
                        (xo_ptr[xo_off + idx * is_xo_off[0]] - xi_beg) /
                        xi_step;

                    // FIXME: Only cubic interpolation is doing clamping
                    // We need to make it consistent across all methods
                    // Not changing the behavior because tests will fail

                    if (x < 0 || yi_dims[xdim] < x + 1) {
                        yo_ptr[yo_off + idx] = scalar<InT>(offGrid);
                    } else {
                        Interp1Spline<InT, LocT> interps;
                        interps(yo, yo_off + idx, yi, yi_off + yi_idx, x, ders,
                                1, true, xdim);
                    }
                }
            }
        }
    }
}

template<typename InT, typename LocT, int order>
void approx2(Param<InT> zo, CParam<InT> zi, CParam<LocT> xo, const int xdim,
             const LocT &xi_beg, const LocT &xi_step, CParam<LocT> yo,
             const int ydim, const LocT &yi_beg, const LocT &yi_step,
             float const offGrid, af_interp_type method) {
    InT *zo_ptr        = zo.get();
    const LocT *xo_ptr = xo.get();
    const LocT *yo_ptr = yo.get();

    af::dim4 const zo_dims    = zo.dims();
    af::dim4 const zi_dims    = zi.dims();
    af::dim4 const xo_dims    = xo.dims();
    af::dim4 const zo_strides = zo.strides();
    af::dim4 const zi_strides = zi.strides();
    af::dim4 const xo_strides = xo.strides();
    af::dim4 const yo_strides = yo.strides();

    Interp2<InT, LocT, order> interp;
    bool is_xo_off[] = {xo_dims[0] > 1, xo_dims[1] > 1, xo_dims[2] > 1,
                        xo_dims[3] > 1};
    bool is_zi_off[] = {true, true, true, true};
    is_zi_off[xdim]  = false;
    is_zi_off[ydim]  = false;

    for (dim_t idw = 0; idw < zo_dims[3]; idw++) {
        for (dim_t idz = 0; idz < zo_dims[2]; idz++) {
            dim_t zo_off_zw = idw * zo_strides[3] + idz * zo_strides[2];
            dim_t zi_off_zw = idw * zi_strides[3] * is_zi_off[3] +
                              idz * zi_strides[2] * is_zi_off[2];
            dim_t xo_off_zw = idw * xo_strides[3] * is_xo_off[3] +
                              idz * xo_strides[2] * is_xo_off[2];
            dim_t yo_off_zw = idw * yo_strides[3] * is_xo_off[3] +
                              idz * yo_strides[2] * is_xo_off[2];

            for (dim_t idy = 0; idy < zo_dims[1]; idy++) {
                dim_t xo_off = xo_off_zw + idy * xo_strides[1] * is_xo_off[1];
                dim_t yo_off = yo_off_zw + idy * yo_strides[1] * is_xo_off[1];
                dim_t zi_off = zi_off_zw + idy * zi_strides[1] * is_zi_off[1];
                dim_t zo_off = zo_off_zw + idy * zo_strides[1];

                for (dim_t idx = 0; idx < zo_dims[0]; idx++) {
                    const LocT x = (xo_ptr[xo_off + idx] - xi_beg) / xi_step;
                    const LocT y = (yo_ptr[yo_off + idx] - yi_beg) / yi_step;

                    dim_t zi_idx = idx * zi_strides[0] * is_zi_off[0];

                    // FIXME: Only cubic interpolation is doing clamping
                    // We need to make it consistent across all methods
                    // Not changing the behavior because tests will fail
                    bool clamp = order == 3;

                    if (x < 0 || zi_dims[xdim] < x + 1 || y < 0 ||
                        zi_dims[ydim] < y + 1) {
                        zo_ptr[zo_off + idx] = scalar<InT>(offGrid);
                    } else {
                        interp(zo, zo_off + idx, zi, zi_off + zi_idx, x, y,
                               method, 1, clamp, xdim, ydim);
                    }
                }
            }
        }
    }
}

// sets up RHS of natural spline linear system
template<typename InT>
void calcDiffs(Param<InT> diffs, CParam<InT> yi, const int xdim) {
    const InT *yiptr   = yi.get();
    InT *diffptr = diffs.get();

    bool is_yi_off[] = {true, true, true, true};
    //is_yi_off[xdim]  = false;

    af::dim4 const yi_dims    = yi.dims();
    af::dim4 const yi_strides = yi.strides();

    for (dim_t idw = 0; idw < yi_dims[3]; idw++) {
        for (dim_t idz = 0; idz < yi_dims[2]; idz++) {
            dim_t yi_off_zw = idw * yi_strides[3] * is_yi_off[3] +
                              idz * yi_strides[2] * is_yi_off[2];

            for (dim_t idy = 0; idy < yi_dims[1]; idy++) {
                dim_t yi_off = yi_off_zw + idy * yi_strides[1] * is_yi_off[1];

                for (dim_t idx = 0; idx < yi_dims[0]; idx++) {
                    dim_t yi_idx = idx * is_yi_off[0];
                    //printf("%f \n", yiptr[yi_off + idx]);
                }
                printf("----\n");

                for (dim_t idx = 0; idx < yi_dims[0]; idx++) {
                    if(idx == 0) {
                        dim_t yi0_idx = (idx-1) * is_yi_off[0];
                        dim_t yi1_idx = (idx+1) * is_yi_off[0];
                        InT y0 = yiptr[yi_off + yi0_idx];
                        InT y1 = yiptr[yi_off + yi1_idx];
                        diffptr[idx] =  scalar<InT>(3) * (y1 - y0);
                        printf("%f \n", diffptr[idx]);
                    //TODO: fix condition for yi_dims[xdim]?
                    } else if (idx == (yi_dims[0] - 1)) {
                        dim_t yi0_idx = (idx-1) * is_yi_off[0];
                        dim_t yi1_idx = (idx)   * is_yi_off[0];
                        InT ym_1 = yiptr[yi_off + yi0_idx];
                        InT ym   = yiptr[yi_off + yi1_idx];
                        diffptr[idx] = scalar<InT>(3) * (ym - ym_1);
                        printf("%f \n", diffptr[idx]);
                    } else {
                        dim_t yi0_idx = (idx-1) * is_yi_off[0];
                        dim_t yi1_idx = (idx+1) * is_yi_off[0];
                        InT y0 = yiptr[yi_off + yi0_idx];
                        InT y1 = yiptr[yi_off + yi1_idx];
                        diffptr[idx] = scalar<InT>(3) *  (y1 - y0);
                        printf("%f \n", diffptr[idx]);
                    }
                }
            }
        }
    }
}

// sets up LHS of natural spline linear system
template<typename InT>
void fillCollocationMatrix(Param<InT> M) {
    af::dim4 const mdims    = M.dims();
    af::dim4 const mstrides = M.strides();
    InT *mptr               = M.get();

    // printf("%d %d %d %d\n", mstrides[0], mstrides[1], mstrides[2], mstrides[3]);

    for(size_t r=0; r<mdims[0]; ++r) {
        size_t r_offset = r * mstrides[1];

        if(r == 0) {
            mptr[r_offset + (r)]   = scalar<InT>(2);
            mptr[r_offset + (r+1)] = scalar<InT>(1);
        } else if (r == (mdims[0] - 1)) {
            mptr[r_offset + (r-1)] = scalar<InT>(1);
            mptr[r_offset + (r)]   = scalar<InT>(2);
        } else {
            mptr[r_offset + (r-1)] = scalar<InT>(1);
            mptr[r_offset + (r)]   = scalar<InT>(4);
            mptr[r_offset + (r+1)] = scalar<InT>(1);
        }

        for(size_t c=0; c<mdims[0]; ++c) {
            printf("%f ", mptr[r_offset + c]);
        }
        printf("\n");
        /*
        */
    }
}

template<typename InT>
void pd(Param<InT> der) {
    printf("pd\n");
    af::dim4 const ddims    = der.dims();
    af::dim4 const dstrides = der.strides();
    InT *dptr               = der.get();

    printf("%d %d %d %d\n", dstrides[0], dstrides[1], dstrides[2], dstrides[3]);

    for(size_t r=0; r<ddims[0]; ++r) {
        printf("%f\n", dptr[r]);
    }


}

}  // namespace kernel
}  // namespace cpu
