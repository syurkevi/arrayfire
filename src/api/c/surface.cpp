/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/graphics.h>
#include <af/image.h>
#include <af/array.h>
#include <af/seq.h>
#include <af/util.h>
#include <af/data.h>

#include <ArrayInfo.hpp>
#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <surface.hpp>
#include <reduce.hpp>
#include <join.hpp>
#include <transpose.hpp>
#include <tile.hpp>
#include <range.hpp>
#include <reorder.hpp>
#include <handle.hpp>
#include <iostream>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
fg::Surface* setup_surface(const af_array xVals, const af_array yVals, const af_array zVals)
{
    Array<T> xIn = getArray<T>(xVals);
    Array<T> yIn = getArray<T>(yVals);
    Array<T> zIn = getArray<T>(zVals);

    ArrayInfo Xinfo = getInfo(xVals);
    ArrayInfo Yinfo = getInfo(yVals);
    ArrayInfo Zinfo = getInfo(zVals);

    af::dim4 X_dims = Xinfo.dims();
    af::dim4 Y_dims = Yinfo.dims();
    af::dim4 Z_dims = Zinfo.dims();

    DIM_ASSERT(2, Z_dims[0] > 1 );
    DIM_ASSERT(2, Z_dims[1] > 1 );
    DIM_ASSERT(1, Y_dims[0] > 1 );
    DIM_ASSERT(1, Y_dims[1] > 1 );
    DIM_ASSERT(0, X_dims[0] > 1 );
    DIM_ASSERT(0, X_dims[1] > 1 );

    transpose_inplace(xIn,false);
    transpose_inplace(yIn,false);
    xIn.modDims(xIn.elements());
    yIn.modDims(yIn.elements());

    zIn.modDims(zIn.elements());

    T xmax = reduce_all<af_max_t, T, T>(xIn);
    T xmin = reduce_all<af_min_t, T, T>(xIn);
    T ymax = reduce_all<af_max_t, T, T>(yIn);
    T ymin = reduce_all<af_min_t, T, T>(yIn);
    T zmax = reduce_all<af_max_t, T, T>(zIn);
    T zmin = reduce_all<af_min_t, T, T>(zIn);

    ForgeManager& fgMngr = ForgeManager::getInstance();
    fg::Surface* surface = fgMngr.getSurface(Z_dims[0], Z_dims[1], getGLType<T>());
    surface->setColor(1.0, 0.0, 0.0);
    surface->setAxesLimits(xmax, xmin, ymax, ymin, zmax, zmin);
    surface->setAxesTitles("X Axis", "Y Axis", "Z Axis");

    Array<T> Z = join(1, join(1, xIn, yIn), zIn);
    Z = transpose(Z,false);
    Z.modDims(Z.elements());

    copy_surface<T>(Z, surface);

    zIn.modDims(Z_dims);
    return surface;
}
#endif

af_err af_draw_surface_s(const af_window wind, const af_array S, const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        ArrayInfo Sinfo = getInfo(S);
        af::dim4 S_dims = Sinfo.dims();
        af_dtype Stype  = Sinfo.getType();

        DIM_ASSERT(1, S_dims[0] > 1 );
        DIM_ASSERT(1, S_dims[1] > 1 );

        fg::Window* window = reinterpret_cast<fg::Window*>(wind);
        window->makeCurrent();
        fg::Surface* surface = NULL;

        af::array X = range(S_dims, 0);
        af::array Y = range(S_dims, 1);

        switch(Stype) {
            case f32: surface = setup_surface<float>(X.get(), Y.get(), S); break;
            case s32: surface = setup_surface<int  >(X.get(), Y.get(), S); break;
            case u32: surface = setup_surface<uint >(X.get(), Y.get(), S); break;
            case u8 : surface = setup_surface<uchar>(X.get(), Y.get(), S); break;
            default:  TYPE_ERROR(1, Stype);
        }

        if (props->col > -1 && props->row > -1)
            window->draw(props->col, props->row, *surface, props->title);
        else
            window->draw(*surface);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    return AF_ERR_NO_GFX;
#endif
}

af_err af_draw_surface(const af_window wind, const af_array xVals, const af_array yVals, const af_array zVals, const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        ArrayInfo Xinfo = getInfo(xVals);
        af::dim4 X_dims = Xinfo.dims();
        af_dtype Xtype  = Xinfo.getType();

        ArrayInfo Yinfo = getInfo(yVals);
        af::dim4 Y_dims = Yinfo.dims();
        af_dtype Ytype  = Yinfo.getType();

        ArrayInfo Zinfo = getInfo(zVals);
        af::dim4 Z_dims = Zinfo.dims();
        af_dtype Ztype  = Zinfo.getType();

        DIM_ASSERT(2, X_dims == Y_dims);
        DIM_ASSERT(3, X_dims == Z_dims);

        TYPE_ASSERT(Xtype == Ytype);
        TYPE_ASSERT(Ytype == Ztype);

        fg::Window* window = reinterpret_cast<fg::Window*>(wind);
        window->makeCurrent();
        fg::Surface* surface = NULL;

        switch(Xtype) {
            case f32: surface = setup_surface<float>(xVals, yVals , zVals); break;
            case s32: surface = setup_surface<int  >(xVals, yVals , zVals); break;
            case u32: surface = setup_surface<uint >(xVals, yVals , zVals); break;
            case u8 : surface = setup_surface<uchar>(xVals, yVals , zVals); break;
            default:  TYPE_ERROR(1, Xtype);
        }

        if (props->col > -1 && props->row > -1)
            window->draw(props->col, props->row, *surface, props->title);
        else
            window->draw(*surface);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    return AF_ERR_NO_GFX;
#endif
}
