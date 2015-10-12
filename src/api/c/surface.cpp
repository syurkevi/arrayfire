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

#include <ArrayInfo.hpp>
#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <surface.hpp>
#include <reduce.hpp>
#include <join.hpp>
#include <reorder.hpp>
#include <handle.hpp>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
fg::Surface* setup_surface(const af_array xVals, const af_array yVals, const af_array S)
{
    Array<T> xIn = getArray<T>(xVals);
    Array<T> yIn = getArray<T>(yVals);
    Array<T> sIn = getArray<T>(S);

    T xmax = 10;
    T xmin = -10;
    T ymax = 10;
    T ymin = -10;
    T zmax = reduce_all<af_max_t, T, T>(sIn);
    T zmin = reduce_all<af_min_t, T, T>(sIn);
    //T ymax = reduce_all<af_max_t, T, T>(yIn);
    // ymin = reduce_all<af_min_t, T, T>(yIn);

    dim4 rdims(1, 0, 2, 3);


    ArrayInfo Xinfo = getInfo(xVals);
    ArrayInfo Yinfo = getInfo(yVals);
    ArrayInfo Sinfo = getInfo(S);

    //Array<T> Z = join(1, xIn, yIn, sIn); //TODO fixthis!
    //Array<T> P = reorder(Z, rdims);

    af::dim4 X_dims = Xinfo.dims();
    af::dim4 Y_dims = Yinfo.dims();
    af::dim4 S_dims = Sinfo.dims();
    
    Array<T> Z;
    for(int i=0; i<S_dims.dim(0); ++i){
        Z = join(1, Z, join(1, xIn, yIn, S.col(i)));
    }
    /* some form of DIM_ASSERT
       assert( (S_dims[0] == X_dims[0] && S_dims[1] == Y_dims[0]) || 
			(S_dims[0] * S_dims[1] == X_dims[0] && 
			 S_dims[0] * S_dims[1] == Y_dims[0] ) 
    */

    ForgeManager& fgMngr = ForgeManager::getInstance();
    fg::Surface* surface = fgMngr.getSurface(X_dims.elements(), X_dims.elements(), getGLType<T>());
    surface->setColor(1.0, 0.0, 0.0);
    surface->setAxesLimits(xmax, xmin, ymax, ymin, zmin, zmax);
    surface->setXAxisTitle("X Axis");
    surface->setYAxisTitle("Y Axis");
    surface->setZAxisTitle("Z Axis");

    copy_surface<T>(Z, surface);

    return surface;
}
#endif

af_err af_draw_surface(const af_window wind, const af_array xVals, const af_array yVals, const af_array S, const af_cell* const props)
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

        DIM_ASSERT(0, X_dims == Y_dims);
        DIM_ASSERT(0, X_dims == Y_dims);
        DIM_ASSERT(0, Xinfo.isVector());

        TYPE_ASSERT(Xtype == Ytype);

        fg::Window* window = reinterpret_cast<fg::Window*>(wind);
        window->makeCurrent();
        fg::Surface* surface = NULL;

        switch(Xtype) {
            case f32: surface = setup_surface<float  >(xVals, yVals , S); break;
            case s32: surface = setup_surface<int    >(xVals, yVals , S); break;
            case u32: surface = setup_surface<uint   >(xVals, yVals , S); break;
            case u8 : surface = setup_surface<uchar  >(xVals, yVals , S); break;
            default:  TYPE_ERROR(1, Xtype);
        }

        if (props->col>-1 && props->row>-1)
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
