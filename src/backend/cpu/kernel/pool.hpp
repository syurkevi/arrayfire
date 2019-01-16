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
#include <af/defines.h>
#include <math.hpp>

namespace cpu
{
namespace kernel
{

static void conv_cpu_ref (
    const T_ELEM* inputData,
    const T_ELEM* filterData,
    T_ELEM*       outputData,
    float        alpha,
    float        beta,
    bool         isNchw,
    const int*   inDims,
    const int*   filDims,
    const int*   outDims,
    const int*   inStride,
    const int*   outStride,
    const int*   stride,
    const int*   pad,
    const int*   dilation,
    int          nbDims
) {
    int imDims = nbDims - 2 ;

    int filStride[8] = {0} ;
    generateStrides(filDims, filStride, nbDims, isNchw);

    bool isConv = true; //(CUDNN_CONVOLUTION == mode) ;
    // Number of pixels in output
    int nPixelsOut = 1 ;
    for(int i = 2 ; i < nbDims ; i++)
        nPixelsOut *= outDims[i] ;
    // Number of pixels in filter
    int nPixelsFil = 1 ;
    for(int i = 2 ; i < nbDims ; i++)
        nPixelsFil *= filDims[i] ;
    // Used to store coordinates
    int filIds[8] = {0} ;
    int outIds[8] = {0} ;
    int inIds [8] = {0} ;
    int tmpIds[8] = {0} ;
    // For each image in the output
    for(int ni = 0 ; ni < outDims[0] ; ni++) {
        // For each feature layer of the output
        for(int ki = 0 ; ki < outDims[1] ; ki++) {
            int outputOffset = ni * outStride[0] + ki * outStride[1] ;
            // Loop over all entries of the result
            for(int outId = 0 ; outId < nPixelsOut ; outId++) {
                // Get output pixel ids
                lin2dim(outId, outIds, outDims+2, imDims) ; // Skip n and k dimensions
                // Now we get the coordinates in input space of the "top left" corner of the filter: multiply by stride and remove pad
                for(int d = 0 ; d < imDims ; d++) {
                    inIds[d] = outIds[d] * stride[d] - pad[d] ;
                }
                // We then accumulate
                float tmp = 0.f;
                for(int ci = 0 ; ci < inDims[1] ; ci++) {
                    int inputOffset = ni * inStride[0] + ci * inStride[1] ;
                    int filterOffset = ki * filStride[0] + ci * filStride[1] ;
                    for(int filId = 0 ; filId < nPixelsFil ; filId ++) {
                        // Get the position of the pixel
                        lin2dim(filId, filIds, filDims+2, imDims) ;
                        // Compute the corresponding output pixel
                        // and check wether we are in the padding area on the fly too (not that for convolution, we flip the image patch (equivalent to flipping the filter patch))
                        bool inside = true ;
                        for(int d = 0 ; d < imDims && inside ; d++) {
                            if (isConv) {
                                tmpIds[d] = inIds[d] + dilation[d] * (filDims[2+d]-1 - filIds[d]) ;
                            } else {
                                tmpIds[d] = inIds[d] + dilation[d] * filIds[d] ;
                            }
                            inside &= (tmpIds[d] >= 0 && tmpIds[d] < inDims[2+d]) ; // If we are in the padding area: stop and skip computations
                        }
                        if(inside) {
                            int actualTmpId = inputOffset + dim2lin(tmpIds, (inStride)+2, imDims) ;
                            //int actualFilId = filterOffset + filId ;
                            int actualFilId = filterOffset + dim2lin(filIds, (filStride)+2, imDims) ;
                            T_ELEM fval = filterData[actualFilId] ;
                            T_ELEM ival = inputData [actualTmpId] ;
                            tmp = doFma(fval, ival, tmp);
                        }
                    }
                }

                // We put the result in the output
                int actualOutId = outputOffset + dim2lin(outIds, (outStride)+2, imDims) ;
                doEpilog(outputData, actualOutId, alpha*tmp, beta);
            }
        }
    }
}


template<typename InT, typename AccT, bool Expand>
void convolve2(Param<InT> out, CParam<InT> signal,
               CParam<AccT> c_filter, CParam<AccT> r_filter,
               Param<InT> temp)
{
    dim_t cflen = (dim_t)c_filter.dims().elements();
    dim_t rflen = (dim_t)r_filter.dims().elements();

    auto oDims = out.dims();
    auto sDims = signal.dims();

    auto oStrides = out.strides();
    auto sStrides = signal.strides();
    auto tStrides = temp.strides();

    for (dim_t b3=0; b3<oDims[3]; ++b3) {

        dim_t i_b3Off = b3*sStrides[3];
        dim_t t_b3Off = b3*tStrides[3];
        dim_t o_b3Off = b3*oStrides[3];

        for (dim_t b2=0; b2<oDims[2]; ++b2) {

            InT const * const iptr = signal.get()+ b2*sStrides[2] + i_b3Off;
            InT *tptr = temp.get() + b2*tStrides[2] + t_b3Off;
            InT *optr = out.get()  + b2*oStrides[2] + o_b3Off;

            convolve2_separable<InT, AccT, 0, Expand>(tptr, iptr, c_filter.get(),
                                                      temp.dims(), sDims, sDims, cflen,
                                                      tStrides, sStrides, c_filter.strides(0));

            convolve2_separable<InT, AccT, 1, Expand>(optr, tptr, r_filter.get(),
                                                      oDims, temp.dims(), sDims, rflen,
                                                      oStrides, tStrides, r_filter.strides(0));
        }
    }
}

}
}
