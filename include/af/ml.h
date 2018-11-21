/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
class array;
class dim4;

#if AF_API_VERSION >= 37
    /**
        C++ interface for calculating backward pass gradient of 2D convolution

        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_signal input signal to forward pass of convolution
                    assumed structure of input is ( W x H x C x N )
        \param[in]  original_filter input filter to forward pass of convolution
                    assumed structure of input is ( W x H x C x N )
        \param[in]  convolved_output output from forward pass of convolution
        \param[in]  stride specifies width of pooling window
        \param[in]  padding specifies width of pooling window
        \param[in]  dilation specifies amount of padding added to width of image
        \param[in]  gradType specifies which gradient to return
        \return     gradient wrt/gradType

        \ingroup ml_convolution
    */
    AFAPI array convolve2Gradient(const array& incoming_gradient,
                                  const array& original_signal,
                                  const array& original_filter,
                                  const array& convolved_output,
                                  const dim4 stride, const dim4 padding, const dim4 dilation,
                                  af_conv_gradient_type gradType);

#endif

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 37
    /**
        C++ interface for calculating backward pass gradient of 2D convolution

        \param[out] out gradient wrt/gradType
        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_signal input signal to forward pass of convolution
                    assumed structure of input is ( W x H x C x N )
        \param[in]  original_filter input filter to forward pass of convolution
                    assumed structure of input is ( W x H x C x N )
        \param[in]  convolved_output output from forward pass of convolution
        \param[in]  stride_dims specifies number of stride dimensions. A single value
                    will use the same value for the stride in both x,y directions
        \param[in]  strides array of stride values
        \param[in]  padding_dims number of padding dimensions. A single value
                    will use the same value for the stride in both x,y directions
        \param[in]  paddings array of padding values
        \param[in]  dilation_dims number of dilation dimensions. A single value
                    will use the same value for the stride in both x,y directions
        \param[in]  dilations array of dilation values
        \param[in]  gradType specifies which gradient to return
        \return     \ref AF_SUCCESS if the execution completes properly

        \ingroup ml_convolution
    */
    AFAPI af_err af_convolve2Gradient(af_array *out,
                                      const af_array incoming_gradient,
                                      const af_array original_signal,
                                      const af_array original_filter,
                                      const af_array convolved_output,
                                      const unsigned stride_dims,   const dim_t *strides,
                                      const unsigned padding_dims,  const dim_t *paddings,
                                      const unsigned dilation_dims, const dim_t *dilations,
                                      af_conv_gradient_type gradType);
#endif


#ifdef __cplusplus
}
#endif
