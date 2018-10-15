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

#if AF_API_VERSION >= 37
    /**
        C++ interface for forward pass of 2-D pooling operation

        \param[in]  in array that will be pooled along the first two dimensions
                    assumed structure of input is ( W x H x C x N )
                    this operation can be batched over any number of channels or images in the
                    third and fourth dimensions
        \param[in]  pool_width specifies width of pooling window
        \param[in]  pool_height specifies width of pooling window
        \param[in]  padding_width specifies amount of padding to add to width of image
        \param[in]  padding_height specifies amount of padding to add to height of image
        \param[in]  stride_width specifies stride by which window will be moved along
                    first dimension for each pooling operation
        \param[in]  stride_height specifies stride by which window will be moved along
                    secong dimension for each pooling operation
        \param[in]  pool_type specifies which pooling operation to perform,
                    currently supports af_pooling_max and af_pooling_avg
        \return     pooled array

        \ingroup ml_pooling
    */
    AFAPI array pool2(const array& in,
                      const dim_t pool_width,    const dim_t pool_height,
                      const dim_t padding_width, const dim_t padding_height,
                      const dim_t stride_width,  const dim_t stride_height,
                      af_pooling_type pool_type=AF_POOLING_MAX);

    /**
        C++ interface for backward pass of 2-D pooling operation

        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_input input to forward pass of pooling operation
                    assumed structure of input is ( W x H x C x N )
        \param[in]  pooled_output output from forward pass of pooling operation
        \param[in]  pool_width specifies width of pooling window
        \param[in]  pool_height specifies width of pooling window
        \param[in]  padding_width specifies amount of padding added to width of image
        \param[in]  padding_height specifies amount of padding added to height of image
        \param[in]  stride_width specifies stride by which window moved along
                    first dimension for each pooling operation
        \param[in]  stride_height specifies stride by which window moved along
                    secong dimension for each pooling operation
        \param[in]  pool_type specifies which pooling operation was performed
                    currently supports af_pooling_max and af_pooling_avg
        \return     gradient wrt/input

        \ingroup ml_pooling
    */
    AFAPI array pool2Gradient(const array &incoming_gradient,
                              const array& original_input,
                              const array& pooled_output,
                              const dim_t pool_width,     const dim_t pool_height,
                              const dim_t padding_width,  const dim_t padding_height,
                              const dim_t stride_width,   const dim_t stride_height,
                              af_pooling_type pool_type=AF_POOLING_MAX);
#endif

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 37
    /**
    C Interface for forward pass of 2-D pooling operation
    \param[out] out pooled array
                assumed structure of input is ( W x H x C x N )
                this operation can be batched over any number of channels or images in the
                third and fourth dimensions
    \param[in]  in array that will be pooled along the first two dimensions
                assumed structure of input is ( W x H x C x N )
                this operation can be batched over any number of channels or images in the
                third and fourth dimensions
    \param[in]  pool_width specifies width of pooling window
    \param[in]  pool_height specifies width of pooling window
    \param[in]  padding_width specifies amount of padding to add to width of image
    \param[in]  padding_height specifies amount of padding to add to height of image
    \param[in]  stride_width specifies stride by which window will be moved along
                first dimension for each pooling operation
    \param[in]  stride_height specifies stride by which window will be moved along
                secong dimension for each pooling operation
    \param[in]  pool_type specifies which pooling operation to perform,
                currently supports AF_POOLING_MAX and AF_POOLING_AVG
    \return     \ref AF_SUCCESS if the execution completes properly

        \ingroup ml_pooling
    */
    AFAPI af_err af_pool2(af_array *out, const af_array in,
                          const dim_t pool_width,    const dim_t pool_height,
                          const dim_t padding_width, const dim_t padding_height,
                          const dim_t stride_width,  const dim_t stride_height,
                          af_pooling_type pool_type);

    /**
        C interface for backward pass of 2-D pooling operation

        \param[out] out gradient wrt/input
                    third and fourth dimensions
        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_input input to forward pass of pooling operation
                    assumed structure of input is ( W x H x C x N )
        \param[in]  pooled_output output from forward pass of pooling operation
        \param[in]  pool_width specifies width of pooling window
        \param[in]  pool_height specifies width of pooling window
        \param[in]  padding_width specifies amount of padding added to width of image
        \param[in]  padding_height specifies amount of padding added to height of image
        \param[in]  stride_width specifies stride by which window moved along
                    first dimension for each pooling operation
        \param[in]  stride_height specifies stride by which window moved along
                    secong dimension for each pooling operation
        \param[in]  pool_type specifies which pooling operation was performed
                    currently supports af_pooling_max and af_pooling_avg
        \return     \ref AF_SUCCESS if the execution completes properly

        \ingroup ml_pooling
    */
    AFAPI af_err af_pool2Gradient(af_array *out,
                                  const af_array incoming_gradient,
                                  const af_array original_input,
                                  const af_array pooled_output,
                                  const dim_t pool_width,    const dim_t pool_height,
                                  const dim_t padding_width, const dim_t padding_height,
                                  const dim_t stride_width,  const dim_t stride_height,
                                  af_pooling_type pool_type);
#endif


#ifdef __cplusplus
}
#endif
