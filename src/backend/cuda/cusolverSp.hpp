/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/unique_handle.hpp>
#include <cusolverSp.h>

DEFINE_HANDLER(cusolverSpHandle_t, cusolverSpCreate, cusolverSpDestroy);

namespace cuda {

const char* errorStringSp(cusolverStatus_t err);

#define CUSOLVER_SP_CHECK(fn)                                                 \
    do {                                                                      \
        cusolverStatus_t _error = fn;                                         \
        if (_error != CUSOLVER_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                              \
            snprintf(_err_msg, sizeof(_err_msg), "CUSOLVER Error (%d): %s\n", \
                     (int)(_error), cuda::errorStringSp(_error));             \
                                                                              \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                              \
        }                                                                     \
    } while (0)

}  // namespace cuda
