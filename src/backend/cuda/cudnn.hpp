/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <cudnn.h>
#include <common/defines.hpp>
#include <common/NNHandle.hpp>

namespace cuda
{
typedef cudnnHandle_t NNHandle;

const char * errorString(cudnnStatus_t err);

#define CUDNN_CHECK(fn) do {                    \
        cudnnStatus_t _error = (fn);            \
        if (_error != CUDNN_STATUS_SUCCESS) {   \
            char _err_msg[1024];                \
            snprintf(_err_msg,                  \
                     sizeof(_err_msg),          \
                     "CUDNN Error (%d): %s\n",  \
                     (int)(_error),             \
                     errorString(_error));      \
                                                \
            AF_ERROR(_err_msg,                  \
                     AF_ERR_INTERNAL);          \
        }                                       \
    } while(0)

class cudnnHandle : public common::MatrixAlgebraHandle<cudnnHandle, NNHandle>
{
    public:
        void createHandle(NNHandle* handle);
        void destroyHandle(NNHandle handle) {
            CUDNN_CHECK(cudnnDestroy(handle));
        }
};
}
