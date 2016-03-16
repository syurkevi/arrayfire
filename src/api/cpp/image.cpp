/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/image.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

#define INSTANTIATE_REAL(T)                                 \
    template<> AFAPI                                        \
    T moments(const array &in, const af_moment moment)      \
    {                                                       \
        double val;                                         \
        printf("moment, %d\n",moment);\
        AF_THROW(af_moments(&val, in.get(), moment));       \
        return (T)(val);                                    \
    }                                                       \


//TODO: INSTANTIATE_CPLX?

INSTANTIATE_REAL(float)
INSTANTIATE_REAL(double)
INSTANTIATE_REAL(int)
INSTANTIATE_REAL(unsigned)
INSTANTIATE_REAL(long)
INSTANTIATE_REAL(unsigned long)
INSTANTIATE_REAL(long long)
INSTANTIATE_REAL(unsigned long long)
INSTANTIATE_REAL(short)
INSTANTIATE_REAL(unsigned short)
INSTANTIATE_REAL(char)
INSTANTIATE_REAL(unsigned char)

#undef INSTANTIATE_REAL

}
