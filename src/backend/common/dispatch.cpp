/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "dispatch.hpp"

long long unsigned nextpow2(long long unsigned x) {
    x = x - 1ULL;
    x = x | (x >> 1ULL);
    x = x | (x >> 2ULL);
    x = x | (x >> 4ULL);
    x = x | (x >> 8ULL);
    x = x | (x >> 16ULL);
    x = x | (x >> 32ULL);
    return x + 1ULL;
}
