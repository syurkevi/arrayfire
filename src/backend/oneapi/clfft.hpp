/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

//#include <clFFT.h>
#include <common/FFTPlanCache.hpp>
#include <memory.hpp>

#include <cstdio>

namespace oneapi {
typedef clfftPlanHandle PlanType;
typedef std::shared_ptr<PlanType> SharedPlan;

const char *_clfftGetResultString(clfftStatus st);

SharedPlan findPlan(clfftLayout iLayout, clfftLayout oLayout, clfftDim rank,
                    size_t *clLengths, size_t *istrides, size_t idist,
                    size_t *ostrides, size_t odist, clfftPrecision precision,
                    size_t batch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType> {
    friend SharedPlan findPlan(clfftLayout iLayout, clfftLayout oLayout,
                               clfftDim rank, size_t *clLengths,
                               size_t *istrides, size_t idist, size_t *ostrides,
                               size_t odist, clfftPrecision precision,
                               size_t batch);
};
}  // namespace oneapi

