/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void compact(__global int *reduced_block_sizes,
             __global Tk *oKeys,
             KParam oKInfo,
             __global To *oVals,
             KParam oVInfo,
             const __global Tk *iKeys,
             KParam iKInfo,
             const __global To *iVals,
             KParam iVInfo)
{
    const uint lid = get_local_id(0);
    const uint bid = get_group_id(0);
    const uint gid = get_global_id(0);

    Tk k;
    To v;

    // reduced_block_sizes should have inclusive sum of block sizes
    int nwrite   = (bid == 0) ? reduced_block_sizes[0] : reduced_block_sizes[bid] - reduced_block_sizes[bid - 1];
    int writeloc = (bid == 0) ? 0 : reduced_block_sizes[bid - 1];

    k = iKeys[gid];
    v = iVals[gid];

    if(lid < nwrite) {
        oKeys[writeloc + lid] = k;
        oVals[writeloc + lid] = v;
    }
}
