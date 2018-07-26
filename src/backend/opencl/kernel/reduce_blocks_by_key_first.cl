/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

void inplace_blockscan_inclusive_add(__local Tk *arr) {
    __local Tk tmp[DIMX];
    __local Tk *l_val = arr;

    const int lid = get_local_id(0);
    bool wbuf = 0;

    Tk val = arr[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = 1; off <= DIMX; off *= 2) {
        if (lid >= off) val = val + l_val[lid - off];

        wbuf = 1 - wbuf;
        l_val = wbuf ? tmp : arr;
        l_val[lid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(arr != l_val)
        arr = l_val;

    barrier(CLK_LOCAL_MEM_FENCE);
}


__kernel
void reduce_blocks_by_key_first(__global int *reduced_block_sizes,
                                __global Tk *oKeys,
                                KParam oKInfo,
                                __global To *oVals,
                                KParam oVInfo,
                                const __global Tk *iKeys,
                                KParam iKInfo,
                                const __global Ti *iVals,
                                KParam iVInfo,
                                int change_nan, To nanval, int n)
{
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);

    __local Tk keys[DIMX];
    __local Ti vals[DIMX];

    __local Tk reduced_keys[DIMX];
    __local To reduced_vals[DIMX];

    __local int unique_flags[DIMX];
    __local int unique_ids[DIMX];

    const To init_val = init;

    //
    // will hold final number of reduced elements in block
    __local int reducedBlockSize;

    if(lid == 0) {
        reducedBlockSize = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // load keys and values to threads
    Tk k;
    To v;
    if(gid < iKInfo.dims[0]) {
        k = iKeys[gid];
        v = transform(iVals[gid]);
        if (change_nan) v = !IS_NAN(v) ? v : nanval;
    }

    keys[lid] = k;
    vals[lid] = v;

    reduced_keys[lid] = k;
    barrier(CLK_LOCAL_MEM_FENCE);

    // mark threads containing unique keys
    int eq_check = (lid > 0) ? (k != reduced_keys[lid - 1]) : 0;
    char unique_flag = (eq_check || (lid == 0)) && (gid < n);
    unique_flags[lid] = unique_flag;

    unique_ids[lid] = unique_flag;
    inplace_blockscan_inclusive_add(unique_ids);
    int unique_id = unique_ids[lid];

    if(lid == DIMX - 1)
        reducedBlockSize = unique_id;

    for (int off = 1; off < DIMX; off *= 2) {
        if(lid + off < DIMX && lid + off < n) {
            int test_unique_id = unique_ids[lid + off];
            if(unique_id == test_unique_id) {
                v = binOp(v, vals[lid + off]);
                vals[lid] = v;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(unique_flag) {
        reduced_keys[unique_id-1] = k;
        reduced_vals[unique_id-1] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int bid = get_group_id(0);
    if(lid < reducedBlockSize) {
        oKeys[bid * DIMX + lid] = reduced_keys[lid];
        oVals[bid * DIMX + lid] = reduced_vals[lid];
    }
    reduced_block_sizes[bid] = reducedBlockSize;
}
