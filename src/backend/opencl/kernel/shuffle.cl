/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
 ********************************************************/

#define PHILOX_M2_64_0 0xD2B74407B1CE6E93
#define w32_0 0x9E3779B9
#define rounds 24

#if __OPENCL_VERSION__ < 200
size_t work_group_scan_inclusive_add(local size_t *wg_temp, local size_t *arr) {
    local int *active_buf;

    const int lid = get_local_id(0);
    size_t val    = arr[lid];
    active_buf    = arr;

    bool swap_buffer = false;
    for (int off = 1; off <= DIMX; off *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= off) { val = val + active_buf[lid - off]; }
        swap_buffer     = !swap_buffer;
        active_buf      = swap_buffer ? wg_temp : arr;
        active_buf[lid] = val;
    }

    size_t res = active_buf[lid];
    return res;
}
#endif  // __OPENCL_VERSION__ < 200


inline void mulhilo(const uint a, const uint b, uint *const hi, uint *const lo) {
    *hi = mul_hi(a, b);
    *lo = a * b;
}

inline unsigned long varlenPhilox(unsigned int *key, unsigned int state[2],
                                  unsigned long left_side_bits,
                                  unsigned long left_side_mask,
                                  unsigned long right_side_bits,
                                  unsigned long right_side_mask) {
    for(unsigned r=0; r<rounds; ++r) {
        unsigned int hi, lo;
        mulhilo(PHILOX_M2_64_0, state[0], &hi, &lo);
        lo = (lo << (right_side_bits - left_side_bits)) |
             (state[1] >> left_side_bits);

        state[0] = ((hi ^ *key) ^ state[1]) & left_side_mask;
        state[1] = lo & right_side_mask;

        *key += w32_0;
    }
    return (unsigned long)state[0] << right_side_bits | (unsigned long) state[1];
}

struct lookback {
    unsigned long localsum;
    unsigned long inclusive_prefix;
    bool local_complete;
    bool prefix_complete;
};

kernel void shufflePhilox1Pass(global T* out, const KParam oinfo,
                               global T const * const in, const KParam iinfo,
                               global volatile struct lookback *lookback_table,
                               global int *block_counter,
                               unsigned long seed,
                               unsigned long counter,
                               uint nBits,
                               unsigned long left_side_bits,
                               unsigned long left_side_mask,
                               unsigned long right_side_bits,
                               unsigned long right_side_mask) {
    local uint partition_id;
    if(get_local_id(0) == 0) {
        partition_id = atomic_add(block_counter, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint index  = partition_id * get_local_size(0) + get_local_id(0);

    uint key;
    key = seed;
    uint state[2] = { (uint)((counter + index) >> right_side_bits),
                      (uint)((counter + index) & right_side_mask) };

    //TODO: fix varlen casting issues
    unsigned long ridx = varlenPhilox(&key, state,
                                      left_side_bits, left_side_mask,
                                      right_side_bits, right_side_mask);
    size_t readFlag = (ridx < iinfo.dims[0]);

#if __OPENCL_VERSION__ < 200
    local size_t wg_temp[DIMX];
    local size_t read_flags[DIMX];

    read_flags[get_local_id(0)] = readFlag;
    readFlag = work_group_scan_inclusive_add(wg_temp, read_flags);
#else
    // TODO: ifdefs for block scan, if not supported on card
    readFlag = work_group_scan_inclusive_add(readFlag);
#endif

    local unsigned long running_prefix;

    // first blocks to be scheduled will fill in lookback table and determine
    // inclusive prefixes for write indexes
    if(partition_id < get_num_groups(0)) {

        bool amLastLocalThread = (get_local_id(0) == get_local_size(0)-1);
        // final thread of each block will set local scan
        if(amLastLocalThread) {
            lookback_table[partition_id].localsum = readFlag;
            mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            lookback_table[partition_id].local_complete = true;

            // final thread of first block will also set completed
            // inclusive_prefix for further blocks
            if(partition_id == 0) {
                lookback_table[0].inclusive_prefix = readFlag;
                mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                lookback_table[0].prefix_complete = true;
            }
        }

        mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        if(partition_id > 0) {
            int lookback_id = partition_id - 1;
            bool check_prefix_ready = false;

            // traverse lookback table to obtain required ids for exclusive prefix
            if(amLastLocalThread) {
                running_prefix = 0;
                struct lookback l;
                do {
                    l = lookback_table[lookback_id];
                    check_prefix_ready = l.prefix_complete;
                    bool check_local_ready  = l.local_complete;

                    if(check_local_ready && !check_prefix_ready && lookback_id > 0) {
                        running_prefix += l.localsum;
                        lookback_id--;
                    }
                } while(!check_prefix_ready);
                running_prefix += l.inclusive_prefix;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            readFlag += running_prefix;

            // final thread to update this block's final prefix
            if(amLastLocalThread) {
                lookback_table[partition_id].inclusive_prefix = readFlag;
                mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                lookback_table[partition_id].prefix_complete = true;
            }
        }
    }

    if(ridx < oinfo.dims[0] && (readFlag - 1) < oinfo.dims[0]) {
        out[(readFlag - 1)] = in[ridx];
    }
}

kernel void shufflePhilox2Pass(global T* out,             const KParam oinfo,
                               global T const * const in, const KParam iinfo,
                               global volatile struct lookback *lookback_table,
                               unsigned long seed,
                               unsigned long counter,
                               uint nBits,
                               unsigned long left_side_bits,
                               unsigned long left_side_mask,
                               unsigned long right_side_bits,
                               unsigned long right_side_mask,
                               unsigned nblk_x, unsigned nblk_y) {
    const int bzid = get_group_id(0) / nblk_x;
    const int bwid = get_group_id(1) / nblk_y;

    const int bxid = get_group_id(0) - bzid * nblk_x;
    const int byid = get_group_id(1) - bwid * nblk_y;

    if (bwid >= iinfo.dims[3] || bzid >= iinfo.dims[2] || byid >= iinfo.dims[1]) {
        return;
    }
    if(bwid == 0 && bzid == 0 && byid == 0) {
        return;
    }

    uint index  = bxid * get_local_size(0) + get_local_id(0);

    uint key;
    key = seed;
    uint state[2] = { (uint)((counter + index) >> right_side_bits),
                      (uint)((counter + index) & right_side_mask) };

    //TODO: fix varlen casting issues
    unsigned long ridx = varlenPhilox(&key, state,
                                      left_side_bits, left_side_mask,
                                      right_side_bits, right_side_mask);
    size_t readFlag = (ridx < iinfo.dims[0]);

#if __OPENCL_VERSION__ < 200
    local size_t wg_temp[DIMX];
    local size_t read_flags[DIMX];

    read_flags[get_local_id(0)] = readFlag;
    readFlag = work_group_scan_inclusive_add(wg_temp, read_flags);
#else
    // TODO: ifdefs for block scan, if not supported on card
    readFlag = work_group_scan_inclusive_add(readFlag);
#endif

    local unsigned long running_prefix;

    if(bxid > 0) {
        bool amLastLocalThread = (get_local_id(0) == get_local_size(0) - 1);

        struct lookback l;
        if(amLastLocalThread) {
            l = lookback_table[bxid-1];
            running_prefix = l.inclusive_prefix;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        readFlag += running_prefix;
    }

    uint offYZW =
        bwid * iinfo.strides[3] + bzid * iinfo.strides[2] + byid * iinfo.strides[1];

    if(ridx < oinfo.dims[0] && (readFlag - 1) < oinfo.dims[0]) {
        out[offYZW + (readFlag - 1)] = in[offYZW + ridx];
    }
}

