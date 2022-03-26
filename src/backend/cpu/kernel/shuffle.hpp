/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/VarPhilox.hpp>
#include <err_cpu.hpp>
#include <kernel/random_engine_philox.hpp>
#include <types.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

using std::array;

namespace cpu {
namespace kernel {

void varPhiloxRound(const uint* const k, uint* const c) {
    uint hi0, lo0, hi1, lo1;
    mulhilo(m4x32_0, c[0], &hi0, &lo0);
    mulhilo(m4x32_1, c[2], &hi1, &lo1);
    c[0] = hi1 ^ c[1] ^ k[0];
    c[1] = lo1;
    c[2] = hi0 ^ c[3] ^ k[1];
    c[3] = lo0;
}

const static unsigned int rounds = 24;
const static uint64_t PHILOX_M2_64_0 = 0xD2B74407B1CE6E93;

uint64_t varPhilox(uint32_t &key, uint32_t *state,
                   uint64_t left_side_bits, uint64_t left_side_mask,
                   uint64_t right_side_bits, uint64_t right_side_mask) {
    for(unsigned r=0; r<rounds; ++r) {
        uint32_t hi, lo;
        mulhilo(PHILOX_M2_64_0, state[0], &hi, &lo);
        lo = (lo << (right_side_bits - left_side_bits)) |
             (state[1] >> left_side_bits);

        state[0] = ((hi ^ key) ^ state[1]) & left_side_mask;
        state[1] = lo & right_side_mask;
        key += w32_0;
    }
    return (uint64_t)state[0] << right_side_bits | (uint64_t) state[1];
}

template<typename T>
void shufflePhilox(Param<T> out, const CParam<T> in, const uintl seed, uintl counter) {
    T *optr = out.get();
    const T * const iptr = in.get();
    af::dim4 odims    = out.dims();
    af::dim4 ostrides = out.strides();

    uint64_t total_bits = common::getVarPhiloxCipherBits(odims[0]);

    uint64_t left_side_bits = total_bits / 2;
    uint64_t left_side_mask = (1ull << left_side_bits) - 1;

    uint64_t right_side_bits = total_bits - left_side_bits;
    uint64_t right_side_mask = (1ull << right_side_bits) - 1;

    std::mt19937_64 random_function;
    uint32_t key;
    key = seed;

    dim_t oidx = 0;

    dim_t nextpow2 = (1 << total_bits);
    for(dim_t m=0; m<nextpow2; ++m) {
        uint32_t state[2] = { uint32_t(counter >> right_side_bits),
                              uint32_t(counter & right_side_mask) };

        uint64_t ridx = varPhilox(key, state,
                                  left_side_bits, left_side_mask,
                                  right_side_bits, right_side_mask);

        if(ridx < odims[0]) {
            //TODO: combine
            for (dim_t w = 0; w < odims[3]; w++) {
                dim_t offW = w * ostrides[3];
                for (dim_t z = 0; z < odims[2]; z++) {
                    dim_t offWZ = offW + z * ostrides[2];
                    for (dim_t y = 0; y < odims[1]; y++) {
                        dim_t offWZY = offWZ + y * ostrides[1];

                        optr[oidx + offWZY] = iptr[ridx + offWZY];
                    }
                }
            }
            ++oidx;
        }
        ++counter;
    }
}

template<typename T>
void shuffle(Param<T> out, const CParam<T> in, const uintl seed, uintl counter) {
    shufflePhilox(out, in, seed, counter);
}

}  // namespace kernel
}  // namespace cpu
