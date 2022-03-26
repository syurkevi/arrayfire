#pragma once

#include <af/defines.h>

namespace common {

inline uint64_t getVarPhiloxCipherBits(uint64_t capacity) {
    if(capacity == 0) {
        return 0;
    }
    uint64_t i = 0;
    capacity--;
    while(capacity != 0) {
        i++;
        capacity >>= 1;
    }
    return std::max(i, uint64_t(4));
}

}
