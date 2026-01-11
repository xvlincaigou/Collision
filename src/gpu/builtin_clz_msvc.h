#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)

inline int __builtin_clz(unsigned int x) {
    unsigned long msb;
    if (_BitScanReverse(&msb, x))
        return 31 - msb;
    return 32;   // x == 0
}
#endif