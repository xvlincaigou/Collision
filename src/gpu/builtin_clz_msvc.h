/*
 * MSVC Compatibility Layer for __builtin_clz
 */
#ifndef PHYS3D_CLZ_COMPAT_HPP
#define PHYS3D_CLZ_COMPAT_HPP

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)

inline int __builtin_clz(unsigned int x) 
{
    unsigned long msb;
    if (_BitScanReverse(&msb, x))
        return 31 - msb;
    return 32;
}
#endif

#endif // PHYS3D_CLZ_COMPAT_HPP
