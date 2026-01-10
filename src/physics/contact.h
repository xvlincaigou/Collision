/**
 * @file contact.h
 * @brief Contact information for collision detection.
 */
#pragma once

#include "core/common.h"

namespace rigid {

/**
 * @struct Contact
 * @brief Represents a collision contact between bodies or body-environment.
 */
struct Contact {
    Vec3  position;     ///< World-space contact point
    Vec3  normal;       ///< World-space contact normal (points out of body A)
    Float depth;        ///< Penetration depth
    Int   bodyIndexA;   ///< Body A index (-1 for environment)
    Int   bodyIndexB;   ///< Body B index (the penetrating body)

    Contact()
        : position(Vec3::Zero())
        , normal(Vec3::Zero())
        , depth(0.0f)
        , bodyIndexA(-1)
        , bodyIndexB(-1) {}
};

}  // namespace rigid
