/**
 * @file environment.h
 * @brief Simulation environment with boundary planes.
 */
#pragma once

#include "core/common.h"

namespace rigid {

/**
 * @class Environment
 * @brief Defines the simulation environment boundaries.
 *
 * The environment is an axis-aligned box defined by 6 boundary planes.
 */
class Environment {
public:
    enum BoundaryId : Int {
        kNegX = 0,
        kPosX,
        kNegY,
        kPosY,
        kNegZ,
        kPosZ,
        kBoundaryCount
    };

    Environment();

    /// Set the environment bounds as an AABB
    void setBounds(const Vec3& minCorner, const Vec3& maxCorner);

    /// Get a boundary plane by ID
    [[nodiscard]] Plane& plane(BoundaryId id) { return planes_[id]; }
    [[nodiscard]] const Plane& plane(BoundaryId id) const { return planes_[id]; }

    /// Get boundary count
    static constexpr Int boundaryCount() { return kBoundaryCount; }

    /// Get the AABB of the environment
    [[nodiscard]] const Vec3& minCorner() const { return minCorner_; }
    [[nodiscard]] const Vec3& maxCorner() const { return maxCorner_; }

private:
    Plane planes_[kBoundaryCount];
    Vec3 minCorner_;
    Vec3 maxCorner_;
};

}  // namespace rigid
