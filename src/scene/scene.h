/**
 * @file scene.h
 * @brief Scene container for rigid bodies and environment.
 */
#pragma once

#include "core/common.h"
#include "environment.h"
#include "physics/rigid_body.h"

namespace rigid {

/**
 * @class Scene
 * @brief Container for all simulation objects.
 */
class Scene {
public:
    Scene() = default;

    // ========================================================================
    // Body Management
    // ========================================================================

    /// Create a new rigid body with the given name
    RigidBody& createBody(const String& name);

    /// Get a body by index (returns nullptr if invalid)
    [[nodiscard]] RigidBody* body(Int index);
    [[nodiscard]] const RigidBody* body(Int index) const;

    /// Get the list of all bodies
    [[nodiscard]] Vector<UniquePtr<RigidBody>>& bodies() { return bodies_; }
    [[nodiscard]] const Vector<UniquePtr<RigidBody>>& bodies() const { return bodies_; }

    /// Get the number of bodies
    [[nodiscard]] Int bodyCount() const { return static_cast<Int>(bodies_.size()); }

    /// Clear all bodies
    void clearBodies() { bodies_.clear(); }

    /// Check if scene is empty
    [[nodiscard]] bool isEmpty() const { return bodies_.empty(); }

    // ========================================================================
    // Environment
    // ========================================================================

    [[nodiscard]] Environment& environment() { return environment_; }
    [[nodiscard]] const Environment& environment() const { return environment_; }

    void setEnvironmentBounds(const Vec3& minCorner, const Vec3& maxCorner) {
        environment_.setBounds(minCorner, maxCorner);
    }

private:
    Vector<UniquePtr<RigidBody>> bodies_;
    Environment environment_;
};

}  // namespace rigid
