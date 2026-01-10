/**
 * @file rigid_body.h
 * @brief Rigid body representation with state and properties.
 */
#pragma once

#include "body_properties.h"
#include "core/common.h"
#include "core/mesh.h"

namespace rigid {

/**
 * @struct BodyState
 * @brief Dynamic state of a rigid body (position, velocity, orientation, scale).
 */
struct BodyState {
    Vec3 position    = Vec3::Zero();
    Quat orientation = Quat::Identity();
    Vec3 linearVel   = Vec3::Zero();
    Vec3 angularVel  = Vec3::Zero();
    Float scale      = 1.0f;  ///< Uniform scale factor (affects radius)

    /// Get the rotation matrix from orientation
    [[nodiscard]] Mat3 rotationMatrix() const {
        return orientation.toRotationMatrix();
    }

    /// Transform a local point to world space (applies scale)
    [[nodiscard]] Vec3 localToWorld(const Vec3& localPt) const {
        return orientation * (localPt * scale) + position;
    }

    /// Transform a world point to local space (applies inverse scale)
    [[nodiscard]] Vec3 worldToLocal(const Vec3& worldPt) const {
        Vec3 local = orientation.inverse() * (worldPt - position);
        return (scale > 0.0f) ? local / scale : local;
    }

    /// Get local point scaled but not transformed
    [[nodiscard]] Vec3 scalePoint(const Vec3& localPt) const {
        return localPt * scale;
    }
};

/**
 * @class RigidBody
 * @brief A rigid body with geometry, physical properties, and state.
 */
class RigidBody {
public:
    RigidBody() = default;
    explicit RigidBody(String name);

    // ========================================================================
    // Mesh Management
    // ========================================================================

    bool loadMesh(const String& path, bool buildBVH = true);
    void setMesh(SharedPtr<Mesh> mesh);

    [[nodiscard]] bool hasMesh() const { return mesh_ != nullptr; }
    [[nodiscard]] Mesh& mesh() { return *mesh_; }
    [[nodiscard]] const Mesh& mesh() const { return *mesh_; }

    // ========================================================================
    // Properties and State
    // ========================================================================

    [[nodiscard]] BodyProperties& properties() { return properties_; }
    [[nodiscard]] const BodyProperties& properties() const { return properties_; }

    [[nodiscard]] BodyState& state() { return state_; }
    [[nodiscard]] const BodyState& state() const { return state_; }

    void setProperties(const BodyProperties& props);
    void setState(const BodyState& state);

    [[nodiscard]] const String& name() const { return name_; }

    // ========================================================================
    // Dynamics
    // ========================================================================

    [[nodiscard]] bool isDynamic() const { return properties_.isDynamic(); }

    /// Apply a force at a world-space point
    void applyForce(const Vec3& force, const Vec3& worldPoint);

    /// Apply a torque in world space
    void applyTorque(const Vec3& torque);

    /// Clear accumulated forces and torques
    void clearAccumulators();

    [[nodiscard]] const Vec3& accumulatedForce() const { return forceAccum_; }
    [[nodiscard]] const Vec3& accumulatedTorque() const { return torqueAccum_; }

    /// Get world-space inverse inertia tensor
    [[nodiscard]] const Mat3& worldInertiaInv();

    // ========================================================================
    // Bounds
    // ========================================================================

    /// Get world-space AABB (computed lazily)
    [[nodiscard]] AABB& worldBounds();

private:
    void updateWorldBounds();
    void syncWorldInertia();

    String name_;
    SharedPtr<Mesh> mesh_;
    BodyProperties properties_;
    BodyState state_;

    Vec3 forceAccum_  = Vec3::Zero();
    Vec3 torqueAccum_ = Vec3::Zero();

    AABB worldBounds_;
    Mat3 worldInertiaInv_ = Mat3::Identity();

    bool boundsDirty_   = true;
    bool inertiaDirty_  = true;
};

}  // namespace rigid
