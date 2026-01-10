/**
 * @file body_properties.h
 * @brief Physical properties for rigid bodies.
 */
#pragma once

#include "core/common.h"

namespace rigid {

/**
 * @struct BodyProperties
 * @brief Physical properties of a rigid body (mass, inertia, material).
 */
struct BodyProperties {
    Float mass          = 1.0f;
    Float massInv       = 1.0f;
    Float linearDamping = 0.4f;
    Float angularDamping = 0.1f;
    Float restitution   = 0.0f;
    Float friction      = 0.5f;

    Mat3 inertiaBody    = Mat3::Identity();
    Mat3 inertiaBodyInv = Mat3::Identity();
    Vec3 centerOfMass   = Vec3::Zero();

    // ========================================================================
    // Mutators
    // ========================================================================

    void setMass(Float value) {
        mass = clampPositive(value);
        massInv = 1.0f / mass;
    }

    void setInertiaBody(const Mat3& value) {
        inertiaBody = value;
        inertiaBodyInv = invertSymmetric(inertiaBody);
    }

    void setCenterOfMass(const Vec3& value) {
        centerOfMass = value;
    }

    /// Finalize properties (recompute derived values)
    void finalize() {
        setMass(mass);
        setInertiaBody(inertiaBody);
    }

    // ========================================================================
    // Validation
    // ========================================================================

    [[nodiscard]] bool isValid() const {
        bool positiveMass = mass > 0.0f && std::isfinite(mass);
        bool finiteInertia = inertiaBody.allFinite() && inertiaBodyInv.allFinite();
        return positiveMass && finiteInertia;
    }

    [[nodiscard]] bool isDynamic() const {
        return massInv > 0.0f;
    }

    // ========================================================================
    // Factory methods
    // ========================================================================

    /// Create properties for a static (immovable) body
    static BodyProperties createStatic() {
        BodyProperties props;
        props.mass = std::numeric_limits<Float>::infinity();
        props.massInv = 0.0f;
        return props;
    }

    /// Create default dynamic body properties
    static BodyProperties createDynamic(Float mass = 1.0f) {
        BodyProperties props;
        props.setMass(mass);
        return props;
    }
};

}  // namespace rigid
