/**
 * @file force_builder.h
 * @brief Assembles forces and Jacobians for the physics simulation.
 */
#pragma once

#include "contact.h"
#include "core/common.h"
#include "scene/scene.h"

namespace rigid {

/**
 * @class ForceBuilder
 * @brief Computes forces (gravity, penalty, friction) and Jacobians.
 */
class ForceBuilder {
public:
    ForceBuilder();

    /// Initialize with the total degrees of freedom
    void setup(Int dofCount);

    /// Assemble all forces into the force vector
    void assembleForces(Scene& scene, const Vector<Contact>& contacts, VecX& force);

    /// Assemble the stiffness Jacobian
    void assembleJacobian(Scene& scene, const Vector<Contact>& contacts, SparseMat& jacobian);

    // Configuration
    void setGravity(const Vec3& g) { gravity_ = g; }
    void setPenaltyStiffness(Float k) { penaltyStiffness_ = k; }
    void setFrictionStiffness(Float k) { frictionStiffness_ = k; }

    [[nodiscard]] const Vec3& gravity() const { return gravity_; }
    [[nodiscard]] Float penaltyStiffness() const { return penaltyStiffness_; }

private:
    void addGravityForce(Scene& scene, VecX& force);
    void addCollisionForce(Scene& scene, const Vector<Contact>& contacts, VecX& force);
    void addCollisionJacobian(const Vector<Contact>& contacts, SparseMat& jacobian);

    Vec3  gravity_;
    Float penaltyStiffness_;
    Float frictionStiffness_;
    Int   dofCount_;
};

}  // namespace rigid
