/**
 * @file integrator.h
 * @brief Time integrator for rigid body dynamics.
 */
#pragma once

#include "collision_detector.h"
#include "force_builder.h"
#include "core/common.h"
#include "scene/scene.h"

namespace rigid {

/**
 * @class Integrator
 * @brief Implicit Euler integrator with penalty-based collision response.
 */
class Integrator {
public:
    Integrator();

    /// Initialize with a scene
    void initialize(Scene& scene);

    /// Advance the simulation by one time step
    void step(Scene& scene);

    // Configuration
    void setMaxIterations(Int iter) { maxIterations_ = iter; }
    void setTolerance(Float tol) { tolerance_ = tol; }
    void setTimeStep(Float dt) { timeStep_ = dt; }

    [[nodiscard]] Int maxIterations() const { return maxIterations_; }
    [[nodiscard]] Float tolerance() const { return tolerance_; }
    [[nodiscard]] Float timeStep() const { return timeStep_; }

    /// Get the collision detector
    [[nodiscard]] CollisionDetector& collisionDetector() { return collisionDetector_; }
    [[nodiscard]] const CollisionDetector& collisionDetector() const { return collisionDetector_; }

private:
    void buildMassMatrix(Scene& scene, SparseMat& M);
    void updateBodyStates(Scene& scene, const VecX& deltaV);
    bool solve(const SparseMat& A, const VecX& b, VecX& x);

    CollisionDetector collisionDetector_;
    ForceBuilder forceBuilder_;

    SparseMat massMatrix_;
    SparseMat jacobian_;
    SparseMat systemMatrix_;
    VecX forceVector_;
    VecX deltaV_;
    VecX rhs_;

    Int   maxIterations_;
    Float tolerance_;
    Float timeStep_;
};

}  // namespace rigid
