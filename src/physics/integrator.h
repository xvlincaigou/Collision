/*
 * Time Integration System
 */
#ifndef PHYS3D_TIME_INTEGRATOR_HPP
#define PHYS3D_TIME_INTEGRATOR_HPP

#include "collision_detector.h"
#include "force_builder.h"
#include "core/common.h"
#include "scene/scene.h"

namespace phys3d {

/*
 * TimeIntegrator - Implicit Euler integration with penalty response
 */
class TimeIntegrator 
{
public:
    TimeIntegrator();

    void prepare(World& world);
    void advance(World& world);

    void setIterationLimit(IntType limit) { m_iterLimit = limit; }
    void setConvergenceThreshold(RealType thresh) { m_convergenceThresh = thresh; }
    void setDeltaTime(RealType dt) { m_deltaTime = dt; }

    [[nodiscard]] IntType iterationLimit() const { return m_iterLimit; }
    [[nodiscard]] RealType convergenceThreshold() const { return m_convergenceThresh; }
    [[nodiscard]] RealType deltaTime() const { return m_deltaTime; }

    [[nodiscard]] CollisionSystem& collisionSystem() { return m_collisionSystem; }
    [[nodiscard]] const CollisionSystem& collisionSystem() const { return m_collisionSystem; }

private:
    void assembleMassMatrix(World& world, SparseGrid& M);
    void applyVelocityUpdates(World& world, const VectorN& deltaV);
    bool solveLinearSystem(const SparseGrid& A, const VectorN& b, VectorN& x);

    CollisionSystem m_collisionSystem;
    ForceAssembler m_forceAssembler;

    SparseGrid m_massMatrix;
    SparseGrid m_stiffnessMatrix;
    SparseGrid m_systemMatrix;
    VectorN m_forceVector;
    VectorN m_velocityDelta;
    VectorN m_rightHandSide;

    IntType  m_iterLimit;
    RealType m_convergenceThresh;
    RealType m_deltaTime;
};

}  // namespace phys3d

namespace rigid {
    using Integrator = phys3d::TimeIntegrator;
}

#endif // PHYS3D_TIME_INTEGRATOR_HPP
