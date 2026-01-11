/*
 * Implementation: ForceAssembler and TimeIntegrator classes
 */
#include "integrator.h"

#include <iostream>

namespace phys3d {

/* ========== ForceAssembler Implementation ========== */

ForceAssembler::ForceAssembler()
    : m_gravityAccel(static_cast<RealType>(0), static_cast<RealType>(0), static_cast<RealType>(-9.81))
    , m_contactStiffness(static_cast<RealType>(100))
    , m_frictionStiffness(static_cast<RealType>(0.1))
    , m_dofTotal(0)
{}

void ForceAssembler::configure(IntType dofTotal) 
{
    m_dofTotal = dofTotal;
}

void ForceAssembler::buildForceVector(World& world, const DynArray<ContactPoint>& contacts, VectorN& forceOut) 
{
    forceOut.setZero();
    accumulateGravity(world, forceOut);
    accumulateContactForces(world, contacts, forceOut);
}

void ForceAssembler::accumulateGravity(World& world, VectorN& forceOut) 
{
    IntType entityCount = world.entityCount();
    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entity = world.entity(i);
        if (!entity || !entity->isMovable()) continue;

        IntType dofIdx = computeDofOffset(i);
        RealType mass = entity->material().totalMass;
        forceOut.segment<3>(dofIdx) += mass * m_gravityAccel;
    }
}

void ForceAssembler::accumulateContactForces(World& world, const DynArray<ContactPoint>& contacts, VectorN& forceOut) 
{
    for (const auto& contact : contacts) 
    {
        Point3 penalty = m_contactStiffness * contact.depth * contact.depth * contact.direction;

        if (contact.entityB >= 0) 
        {
            DynamicEntity* entityB = world.entity(contact.entityB);
            if (entityB && entityB->isMovable()) 
            {
                IntType dofIdxB = computeDofOffset(contact.entityB);
                forceOut.segment<3>(dofIdxB) += penalty;

                Point3 leverB = contact.location - entityB->kinematic().translation;
                forceOut.segment<3>(dofIdxB + 3) += leverB.cross(penalty);
            }
        }

        if (contact.entityA >= 0) 
        {
            DynamicEntity* entityA = world.entity(contact.entityA);
            if (entityA && entityA->isMovable()) 
            {
                IntType dofIdxA = computeDofOffset(contact.entityA);
                forceOut.segment<3>(dofIdxA) -= penalty;

                Point3 leverA = contact.location - entityA->kinematic().translation;
                forceOut.segment<3>(dofIdxA + 3) += leverA.cross(-penalty);
            }
        }
    }
}

void ForceAssembler::buildStiffnessMatrix(World& /* world */,
                                           const DynArray<ContactPoint>& contacts,
                                           SparseGrid& stiffnessOut) 
{
    accumulateContactStiffness(contacts, stiffnessOut);
}

void ForceAssembler::accumulateContactStiffness(const DynArray<ContactPoint>& contacts, SparseGrid& stiffnessOut) 
{
    SparseEntryList entries;

    for (const auto& contact : contacts) 
    {
        Matrix33 Pn = contact.direction * contact.direction.transpose();
        Matrix33 Pt = Matrix33::Identity() - Pn;
        Matrix33 Klocal = -m_contactStiffness * Pn - m_frictionStiffness * Pt;

        if (contact.entityB >= 0) 
        {
            IntType dofIdxB = computeDofOffset(contact.entityB);
            for (int r = 0; r < 3; ++r) 
            {
                for (int c = 0; c < 3; ++c) 
                {
                    entries.emplace_back(dofIdxB + r, dofIdxB + c, Klocal(r, c));
                }
            }
        }

        if (contact.entityA >= 0) 
        {
            IntType dofIdxA = computeDofOffset(contact.entityA);
            for (int r = 0; r < 3; ++r) 
            {
                for (int c = 0; c < 3; ++c) 
                {
                    entries.emplace_back(dofIdxA + r, dofIdxA + c, Klocal(r, c));
                }
            }
        }
    }

    stiffnessOut.setFromTriplets(entries.begin(), entries.end());
}

/* ========== TimeIntegrator Implementation ========== */

TimeIntegrator::TimeIntegrator()
    : m_iterLimit(20)
    , m_convergenceThresh(static_cast<RealType>(1e-2))
    , m_deltaTime(static_cast<RealType>(1e-2))
{}

void TimeIntegrator::prepare(World& world) 
{
    IntType entityCount = world.entityCount();
    IntType dof = entityCount * 6;

    m_forceAssembler.configure(dof);

    m_massMatrix.resize(dof, dof);
    m_stiffnessMatrix.resize(dof, dof);
    m_systemMatrix.resize(dof, dof);
    m_forceVector.resize(dof);
    m_velocityDelta.resize(dof);
    m_rightHandSide.resize(dof);
}

void TimeIntegrator::advance(World& world) 
{
    IntType entityCount = world.entityCount();
    IntType dof = entityCount * 6;

    if (m_massMatrix.rows() != dof) 
    {
        prepare(world);
    }

    for (IntType i = 0; i < entityCount; ++i) 
    {
        if (auto* entity = world.entity(i)) 
        {
            (void)entity->worldExtent();
        }
    }

    m_collisionSystem.enableDeviceAcceleration(true);
    m_collisionSystem.performDetection(world);

    std::cout << "[SIM_INFO] Contact count: "
              << m_collisionSystem.contactList().size() << std::endl;

    m_massMatrix.setZero();
    assembleMassMatrix(world, m_massMatrix);

    m_forceAssembler.buildForceVector(world, m_collisionSystem.contactList(), m_forceVector);

    m_stiffnessMatrix.setZero();
    m_forceAssembler.buildStiffnessMatrix(world, m_collisionSystem.contactList(), m_stiffnessMatrix);

    SparseGrid stiffnessTerm = m_stiffnessMatrix * (m_deltaTime * m_deltaTime);
    m_systemMatrix = m_massMatrix - stiffnessTerm;
    m_rightHandSide = m_forceVector * m_deltaTime;

    m_velocityDelta.setZero();
    if (solveLinearSystem(m_systemMatrix, m_rightHandSide, m_velocityDelta)) 
    {
        applyVelocityUpdates(world, m_velocityDelta);
    }
}

void TimeIntegrator::assembleMassMatrix(World& world, SparseGrid& M) 
{
    SparseEntryList entries;
    IntType entityCount = world.entityCount();

    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entity = world.entity(i);
        if (!entity) continue;

        IntType dofIdx = computeDofOffset(i);

        if (!entity->isMovable()) 
        {
            for (int k = 0; k < 6; ++k) 
            {
                entries.emplace_back(dofIdx + k, dofIdx + k, static_cast<RealType>(1e10));
            }
        } 
        else 
        {
            RealType m = entity->material().totalMass;
            for (int k = 0; k < 3; ++k) 
            {
                entries.emplace_back(dofIdx + k, dofIdx + k, m);
            }

            Matrix33 Iinv = entity->worldInertiaInverse();
            Matrix33 I = Iinv.inverse();

            for (int r = 0; r < 3; ++r) 
            {
                for (int c = 0; c < 3; ++c) 
                {
                    entries.emplace_back(dofIdx + 3 + r, dofIdx + 3 + c, I(r, c));
                }
            }
        }
    }

    M.setFromTriplets(entries.begin(), entries.end());
}

bool TimeIntegrator::solveLinearSystem(const SparseGrid& A, const VectorN& b, VectorN& x) 
{
    Eigen::ConjugateGradient<SparseGrid, Eigen::Lower | Eigen::Upper> solver;
    solver.setMaxIterations(m_iterLimit);
    solver.setTolerance(m_convergenceThresh);
    solver.compute(A);

    if (solver.info() != Eigen::Success) 
    {
        return false;
    }

    x = solver.solve(b);
    return solver.info() == Eigen::Success;
}

void TimeIntegrator::applyVelocityUpdates(World& world, const VectorN& deltaV) 
{
    IntType entityCount = world.entityCount();

    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entity = world.entity(i);
        if (!entity || !entity->isMovable()) continue;

        IntType dofIdx = computeDofOffset(i);
        Point3 dvLin = deltaV.segment<3>(dofIdx);
        Point3 dvAng = deltaV.segment<3>(dofIdx + 3);

        EntityState& state = entity->kinematic();
        const auto& material = entity->material();

        state.velocity += dvLin;
        state.angularRate += dvAng;

        state.velocity *= (static_cast<RealType>(1) - material.linearDrag * m_deltaTime);
        state.angularRate *= (static_cast<RealType>(1) - material.angularDrag * m_deltaTime);

        state.translation += state.velocity * m_deltaTime;

        Point3 w = state.angularRate;
        Rotation4 qw(static_cast<RealType>(0), w.x(), w.y(), w.z());
        Rotation4 dq = qw * state.orientation;

        state.orientation.coeffs() += dq.coeffs() * (static_cast<RealType>(0.5) * m_deltaTime);
        state.orientation.normalize();

        entity->assignKinematic(state);
        entity->resetAccumulators();
    }
}

}  // namespace phys3d
