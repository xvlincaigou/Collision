/*
 * Implementation: ForceAssembler and TimeIntegrator classes
 * Restructured with helper functions and modified control flow
 */
#include "solver.h"

#include <iostream>
#include <functional>

namespace phys3d {

namespace {

// Helper struct to encapsulate DOF manipulation
struct DofAccessor
{
    static IntType linearOffset(IntType entityId) { return entityId * 6; }
    static IntType angularOffset(IntType entityId) { return entityId * 6 + 3; }
    
    static void addLinear(VectorN& vec, IntType entityId, const Point3& value)
    {
        const IntType off = linearOffset(entityId);
        vec[off]     += value.x();
        vec[off + 1] += value.y();
        vec[off + 2] += value.z();
    }
    
    static void addAngular(VectorN& vec, IntType entityId, const Point3& value)
    {
        const IntType off = angularOffset(entityId);
        vec[off]     += value.x();
        vec[off + 1] += value.y();
        vec[off + 2] += value.z();
    }
    
    static Point3 extractLinear(const VectorN& vec, IntType entityId)
    {
        const IntType off = linearOffset(entityId);
        return Point3(vec[off], vec[off + 1], vec[off + 2]);
    }
    
    static Point3 extractAngular(const VectorN& vec, IntType entityId)
    {
        const IntType off = angularOffset(entityId);
        return Point3(vec[off], vec[off + 1], vec[off + 2]);
    }
};

// Compute penalty force magnitude using quadratic model
inline Point3 computePenaltyForce(RealType stiffness, RealType depth, const Point3& normal)
{
    const RealType magnitude = stiffness * depth * depth;
    return Point3(
        normal.x() * magnitude,
        normal.y() * magnitude,
        normal.z() * magnitude
    );
}

// Compute torque from force at contact point
inline Point3 computeContactTorque(const Point3& contactPos, const Point3& entityPos, const Point3& force)
{
    const Point3 lever(
        contactPos.x() - entityPos.x(),
        contactPos.y() - entityPos.y(),
        contactPos.z() - entityPos.z()
    );
    return lever.cross(force);
}

// Apply force contribution to DOF vector
void applyForceContribution(
    VectorN& forceVec,
    IntType entityIdx,
    const Point3& entityPos,
    const Point3& contactPos,
    const Point3& force,
    RealType sign)
{
    const Point3 scaledForce(
        force.x() * sign,
        force.y() * sign,
        force.z() * sign
    );
    
    DofAccessor::addLinear(forceVec, entityIdx, scaledForce);
    
    const Point3 torque = computeContactTorque(contactPos, entityPos, scaledForce);
    DofAccessor::addAngular(forceVec, entityIdx, torque);
}

// Build local stiffness matrix for contact
void buildLocalStiffnessBlock(
    const Point3& normal,
    RealType contactK,
    RealType frictionK,
    Matrix33& outMatrix)
{
    // Projection matrices: Pn = n⊗n, Pt = I - Pn
    // K = -contactK * Pn - frictionK * Pt
    
    const RealType nx = normal.x(), ny = normal.y(), nz = normal.z();
    
    // Diagonal elements
    outMatrix(0,0) = -contactK * nx * nx - frictionK * (static_cast<RealType>(1) - nx * nx);
    outMatrix(1,1) = -contactK * ny * ny - frictionK * (static_cast<RealType>(1) - ny * ny);
    outMatrix(2,2) = -contactK * nz * nz - frictionK * (static_cast<RealType>(1) - nz * nz);
    
    // Off-diagonal elements (symmetric)
    const RealType coeffDiff = frictionK - contactK;
    outMatrix(0,1) = outMatrix(1,0) = coeffDiff * nx * ny;
    outMatrix(0,2) = outMatrix(2,0) = coeffDiff * nx * nz;
    outMatrix(1,2) = outMatrix(2,1) = coeffDiff * ny * nz;
}

// Insert 3x3 block into sparse triplet list
void insertBlockToTriplets(
    SparseEntryList& triplets,
    IntType rowStart,
    IntType colStart,
    const Matrix33& block)
{
    IntType r = 0;
    do {
        IntType c = 0;
        do {
            triplets.emplace_back(rowStart + r, colStart + c, block(r, c));
            ++c;
        } while (c < 3);
        ++r;
    } while (r < 3);
}

} // anonymous namespace

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
    const IntType entityCount = world.entityCount();
    
    // Process entities in reverse order
    IntType i = entityCount;
    while (i > 0)
    {
        --i;
        DynamicEntity* entity = world.entity(i);
        
        // Combined null and movability check
        if (entity == nullptr)
            continue;
        if (!entity->isMovable())
            continue;

        const RealType mass = entity->material().totalMass;
        const Point3 gravityForce(
            m_gravityAccel.x() * mass,
            m_gravityAccel.y() * mass,
            m_gravityAccel.z() * mass
        );
        
        DofAccessor::addLinear(forceOut, i, gravityForce);
    }
}

void ForceAssembler::accumulateContactForces(World& world, const DynArray<ContactPoint>& contacts, VectorN& forceOut) 
{
    const size_t contactCount = contacts.size();
    size_t idx = 0;
    
    while (idx < contactCount)
    {
        const ContactPoint& contact = contacts[idx];
        const Point3 penalty = computePenaltyForce(m_contactStiffness, contact.depth, contact.direction);

        // Process entity B (penetrating body)
        bool shouldProcessB = contact.entityB >= 0;
        if (shouldProcessB)
        {
            DynamicEntity* entityB = world.entity(contact.entityB);
            bool entityBValid = (entityB != nullptr) && entityB->isMovable();
            
            if (entityBValid)
            {
                applyForceContribution(
                    forceOut,
                    contact.entityB,
                    entityB->kinematic().translation,
                    contact.location,
                    penalty,
                    static_cast<RealType>(1)
                );
            }
        }

        // Process entity A (obstacle body)
        bool shouldProcessA = contact.entityA >= 0;
        if (shouldProcessA)
        {
            DynamicEntity* entityA = world.entity(contact.entityA);
            bool entityAValid = (entityA != nullptr) && entityA->isMovable();
            
            if (entityAValid)
            {
                applyForceContribution(
                    forceOut,
                    contact.entityA,
                    entityA->kinematic().translation,
                    contact.location,
                    penalty,
                    static_cast<RealType>(-1)
                );
            }
        }
        
        ++idx;
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
    entries.reserve(contacts.size() * 18); // Estimate: 2 bodies × 9 entries each
    
    Matrix33 localK;
    
    auto processEntity = [&](IntType entityIdx) {
        if (entityIdx < 0)
            return;
            
        const IntType dofIdx = DofAccessor::linearOffset(entityIdx);
        insertBlockToTriplets(entries, dofIdx, dofIdx, localK);
    };
    
    // Process all contacts
    size_t contactIdx = contacts.size();
    while (contactIdx > 0)
    {
        --contactIdx;
        const ContactPoint& contact = contacts[contactIdx];
        
        buildLocalStiffnessBlock(contact.direction, m_contactStiffness, m_frictionStiffness, localK);
        
        processEntity(contact.entityB);
        processEntity(contact.entityA);
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
    const IntType entityCount = world.entityCount();
    const IntType dof = entityCount * 6;

    m_forceAssembler.configure(dof);

    // Resize all matrices/vectors in a single block
    m_massMatrix.resize(dof, dof);
    m_stiffnessMatrix.resize(dof, dof);
    m_systemMatrix.resize(dof, dof);
    m_forceVector.resize(dof);
    m_velocityDelta.resize(dof);
    m_rightHandSide.resize(dof);
}

namespace {

// Helper to update world bounds for all entities
void refreshAllWorldBounds(World& world)
{
    IntType idx = world.entityCount();
    while (idx > 0)
    {
        --idx;
        DynamicEntity* entity = world.entity(idx);
        if (entity != nullptr)
            static_cast<void>(entity->worldExtent());
    }
}

// Helper to build the implicit system matrix
void assembleSystemMatrix(
    const SparseGrid& massMatrix,
    const SparseGrid& stiffnessMatrix,
    RealType dt,
    SparseGrid& systemOut)
{
    const RealType dtSquared = dt * dt;
    SparseGrid scaledStiffness = stiffnessMatrix * dtSquared;
    systemOut = massMatrix - scaledStiffness;
}

} // anonymous namespace

void TimeIntegrator::advance(World& world) 
{
    const IntType entityCount = world.entityCount();
    const IntType dof = entityCount * 6;

    // Auto-initialize if needed
    if (m_massMatrix.rows() != dof)
        prepare(world);

    // Phase 1: Update bounds
    refreshAllWorldBounds(world);

    // Phase 2: Collision detection
    m_collisionSystem.enableDeviceAcceleration(true);
    m_collisionSystem.performDetection(world);

    std::cout << "[SIM_INFO] Contact count: "
              << m_collisionSystem.contactList().size() << std::endl;

    // Phase 3: Build mass matrix
    m_massMatrix.setZero();
    assembleMassMatrix(world, m_massMatrix);

    // Phase 4: Assemble forces
    const auto& contactList = m_collisionSystem.contactList();
    m_forceAssembler.buildForceVector(world, contactList, m_forceVector);

    // Phase 5: Assemble stiffness
    m_stiffnessMatrix.setZero();
    m_forceAssembler.buildStiffnessMatrix(world, contactList, m_stiffnessMatrix);

    // Phase 6: Build and solve linear system
    assembleSystemMatrix(m_massMatrix, m_stiffnessMatrix, m_deltaTime, m_systemMatrix);
    m_rightHandSide = m_forceVector * m_deltaTime;

    m_velocityDelta.setZero();
    const bool solveSuccess = solveLinearSystem(m_systemMatrix, m_rightHandSide, m_velocityDelta);
    
    // Phase 7: Update states
    if (solveSuccess)
        applyVelocityUpdates(world, m_velocityDelta);
}

void TimeIntegrator::assembleMassMatrix(World& world, SparseGrid& M) 
{
    SparseEntryList entries;
    const IntType entityCount = world.entityCount();
    entries.reserve(entityCount * 12); // Estimate: 6 diagonal + some inertia

    IntType i = 0;
    while (i < entityCount)
    {
        DynamicEntity* entity = world.entity(i);
        
        if (entity == nullptr)
        {
            ++i;
            continue;
        }

        const IntType dofIdx = DofAccessor::linearOffset(i);

        if (!entity->isMovable())
        {
            // Static body: large diagonal entries
            constexpr RealType kLargeMass = static_cast<RealType>(1e10);
            IntType k = 0;
            do {
                entries.emplace_back(dofIdx + k, dofIdx + k, kLargeMass);
                ++k;
            } while (k < 6);
        }
        else
        {
            // Dynamic body: mass and inertia
            const RealType m = entity->material().totalMass;
            
            // Linear DOFs
            entries.emplace_back(dofIdx + 0, dofIdx + 0, m);
            entries.emplace_back(dofIdx + 1, dofIdx + 1, m);
            entries.emplace_back(dofIdx + 2, dofIdx + 2, m);

            // Angular DOFs (inertia tensor)
            const Matrix33 Iinv = entity->worldInertiaInverse();
            const Matrix33 I = Iinv.inverse();
            
            const IntType angularBase = dofIdx + 3;
            insertBlockToTriplets(entries, angularBase, angularBase, I);
        }
        
        ++i;
    }

    M.setFromTriplets(entries.begin(), entries.end());
}

bool TimeIntegrator::solveLinearSystem(const SparseGrid& A, const VectorN& b, VectorN& x) 
{
    Eigen::ConjugateGradient<SparseGrid, Eigen::Lower | Eigen::Upper> solver;
    solver.setMaxIterations(m_iterLimit);
    solver.setTolerance(m_convergenceThresh);
    solver.compute(A);

    bool computeSuccess = (solver.info() == Eigen::Success);
    if (!computeSuccess)
        return false;

    x = solver.solve(b);
    return solver.info() == Eigen::Success;
}

void TimeIntegrator::applyVelocityUpdates(World& world, const VectorN& deltaV) 
{
    const IntType entityCount = world.entityCount();
    const RealType halfDt = m_deltaTime * static_cast<RealType>(0.5);

    IntType i = entityCount;
    while (i > 0)
    {
        --i;
        
        DynamicEntity* entity = world.entity(i);
        
        // Skip invalid or static entities
        bool shouldSkip = (entity == nullptr) || !entity->isMovable();
        if (shouldSkip)
            continue;

        // Extract velocity deltas
        const Point3 dvLin = DofAccessor::extractLinear(deltaV, i);
        const Point3 dvAng = DofAccessor::extractAngular(deltaV, i);

        EntityState& state = entity->kinematic();
        const MaterialProperties& material = entity->material();

        // Update velocities
        state.velocity.x() += dvLin.x();
        state.velocity.y() += dvLin.y();
        state.velocity.z() += dvLin.z();
        
        state.angularRate.x() += dvAng.x();
        state.angularRate.y() += dvAng.y();
        state.angularRate.z() += dvAng.z();

        // Apply damping using multiplicative factor
        const RealType linDampFactor = static_cast<RealType>(1) - material.linearDrag * m_deltaTime;
        const RealType angDampFactor = static_cast<RealType>(1) - material.angularDrag * m_deltaTime;
        
        state.velocity *= linDampFactor;
        state.angularRate *= angDampFactor;

        // Update position
        state.translation.x() += state.velocity.x() * m_deltaTime;
        state.translation.y() += state.velocity.y() * m_deltaTime;
        state.translation.z() += state.velocity.z() * m_deltaTime;

        // Update orientation using quaternion derivative
        // dq/dt = 0.5 * w * q
        const Point3& w = state.angularRate;
        const Rotation4& q = state.orientation;
        
        // Quaternion multiplication: qw * q where qw = (0, wx, wy, wz)
        Rotation4 dq;
        dq.w() = -w.x() * q.x() - w.y() * q.y() - w.z() * q.z();
        dq.x() =  w.x() * q.w() + w.y() * q.z() - w.z() * q.y();
        dq.y() = -w.x() * q.z() + w.y() * q.w() + w.z() * q.x();
        dq.z() =  w.x() * q.y() - w.y() * q.x() + w.z() * q.w();

        state.orientation.coeffs() += dq.coeffs() * halfDt;
        state.orientation.normalize();

        entity->assignKinematic(state);
        entity->resetAccumulators();
    }
}

}  // namespace phys3d
