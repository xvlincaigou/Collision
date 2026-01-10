/**
 * @file integrator.cpp
 * @brief Implementation of ForceBuilder and Integrator classes.
 */
#include "integrator.h"

#include <iostream>

namespace rigid {

// ============================================================================
// ForceBuilder Implementation
// ============================================================================

ForceBuilder::ForceBuilder()
    : gravity_(0.0f, 0.0f, -9.81f)
    , penaltyStiffness_(100.0f)
    , frictionStiffness_(0.1f)
    , dofCount_(0)
{}

void ForceBuilder::setup(Int dofCount) {
    dofCount_ = dofCount;
}

void ForceBuilder::assembleForces(Scene& scene, const Vector<Contact>& contacts,
                                  VecX& force) {
    force.setZero();
    addGravityForce(scene, force);
    addCollisionForce(scene, contacts, force);
}

void ForceBuilder::addGravityForce(Scene& scene, VecX& force) {
    Int nBodies = scene.bodyCount();
    for (Int i = 0; i < nBodies; ++i) {
        RigidBody* body = scene.body(i);
        if (!body || !body->isDynamic()) continue;

        Int idx = globalDofIndex(i);
        Float mass = body->properties().mass;
        force.segment<3>(idx) += mass * gravity_;
    }
}

void ForceBuilder::addCollisionForce(Scene& scene, const Vector<Contact>& contacts,
                                     VecX& force) {
    for (const auto& contact : contacts) {
        // Penalty force: k * d^2 * n
        Vec3 penalty = penaltyStiffness_ * contact.depth * contact.depth * contact.normal;

        // Apply to body B (penetrator) - pushed out
        if (contact.bodyIndexB >= 0) {
            RigidBody* bodyB = scene.body(contact.bodyIndexB);
            if (bodyB && bodyB->isDynamic()) {
                Int idxB = globalDofIndex(contact.bodyIndexB);
                force.segment<3>(idxB) += penalty;

                // Torque from contact
                Vec3 rB = contact.position - bodyB->state().position;
                force.segment<3>(idxB + 3) += rB.cross(penalty);
            }
        }

        // Apply to body A (obstacle) - pushed in (opposite direction)
        if (contact.bodyIndexA >= 0) {
            RigidBody* bodyA = scene.body(contact.bodyIndexA);
            if (bodyA && bodyA->isDynamic()) {
                Int idxA = globalDofIndex(contact.bodyIndexA);
                force.segment<3>(idxA) -= penalty;

                Vec3 rA = contact.position - bodyA->state().position;
                force.segment<3>(idxA + 3) += rA.cross(-penalty);
            }
        }
    }
}

void ForceBuilder::assembleJacobian(Scene& /* scene */,
                                    const Vector<Contact>& contacts,
                                    SparseMat& jacobian) {
    addCollisionJacobian(contacts, jacobian);
}

void ForceBuilder::addCollisionJacobian(const Vector<Contact>& contacts,
                                        SparseMat& jacobian) {
    Triplets triplets;

    for (const auto& contact : contacts) {
        // Linearized stiffness: K = -k * (n ⊗ n) - k_f * (I - n ⊗ n)
        Mat3 Pn = contact.normal * contact.normal.transpose();
        Mat3 Pt = Mat3::Identity() - Pn;
        Mat3 Klinear = -penaltyStiffness_ * Pn - frictionStiffness_ * Pt;

        // Add to body B
        if (contact.bodyIndexB >= 0) {
            Int idxB = globalDofIndex(contact.bodyIndexB);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    triplets.emplace_back(idxB + r, idxB + c, Klinear(r, c));
                }
            }
        }

        // Add to body A
        if (contact.bodyIndexA >= 0) {
            Int idxA = globalDofIndex(contact.bodyIndexA);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    triplets.emplace_back(idxA + r, idxA + c, Klinear(r, c));
                }
            }
        }
    }

    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

// ============================================================================
// Integrator Implementation
// ============================================================================

Integrator::Integrator()
    : maxIterations_(20)
    , tolerance_(1e-2f)
    , timeStep_(1e-2f)
{}

void Integrator::initialize(Scene& scene) {
    Int nBodies = scene.bodyCount();
    Int dof = nBodies * 6;

    forceBuilder_.setup(dof);

    massMatrix_.resize(dof, dof);
    jacobian_.resize(dof, dof);
    systemMatrix_.resize(dof, dof);
    forceVector_.resize(dof);
    deltaV_.resize(dof);
    rhs_.resize(dof);
}

void Integrator::step(Scene& scene) {
    Int nBodies = scene.bodyCount();
    Int dof = nBodies * 6;

    if (massMatrix_.rows() != dof) {
        initialize(scene);
    }

    // Update world bounds for all bodies
    for (Int i = 0; i < nBodies; ++i) {
        if (auto* body = scene.body(i)) {
            (void)body->worldBounds();  // Trigger lazy update
        }
    }

    // Collision detection
    collisionDetector_.setUseGPU(true);
    collisionDetector_.detectCollisions(scene);

    std::cout << "[SIM_INFO] Contact count: "
              << collisionDetector_.contacts().size() << std::endl;

    // Build mass matrix
    massMatrix_.setZero();
    buildMassMatrix(scene, massMatrix_);

    // Assemble forces and Jacobian
    forceBuilder_.assembleForces(scene, collisionDetector_.contacts(), forceVector_);

    jacobian_.setZero();
    forceBuilder_.assembleJacobian(scene, collisionDetector_.contacts(), jacobian_);

    // Implicit Euler: (M - h² K) Δv = h f
    SparseMat stiffnessTerm = jacobian_ * (timeStep_ * timeStep_);
    systemMatrix_ = massMatrix_ - stiffnessTerm;
    rhs_ = forceVector_ * timeStep_;

    // Solve linear system
    deltaV_.setZero();
    if (solve(systemMatrix_, rhs_, deltaV_)) {
        updateBodyStates(scene, deltaV_);
    }
}

void Integrator::buildMassMatrix(Scene& scene, SparseMat& M) {
    Triplets triplets;
    Int nBodies = scene.bodyCount();

    for (Int i = 0; i < nBodies; ++i) {
        RigidBody* body = scene.body(i);
        if (!body) continue;

        Int idx = globalDofIndex(i);

        if (!body->isDynamic()) {
            // Static body: large diagonal to fix in place
            for (int k = 0; k < 6; ++k) {
                triplets.emplace_back(idx + k, idx + k, 1e10f);
            }
        } else {
            // Dynamic body: mass and inertia
            Float m = body->properties().mass;
            for (int k = 0; k < 3; ++k) {
                triplets.emplace_back(idx + k, idx + k, m);
            }

            Mat3 Iinv = body->worldInertiaInv();
            Mat3 I = Iinv.inverse();

            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    triplets.emplace_back(idx + 3 + r, idx + 3 + c, I(r, c));
                }
            }
        }
    }

    M.setFromTriplets(triplets.begin(), triplets.end());
}

bool Integrator::solve(const SparseMat& A, const VecX& b, VecX& x) {
    Eigen::ConjugateGradient<SparseMat, Eigen::Lower | Eigen::Upper> solver;
    solver.setMaxIterations(maxIterations_);
    solver.setTolerance(tolerance_);
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        return false;
    }

    x = solver.solve(b);
    return solver.info() == Eigen::Success;
}

void Integrator::updateBodyStates(Scene& scene, const VecX& deltaV) {
    Int nBodies = scene.bodyCount();

    for (Int i = 0; i < nBodies; ++i) {
        RigidBody* body = scene.body(i);
        if (!body || !body->isDynamic()) continue;

        Int idx = globalDofIndex(i);
        Vec3 dvLin = deltaV.segment<3>(idx);
        Vec3 dvAng = deltaV.segment<3>(idx + 3);

        BodyState& state = body->state();
        const auto& props = body->properties();

        // Update velocities
        state.linearVel += dvLin;
        state.angularVel += dvAng;

        // Apply damping
        state.linearVel *= (1.0f - props.linearDamping * timeStep_);
        state.angularVel *= (1.0f - props.angularDamping * timeStep_);

        // Update position
        state.position += state.linearVel * timeStep_;

        // Update orientation using quaternion derivative
        Vec3 w = state.angularVel;
        Quat qw(0, w.x(), w.y(), w.z());
        Quat dq = qw * state.orientation;

        state.orientation.coeffs() += dq.coeffs() * (0.5f * timeStep_);
        state.orientation.normalize();

        body->setState(state);
        body->clearAccumulators();
    }
}

}  // namespace rigid
