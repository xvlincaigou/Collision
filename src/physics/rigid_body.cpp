/**
 * @file rigid_body.cpp
 * @brief Implementation of the RigidBody class.
 */
#include "rigid_body.h"
#include "core/mesh_cache.h"

namespace rigid {

RigidBody::RigidBody(String name) : name_(std::move(name)) {}

bool RigidBody::loadMesh(const String& path, bool buildBVH) {
    auto meshPtr = MeshCache::instance().acquire(path, buildBVH);
    if (!meshPtr) {
        return false;
    }

    mesh_ = std::move(meshPtr);
    boundsDirty_ = true;
    inertiaDirty_ = true;
    return true;
}

void RigidBody::setMesh(SharedPtr<Mesh> mesh) {
    mesh_ = std::move(mesh);
    boundsDirty_ = true;
    inertiaDirty_ = true;
}

void RigidBody::setProperties(const BodyProperties& props) {
    properties_ = props;
    inertiaDirty_ = true;
}

void RigidBody::setState(const BodyState& newState) {
    state_ = newState;
    boundsDirty_ = true;
    inertiaDirty_ = true;
}

void RigidBody::applyForce(const Vec3& force, const Vec3& worldPoint) {
    forceAccum_ += force;
    Vec3 r = worldPoint - state_.position;
    torqueAccum_ += r.cross(force);
}

void RigidBody::applyTorque(const Vec3& torque) {
    torqueAccum_ += torque;
}

void RigidBody::clearAccumulators() {
    forceAccum_.setZero();
    torqueAccum_.setZero();
}

const Mat3& RigidBody::worldInertiaInv() {
    if (inertiaDirty_) {
        syncWorldInertia();
    }
    return worldInertiaInv_;
}

AABB& RigidBody::worldBounds() {
    if (boundsDirty_) {
        updateWorldBounds();
    }
    return worldBounds_;
}

void RigidBody::updateWorldBounds() {
    worldBounds_.reset();

    if (!mesh_) {
        boundsDirty_ = false;
        return;
    }

    const auto& verts = mesh_->vertices();
    if (verts.empty()) {
        boundsDirty_ = false;
        return;
    }

    Mat3 rot = state_.rotationMatrix();
    Float scale = state_.scale;
    for (const Vec3& vLocal : verts) {
        Vec3 vScaled = (vLocal - properties_.centerOfMass) * scale;
        Vec3 vWorld = rot * vScaled + state_.position;
        worldBounds_.expand(vWorld);
    }

    boundsDirty_ = false;
}

void RigidBody::syncWorldInertia() {
    Mat3 rot = state_.rotationMatrix();
    // Inertia scales as scale^2 (I = m * r^2, r scales with scale)
    Float scaleSquared = state_.scale * state_.scale;
    Float invScaleSquared = (scaleSquared > 0.0f) ? (1.0f / scaleSquared) : 1.0f;
    worldInertiaInv_ = rot * (properties_.inertiaBodyInv * invScaleSquared) * rot.transpose();
    inertiaDirty_ = false;
}

}  // namespace rigid
