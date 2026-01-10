/**
 * @file collision_detector.h
 * @brief Collision detection system for rigid bodies.
 */
#pragma once

#include "contact.h"
#include "core/common.h"
#include "scene/scene.h"
#include "accel/bvh.h"

namespace rigid {

// Forward declarations
namespace gpu {
class CollisionDetectorGPU;
struct ContactDevice;
struct CollisionPair;
}

/**
 * @class CollisionDetector
 * @brief Detects collisions between bodies and environment.
 */
class CollisionDetector {
public:
    CollisionDetector();
    ~CollisionDetector();

    /// Clear all detected contacts
    void clear() { contacts_.clear(); }

    /// Detect all collisions in the scene
    void detectCollisions(Scene& scene);

    /// Get the detected contacts
    [[nodiscard]] const Vector<Contact>& contacts() const { return contacts_; }

    /// Enable/disable GPU acceleration
    void setUseGPU(bool use) { useGPU_ = use; }
    [[nodiscard]] bool useGPU() const { return useGPU_; }

private:
    // CPU collision detection
    void detectBodyEnvironment(Int bodyIdx, RigidBody& body, Environment& env);
    void detectBodyBody(Int idxA, RigidBody& bodyA, Int idxB, RigidBody& bodyB);
    void detectVertexMesh(Int dynamicIdx, RigidBody& dynamic,
                          Int staticIdx, RigidBody& statik);

    // BVH traversal helpers
    void traverseBVHPlane(const BVHNode& node, const BVH& bvh,
                          const Plane& planeLocal,
                          const Vector<Vec3>& vertices,
                          const Vector<Triangle>& triangles,
                          const Mat3& bodyRot, const Vec3& bodyPos,
                          Int bodyIdx, const Vec3& planeWorldNormal,
                          Vector<bool>& visitedVerts);

    void traverseBVHPoint(const BVHNode& node, const BVH& bvh,
                          const Vec3& pointLocal,
                          const Vector<Vec3>& vertices,
                          const Vector<Triangle>& triangles,
                          Float& minDistSq, Vec3& closestNormal,
                          Vec3& closestPos, bool& found);

    // GPU collision detection
    void detectCollisionsGPU(Scene& scene);
    void convertGPUContacts(const Vector<gpu::ContactDevice>& gpuContacts);
    void broadphaseGPU(Scene& scene, Vector<std::pair<Int, Int>>& pairs);

    Vector<Contact> contacts_;
    bool useGPU_ = true;

    UniquePtr<gpu::CollisionDetectorGPU> gpuDetector_;
};

// ============================================================================
// Triangle collision helper (used by both CPU and GPU paths)
// ============================================================================

/**
 * @brief Check if a point penetrates a triangle.
 * @return True if collision found and closer than current minimum.
 */
inline bool checkTriangleCollision(
    const Vec3& pointLocal,
    const Vec3& v0, const Vec3& v1, const Vec3& v2,
    Float& minDistSq,
    Vec3& outNormal,
    Vec3& outPoint)
{
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 normal = edge1.cross(edge2).normalized();

    Float planeDist = (pointLocal - v0).dot(normal);
    constexpr Float kPenetrationThreshold = -0.5f;

    if (planeDist < 0.0f && planeDist > kPenetrationThreshold) {
        Float distSq = planeDist * planeDist;
        if (distSq < minDistSq) {
            // Project point onto triangle plane
            Vec3 proj = pointLocal - normal * planeDist;
            Vec3 v2Proj = proj - v0;

            // Barycentric coordinates
            Float d00 = edge1.dot(edge1);
            Float d01 = edge1.dot(edge2);
            Float d11 = edge2.dot(edge2);
            Float d20 = v2Proj.dot(edge1);
            Float d21 = v2Proj.dot(edge2);
            Float denom = d00 * d11 - d01 * d01;

            if (std::abs(denom) > 1e-8f) {
                Float invDenom = 1.0f / denom;
                Float v = (d11 * d20 - d01 * d21) * invDenom;
                Float w = (d00 * d21 - d01 * d20) * invDenom;
                Float u = 1.0f - v - w;

                if (u >= 0.0f && v >= 0.0f && w >= 0.0f) {
                    minDistSq = distSq;
                    outNormal = normal;
                    outPoint = proj;
                    return true;
                }
            }
        }
    }
    return false;
}

}  // namespace rigid
