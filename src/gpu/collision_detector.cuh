/**
 * @file collision_detector.cuh
 * @brief GPU-accelerated collision detection.
 */
#pragma once

#include <cuda_runtime.h>

#include "bvh_builder.cuh"
#include "broadphase.cuh"
#include "core/common.h"

namespace rigid {
namespace gpu {

/// Device-side contact
struct ContactDevice {
    Int bodyIndexA;
    Int bodyIndexB;
    Float3 position;
    Float3 normal;
    Float depth;
};

/// Device-side plane
struct PlaneDevice {
    Float3 normal;
    Float offset;
};

/// Device-side body transform
struct BodyTransformDevice {
    Float3 position;
    Float4 orientation;  // Quaternion (x, y, z, w)
};

/**
 * @class CollisionDetectorGPU
 * @brief CUDA-based collision detection.
 */
class CollisionDetectorGPU {
public:
    CollisionDetectorGPU();
    ~CollisionDetectorGPU();

    void setBVH(BVHBuilderGPU* bvhBuilder);

    void detectBodyEnvironment(Int bodyIdx,
                                const BodyTransformDevice& transform,
                                const PlaneDevice* planes,
                                Int numPlanes,
                                Vector<ContactDevice>& outContacts);

    void detectBodyBody(Int bodyAIdx,
                        const BodyTransformDevice& transformA,
                        BVHBuilderGPU* bvhA,
                        Int bodyBIdx,
                        const BodyTransformDevice& transformB,
                        BVHBuilderGPU* bvhB,
                        Vector<ContactDevice>& outContacts);

    void broadphaseDetect(const Vector<Vec3>& aabbMins,
                          const Vector<Vec3>& aabbMaxs,
                          Vector<CollisionPair>& outPairs);

    void free();

private:
    ContactDevice* dContacts_ = nullptr;
    Int* dContactCount_       = nullptr;
    Int maxContacts_          = 65536;

    PlaneDevice* dPlanes_     = nullptr;
    Int numPlanes_            = 0;

    BVHBuilderGPU* bvh_       = nullptr;
    UniquePtr<BroadphaseGPU> broadphase_;
};

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ inline Float3 operator+(Float3 a, Float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline Float3 operator-(Float3 a, Float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline Float3 operator*(Float3 a, Float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline Float dot(Float3 a, Float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline Float3 cross(Float3 a, Float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ inline Float3 rotateByQuat(Float3 v, Float4 q) {
    Float3 u = make_float3(q.x, q.y, q.z);
    Float s = q.w;
    return u * (2.0f * dot(u, v)) + v * (s * s - dot(u, u)) + cross(u, v) * (2.0f * s);
}

__device__ inline bool aabbIntersectsPlane(const AABBDevice& box,
                                            Float3 normal, Float offset) {
    Float3 center = (box.min + box.max) * 0.5f;
    Float3 extents = (box.max - box.min) * 0.5f;
    Float r = extents.x * fabsf(normal.x) +
              extents.y * fabsf(normal.y) +
              extents.z * fabsf(normal.z);
    Float s = dot(normal, center) - offset;
    return s <= r;
}

}  // namespace gpu
}  // namespace rigid
