/*
 * GPU-Accelerated Narrow Phase Collision Detection
 */
#ifndef PHYS3D_DEVICE_COLLISION_DETECTOR_CUH
#define PHYS3D_DEVICE_COLLISION_DETECTOR_CUH

#include <cuda_runtime.h>

#include "bvh_builder.cuh"
#include "broadphase.cuh"
#include "core/common.h"

namespace phys3d {
namespace gpu {

struct DeviceContact 
{
    IntType entityIdxA;
    IntType entityIdxB;
    CudaFloat3 worldPoint;
    CudaFloat3 worldNormal;
    RealType penetration;
};

struct DevicePlane 
{
    CudaFloat3 direction;
    RealType offset;
};

struct EntityTransform 
{
    CudaFloat3 translation;
    CudaFloat4 rotation;
};

class NarrowPhaseDetector 
{
public:
    NarrowPhaseDetector();
    ~NarrowPhaseDetector();

    void assignTree(DeviceTreeBuilder* treeBuilder);

    void detectEntityEnvironment(IntType entityIdx,
                                  const EntityTransform& transform,
                                  const DevicePlane* planeArray,
                                  IntType planeCount,
                                  DynArray<DeviceContact>& contactResults);

    void detectEntityEntity(IntType entityAIdx,
                             const EntityTransform& transformA,
                             DeviceTreeBuilder* treeA,
                             IntType entityBIdx,
                             const EntityTransform& transformB,
                             DeviceTreeBuilder* treeB,
                             DynArray<DeviceContact>& contactResults);

    void performBroadPhase(const DynArray<Point3>& boundsMin,
                            const DynArray<Point3>& boundsMax,
                            DynArray<EntityPair>& candidatePairs);

    void release();

private:
    DeviceContact* m_dContactBuffer = nullptr;
    IntType* m_dContactCounter      = nullptr;
    IntType m_contactLimit          = 65536;

    DevicePlane* m_dPlaneBuffer     = nullptr;
    IntType m_planeBufferSize       = 0;

    DeviceTreeBuilder* m_activeTree = nullptr;
    SolePtr<BroadPhaseDetector> m_broadPhase;
};

/* ========== Device Helper Functions ========== */

__device__ inline CudaFloat3 operator+(CudaFloat3 a, CudaFloat3 b) 
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline CudaFloat3 operator-(CudaFloat3 a, CudaFloat3 b) 
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline CudaFloat3 operator*(CudaFloat3 a, RealType s) 
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline RealType innerProduct(CudaFloat3 a, CudaFloat3 b) 
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline CudaFloat3 crossProduct(CudaFloat3 a, CudaFloat3 b) 
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ inline CudaFloat3 applyQuaternion(CudaFloat3 v, CudaFloat4 q) 
{
    CudaFloat3 qVec = make_float3(q.x, q.y, q.z);
    RealType qScalar = q.w;
    return qVec * (static_cast<RealType>(2) * innerProduct(qVec, v)) + 
           v * (qScalar * qScalar - innerProduct(qVec, qVec)) + 
           crossProduct(qVec, v) * (static_cast<RealType>(2) * qScalar);
}

__device__ inline bool boundsPlaneOverlap(const DeviceBounds3D& box,
                                           CudaFloat3 normal, RealType offset) 
{
    CudaFloat3 center = (box.lo + box.hi) * static_cast<RealType>(0.5);
    CudaFloat3 halfDim = (box.hi - box.lo) * static_cast<RealType>(0.5);
    RealType radius = halfDim.x * fabsf(normal.x) +
                      halfDim.y * fabsf(normal.y) +
                      halfDim.z * fabsf(normal.z);
    RealType signedDist = innerProduct(normal, center) - offset;
    return signedDist <= radius;
}

}  // namespace gpu
}  // namespace phys3d

namespace rigid {
namespace gpu {
    using CollisionDetectorGPU = phys3d::gpu::NarrowPhaseDetector;
    using ContactDevice = phys3d::gpu::DeviceContact;
    using PlaneDevice = phys3d::gpu::DevicePlane;
    using BodyTransformDevice = phys3d::gpu::EntityTransform;
    
    __device__ inline phys3d::RealType dot(phys3d::CudaFloat3 a, phys3d::CudaFloat3 b) {
        return phys3d::gpu::innerProduct(a, b);
    }
    __device__ inline phys3d::CudaFloat3 cross(phys3d::CudaFloat3 a, phys3d::CudaFloat3 b) {
        return phys3d::gpu::crossProduct(a, b);
    }
    __device__ inline phys3d::CudaFloat3 rotateByQuat(phys3d::CudaFloat3 v, phys3d::CudaFloat4 q) {
        return phys3d::gpu::applyQuaternion(v, q);
    }
    __device__ inline bool aabbIntersectsPlane(const phys3d::gpu::DeviceBounds3D& b,
                                                phys3d::CudaFloat3 n, phys3d::RealType o) {
        return phys3d::gpu::boundsPlaneOverlap(b, n, o);
    }
}
}

#endif // PHYS3D_DEVICE_COLLISION_DETECTOR_CUH
