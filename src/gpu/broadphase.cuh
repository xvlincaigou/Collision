/*
 * GPU-Accelerated Broad Phase Collision Detection
 */
#ifndef PHYS3D_DEVICE_BROADPHASE_CUH
#define PHYS3D_DEVICE_BROADPHASE_CUH

#include <cuda_runtime.h>
#include <vector>

#include "core/common.h"

namespace phys3d {
namespace gpu {

struct BoundsEndpoint 
{
    float coord;
    int entityId;
    int isUpperBound;
};

struct EntityPair 
{
    int entityA;
    int entityB;
};

class BroadPhaseDetector 
{
public:
    BroadPhaseDetector();
    ~BroadPhaseDetector();

    void findCandidatePairs(const DynArray<Point3>& boundsMin,
                            const DynArray<Point3>& boundsMax,
                            DynArray<EntityPair>& resultPairs);

    void release();

private:
    void ensureBufferCapacity(int entityCount);

    BoundsEndpoint* m_dEndpoints  = nullptr;
    int* m_dSortedIdx             = nullptr;
    int* m_dActiveSet             = nullptr;
    EntityPair* m_dPairs          = nullptr;
    int* m_dPairCounter           = nullptr;

    float* m_dBoundsMin           = nullptr;
    float* m_dBoundsMax           = nullptr;

    int m_bufferCapacity  = 0;
    int m_maxPairCount    = 0;
};

}  // namespace gpu
}  // namespace phys3d

namespace rigid {
namespace gpu {
    using BroadphaseGPU = phys3d::gpu::BroadPhaseDetector;
    using AABBEndpoint = phys3d::gpu::BoundsEndpoint;
    using CollisionPair = phys3d::gpu::EntityPair;
}
}

#endif // PHYS3D_DEVICE_BROADPHASE_CUH
