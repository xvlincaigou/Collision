/**
 * @file broadphase.cuh
 * @brief GPU-accelerated broadphase collision detection.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "core/common.h"

namespace rigid {
namespace gpu {

/// AABB endpoint for sweep-and-prune
struct AABBEndpoint {
    float value;
    int bodyId;
    int isMax;  // 0 = min, 1 = max
};

/// Collision pair result
struct CollisionPair {
    int bodyA;
    int bodyB;
};

/**
 * @class BroadphaseGPU
 * @brief Sweep-and-prune broadphase using CUDA.
 */
class BroadphaseGPU {
public:
    BroadphaseGPU();
    ~BroadphaseGPU();

    /// Detect potentially colliding pairs
    void detectPairs(const Vector<Vec3>& aabbMins,
                     const Vector<Vec3>& aabbMaxs,
                     Vector<CollisionPair>& outPairs);

    /// Free GPU resources
    void free();

private:
    void ensureCapacity(int numBodies);

    AABBEndpoint* dEndpoints_ = nullptr;
    int* dSortedIndices_      = nullptr;
    int* dActiveList_         = nullptr;
    CollisionPair* dPairs_    = nullptr;
    int* dPairCount_          = nullptr;

    float* dAabbMins_         = nullptr;
    float* dAabbMaxs_         = nullptr;

    int capacity_  = 0;
    int maxPairs_  = 0;
};

}  // namespace gpu
}  // namespace rigid
