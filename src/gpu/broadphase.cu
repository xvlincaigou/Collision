/**
 * @file broadphase.cu
 * @brief CUDA implementation of GPU broadphase detection.
 */
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <algorithm>
#include <cstdio>

#include "broadphase.cuh"

namespace rigid {
namespace gpu {

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void createEndpointsKernel(
    const float* aabbMins,
    const float* aabbMaxs,
    AABBEndpoint* endpoints,
    int n,
    int axis)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    endpoints[idx * 2].value = aabbMins[idx * 3 + axis];
    endpoints[idx * 2].bodyId = idx;
    endpoints[idx * 2].isMax = 0;

    endpoints[idx * 2 + 1].value = aabbMaxs[idx * 3 + axis];
    endpoints[idx * 2 + 1].bodyId = idx;
    endpoints[idx * 2 + 1].isMax = 1;
}

struct EndpointComparator {
    __host__ __device__ bool operator()(const AABBEndpoint& a,
                                        const AABBEndpoint& b) const {
        if (a.value != b.value) return a.value < b.value;
        return a.isMax < b.isMax;
    }
};

__global__ void sweepPruneKernel(
    const AABBEndpoint* sortedEndpoints,
    const float* aabbMins,
    const float* aabbMaxs,
    int numEndpoints,
    int numBodies,
    CollisionPair* pairs,
    int* pairCount,
    int maxPairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEndpoints) return;

    AABBEndpoint ep = sortedEndpoints[idx];
    if (ep.isMax) return;

    int bodyA = ep.bodyId;
    float minA[3], maxA[3];
    for (int k = 0; k < 3; ++k) {
        minA[k] = aabbMins[bodyA * 3 + k];
        maxA[k] = aabbMaxs[bodyA * 3 + k];
    }

    for (int j = idx + 1; j < numEndpoints; ++j) {
        AABBEndpoint other = sortedEndpoints[j];

        if (other.bodyId == bodyA && other.isMax) break;
        if (other.isMax) continue;
        if (other.bodyId == bodyA) continue;

        int bodyB = other.bodyId;

        float minB[3], maxB[3];
        for (int k = 0; k < 3; ++k) {
            minB[k] = aabbMins[bodyB * 3 + k];
            maxB[k] = aabbMaxs[bodyB * 3 + k];
        }

        bool overlap = true;
        for (int k = 0; k < 3; ++k) {
            if (minA[k] > maxB[k] || maxA[k] < minB[k]) {
                overlap = false;
                break;
            }
        }

        if (overlap) {
            int pairIdx = atomicAdd(pairCount, 1);
            if (pairIdx < maxPairs) {
                if (bodyA < bodyB) {
                    pairs[pairIdx].bodyA = bodyA;
                    pairs[pairIdx].bodyB = bodyB;
                } else {
                    pairs[pairIdx].bodyA = bodyB;
                    pairs[pairIdx].bodyB = bodyA;
                }
            }
        }
    }
}

// ============================================================================
// BroadphaseGPU Implementation
// ============================================================================

BroadphaseGPU::BroadphaseGPU() {
    cudaMalloc(&dPairCount_, sizeof(int));
}

BroadphaseGPU::~BroadphaseGPU() {
    free();
}

void BroadphaseGPU::free() {
    if (dEndpoints_) cudaFree(dEndpoints_);
    if (dAabbMins_) cudaFree(dAabbMins_);
    if (dAabbMaxs_) cudaFree(dAabbMaxs_);
    if (dPairs_) cudaFree(dPairs_);
    if (dPairCount_) cudaFree(dPairCount_);

    dEndpoints_ = nullptr;
    dAabbMins_ = nullptr;
    dAabbMaxs_ = nullptr;
    dPairs_ = nullptr;
    dPairCount_ = nullptr;
    capacity_ = 0;
}

void BroadphaseGPU::ensureCapacity(int numBodies) {
    if (numBodies <= capacity_) return;

    if (dEndpoints_) cudaFree(dEndpoints_);
    if (dAabbMins_) cudaFree(dAabbMins_);
    if (dAabbMaxs_) cudaFree(dAabbMaxs_);
    if (dPairs_) cudaFree(dPairs_);
    if (dPairCount_) cudaFree(dPairCount_);

    capacity_ = numBodies * 2;
    maxPairs_ = (capacity_ * capacity_) / 4;
    if (maxPairs_ > 1000000) maxPairs_ = 1000000;

    cudaMalloc(&dEndpoints_, capacity_ * 2 * sizeof(AABBEndpoint));
    cudaMalloc(&dAabbMins_, capacity_ * 3 * sizeof(float));
    cudaMalloc(&dAabbMaxs_, capacity_ * 3 * sizeof(float));
    cudaMalloc(&dPairs_, maxPairs_ * sizeof(CollisionPair));
    cudaMalloc(&dPairCount_, sizeof(int));

    cudaMemset(dPairCount_, 0, sizeof(int));
}

void BroadphaseGPU::detectPairs(const Vector<Vec3>& aabbMins,
                                 const Vector<Vec3>& aabbMaxs,
                                 Vector<CollisionPair>& outPairs) {
    int n = static_cast<int>(aabbMins.size());
    if (n < 2) {
        outPairs.clear();
        return;
    }

    ensureCapacity(n);

    // Copy AABBs to GPU
    Vector<float> hMins(n * 3), hMaxs(n * 3);
    for (int i = 0; i < n; ++i) {
        hMins[i * 3 + 0] = aabbMins[i].x();
        hMins[i * 3 + 1] = aabbMins[i].y();
        hMins[i * 3 + 2] = aabbMins[i].z();
        hMaxs[i * 3 + 0] = aabbMaxs[i].x();
        hMaxs[i * 3 + 1] = aabbMaxs[i].y();
        hMaxs[i * 3 + 2] = aabbMaxs[i].z();
    }
    cudaMemcpy(dAabbMins_, hMins.data(), n * 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dAabbMaxs_, hMaxs.data(), n * 3 * sizeof(float),
               cudaMemcpyHostToDevice);

    int axis = 0;  // Use X axis
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    createEndpointsKernel<<<numBlocks, blockSize>>>(
        dAabbMins_, dAabbMaxs_, dEndpoints_, n, axis);

    try {
        thrust::sort(thrust::device, dEndpoints_, dEndpoints_ + n * 2,
                     EndpointComparator());
    } catch (thrust::system_error& e) {
        printf("[FATAL] Thrust sort failed: %s\n", e.what());
        outPairs.clear();
        return;
    }

    int zero = 0;
    cudaMemcpy(dPairCount_, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int numEndpoints = n * 2;
    numBlocks = (numEndpoints + blockSize - 1) / blockSize;

    sweepPruneKernel<<<numBlocks, blockSize>>>(
        dEndpoints_, dAabbMins_, dAabbMaxs_, numEndpoints, n,
        dPairs_, dPairCount_, maxPairs_);

    cudaDeviceSynchronize();

    int pairCount;
    cudaMemcpy(&pairCount, dPairCount_, sizeof(int), cudaMemcpyDeviceToHost);
    pairCount = std::min(pairCount, maxPairs_);

    outPairs.resize(pairCount);
    if (pairCount > 0) {
        cudaMemcpy(outPairs.data(), dPairs_, pairCount * sizeof(CollisionPair),
                   cudaMemcpyDeviceToHost);
    }

    // Remove duplicates
    std::sort(outPairs.begin(), outPairs.end(),
              [](const CollisionPair& a, const CollisionPair& b) {
                  if (a.bodyA != b.bodyA) return a.bodyA < b.bodyA;
                  return a.bodyB < b.bodyB;
              });
    auto last = std::unique(outPairs.begin(), outPairs.end(),
                            [](const CollisionPair& a, const CollisionPair& b) {
                                return a.bodyA == b.bodyA && a.bodyB == b.bodyB;
                            });
    outPairs.erase(last, outPairs.end());
}

}  // namespace gpu
}  // namespace rigid
