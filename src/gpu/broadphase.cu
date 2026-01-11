/*
 * Implementation: CUDA Broad Phase Detection
 */
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <algorithm>
#include <cstdio>

#include "broadphase.cuh"

namespace phys3d {
namespace gpu {

/* ========== Kernel Implementations ========== */

__global__ void kernelCreateEndpoints(
    const float* boundsMinData,
    const float* boundsMaxData,
    BoundsEndpoint* endpoints,
    int entityTotal,
    int axisIdx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= entityTotal) return;

    endpoints[tid * 2].coord = boundsMinData[tid * 3 + axisIdx];
    endpoints[tid * 2].entityId = tid;
    endpoints[tid * 2].isUpperBound = 0;

    endpoints[tid * 2 + 1].coord = boundsMaxData[tid * 3 + axisIdx];
    endpoints[tid * 2 + 1].entityId = tid;
    endpoints[tid * 2 + 1].isUpperBound = 1;
}

struct EndpointOrdering 
{
    __host__ __device__ bool operator()(const BoundsEndpoint& lhs, const BoundsEndpoint& rhs) const 
    {
        if (lhs.coord != rhs.coord) return lhs.coord < rhs.coord;
        return lhs.isUpperBound < rhs.isUpperBound;
    }
};

__global__ void kernelSweepAndPrune(
    const BoundsEndpoint* sortedEndpoints,
    const float* boundsMinData,
    const float* boundsMaxData,
    int endpointTotal,
    int entityTotal,
    EntityPair* pairResults,
    int* pairCounter,
    int maxPairLimit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= endpointTotal) return;

    BoundsEndpoint ep = sortedEndpoints[tid];
    if (ep.isUpperBound) return;

    int entityA = ep.entityId;
    float minA[3], maxA[3];
    for (int k = 0; k < 3; ++k) 
    {
        minA[k] = boundsMinData[entityA * 3 + k];
        maxA[k] = boundsMaxData[entityA * 3 + k];
    }

    for (int j = tid + 1; j < endpointTotal; ++j) 
    {
        BoundsEndpoint other = sortedEndpoints[j];

        if (other.entityId == entityA && other.isUpperBound) break;
        if (other.isUpperBound) continue;
        if (other.entityId == entityA) continue;

        int entityB = other.entityId;

        float minB[3], maxB[3];
        for (int k = 0; k < 3; ++k) 
        {
            minB[k] = boundsMinData[entityB * 3 + k];
            maxB[k] = boundsMaxData[entityB * 3 + k];
        }

        bool overlapping = true;
        for (int k = 0; k < 3; ++k) 
        {
            if (minA[k] > maxB[k] || maxA[k] < minB[k]) 
            {
                overlapping = false;
                break;
            }
        }

        if (overlapping) 
        {
            int idx = atomicAdd(pairCounter, 1);
            if (idx < maxPairLimit) 
            {
                if (entityA < entityB) 
                {
                    pairResults[idx].entityA = entityA;
                    pairResults[idx].entityB = entityB;
                } 
                else 
                {
                    pairResults[idx].entityA = entityB;
                    pairResults[idx].entityB = entityA;
                }
            }
        }
    }
}

/* ========== BroadPhaseDetector Implementation ========== */

BroadPhaseDetector::BroadPhaseDetector() 
{
    cudaMalloc(&m_dPairCounter, sizeof(int));
}

BroadPhaseDetector::~BroadPhaseDetector() 
{
    release();
}

void BroadPhaseDetector::release() 
{
    if (m_dEndpoints) cudaFree(m_dEndpoints);
    if (m_dBoundsMin) cudaFree(m_dBoundsMin);
    if (m_dBoundsMax) cudaFree(m_dBoundsMax);
    if (m_dPairs) cudaFree(m_dPairs);
    if (m_dPairCounter) cudaFree(m_dPairCounter);

    m_dEndpoints = nullptr;
    m_dBoundsMin = nullptr;
    m_dBoundsMax = nullptr;
    m_dPairs = nullptr;
    m_dPairCounter = nullptr;
    m_bufferCapacity = 0;
}

void BroadPhaseDetector::ensureBufferCapacity(int entityCount) 
{
    if (entityCount <= m_bufferCapacity) return;

    if (m_dEndpoints) cudaFree(m_dEndpoints);
    if (m_dBoundsMin) cudaFree(m_dBoundsMin);
    if (m_dBoundsMax) cudaFree(m_dBoundsMax);
    if (m_dPairs) cudaFree(m_dPairs);
    if (m_dPairCounter) cudaFree(m_dPairCounter);

    m_bufferCapacity = entityCount * 2;
    m_maxPairCount = (m_bufferCapacity * m_bufferCapacity) / 4;
    if (m_maxPairCount > 1000000) m_maxPairCount = 1000000;

    cudaMalloc(&m_dEndpoints, m_bufferCapacity * 2 * sizeof(BoundsEndpoint));
    cudaMalloc(&m_dBoundsMin, m_bufferCapacity * 3 * sizeof(float));
    cudaMalloc(&m_dBoundsMax, m_bufferCapacity * 3 * sizeof(float));
    cudaMalloc(&m_dPairs, m_maxPairCount * sizeof(EntityPair));
    cudaMalloc(&m_dPairCounter, sizeof(int));

    cudaMemset(m_dPairCounter, 0, sizeof(int));
}

void BroadPhaseDetector::findCandidatePairs(const DynArray<Point3>& boundsMin,
                                             const DynArray<Point3>& boundsMax,
                                             DynArray<EntityPair>& resultPairs) 
{
    int entityCount = static_cast<int>(boundsMin.size());
    if (entityCount < 2) 
    {
        resultPairs.clear();
        return;
    }

    ensureBufferCapacity(entityCount);

    DynArray<float> hostMin(entityCount * 3), hostMax(entityCount * 3);
    for (int i = 0; i < entityCount; ++i) 
    {
        hostMin[i * 3 + 0] = boundsMin[i].x();
        hostMin[i * 3 + 1] = boundsMin[i].y();
        hostMin[i * 3 + 2] = boundsMin[i].z();
        hostMax[i * 3 + 0] = boundsMax[i].x();
        hostMax[i * 3 + 1] = boundsMax[i].y();
        hostMax[i * 3 + 2] = boundsMax[i].z();
    }
    cudaMemcpy(m_dBoundsMin, hostMin.data(), entityCount * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dBoundsMax, hostMax.data(), entityCount * 3 * sizeof(float), cudaMemcpyHostToDevice);

    int primaryAxis = 0;
    int threadsPerBlock = 256;
    int numBlocks = (entityCount + threadsPerBlock - 1) / threadsPerBlock;

    kernelCreateEndpoints<<<numBlocks, threadsPerBlock>>>(
        m_dBoundsMin, m_dBoundsMax, m_dEndpoints, entityCount, primaryAxis);

    try 
    {
        thrust::sort(thrust::device, m_dEndpoints, m_dEndpoints + entityCount * 2, EndpointOrdering());
    } 
    catch (thrust::system_error& e) 
    {
        printf("[ERROR] Thrust sort failed: %s\n", e.what());
        resultPairs.clear();
        return;
    }

    int zero = 0;
    cudaMemcpy(m_dPairCounter, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int endpointTotal = entityCount * 2;
    numBlocks = (endpointTotal + threadsPerBlock - 1) / threadsPerBlock;

    kernelSweepAndPrune<<<numBlocks, threadsPerBlock>>>(
        m_dEndpoints, m_dBoundsMin, m_dBoundsMax, endpointTotal, entityCount,
        m_dPairs, m_dPairCounter, m_maxPairCount);

    cudaDeviceSynchronize();

    int pairCount;
    cudaMemcpy(&pairCount, m_dPairCounter, sizeof(int), cudaMemcpyDeviceToHost);
    pairCount = std::min(pairCount, m_maxPairCount);

    resultPairs.resize(pairCount);
    if (pairCount > 0) 
    {
        cudaMemcpy(resultPairs.data(), m_dPairs, pairCount * sizeof(EntityPair), cudaMemcpyDeviceToHost);
    }

    std::sort(resultPairs.begin(), resultPairs.end(),
              [](const EntityPair& a, const EntityPair& b) {
                  if (a.entityA != b.entityA) return a.entityA < b.entityA;
                  return a.entityB < b.entityB;
              });
    auto lastUnique = std::unique(resultPairs.begin(), resultPairs.end(),
                            [](const EntityPair& a, const EntityPair& b) {
                                return a.entityA == b.entityA && a.entityB == b.entityB;
                            });
    resultPairs.erase(lastUnique, resultPairs.end());
}

}  // namespace gpu
}  // namespace phys3d
