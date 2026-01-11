/*
 * Implementation: CUDA Narrow Phase Detection
 */
#include <cstdio>
#include <algorithm>

#include "collision_detector.cuh"

namespace phys3d {
namespace gpu {

/* ========== Kernel Implementations ========== */

__global__ void kernelDetectPlaneCollision(
    const CudaFloat3* pointData,
    const DeviceTreeNode* nodeData,
    const IntType* faceOrderData,
    const CudaInt3* faceData,
    IntType faceTotal,
    CudaFloat3 entityPos,
    CudaFloat4 entityRot,
    DevicePlane worldPlane,
    IntType entityIdx,
    DeviceContact* contactBuffer,
    IntType* contactCounter,
    IntType contactLimit,
    IntType* processedPoints,
    IntType pointTotal)
{
    IntType leafId = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafId >= faceTotal) return;

    IntType nodeIdx = faceTotal - 1 + leafId;
    DeviceTreeNode node = nodeData[nodeIdx];

    CudaFloat4 rotInv = make_float4(-entityRot.x, -entityRot.y,
                                     -entityRot.z, entityRot.w);
    CudaFloat3 localNormal = applyQuaternion(worldPlane.direction, rotInv);
    RealType localOffset = worldPlane.offset - innerProduct(worldPlane.direction, entityPos);

    if (!boundsPlaneOverlap(node.volume, localNormal, localOffset))
        return;

    IntType primIdx = faceOrderData[leafId];
    CudaInt3 face = faceData[primIdx];
    IntType vertIds[3] = {face.x, face.y, face.z};

    for (IntType k = 0; k < 3; ++k) 
    {
        IntType vid = vertIds[k];
        IntType alreadyProcessed = atomicExch(&processedPoints[vid], 1);
        if (alreadyProcessed) continue;

        CudaFloat3 localPt = pointData[vid];
        RealType signedDist = innerProduct(localPt, localNormal) - localOffset;

        if (signedDist < static_cast<RealType>(0)) 
        {
            CudaFloat3 worldPt = applyQuaternion(localPt, entityRot) + entityPos;
            IntType idx = atomicAdd(contactCounter, 1);
            if (idx < contactLimit) 
            {
                DeviceContact contact;
                contact.entityIdxA = -1;
                contact.entityIdxB = entityIdx;
                contact.worldPoint = worldPt;
                contact.worldNormal = worldPlane.direction;
                contact.penetration = -signedDist;
                contactBuffer[idx] = contact;
            }
        }
    }
}

__device__ CudaFloat3 nearestPointOnTriangle(CudaFloat3 query, CudaFloat3 v0, CudaFloat3 v1, CudaFloat3 v2) 
{
    CudaFloat3 e01 = v1 - v0;
    CudaFloat3 e02 = v2 - v0;
    CudaFloat3 v0q = query - v0;

    RealType d1 = innerProduct(e01, v0q);
    RealType d2 = innerProduct(e02, v0q);
    if (d1 <= static_cast<RealType>(0) && d2 <= static_cast<RealType>(0)) return v0;

    CudaFloat3 v1q = query - v1;
    RealType d3 = innerProduct(e01, v1q);
    RealType d4 = innerProduct(e02, v1q);
    if (d3 >= static_cast<RealType>(0) && d4 <= d3) return v1;

    RealType vc = d1 * d4 - d3 * d2;
    if (vc <= static_cast<RealType>(0) && d1 >= static_cast<RealType>(0) && d3 <= static_cast<RealType>(0)) 
    {
        RealType t = d1 / (d1 - d3);
        return v0 + e01 * t;
    }

    CudaFloat3 v2q = query - v2;
    RealType d5 = innerProduct(e01, v2q);
    RealType d6 = innerProduct(e02, v2q);
    if (d6 >= static_cast<RealType>(0) && d5 <= d6) return v2;

    RealType vb = d5 * d2 - d1 * d6;
    if (vb <= static_cast<RealType>(0) && d2 >= static_cast<RealType>(0) && d6 <= static_cast<RealType>(0)) 
    {
        RealType t = d2 / (d2 - d6);
        return v0 + e02 * t;
    }

    RealType va = d3 * d6 - d5 * d4;
    if (va <= static_cast<RealType>(0) && (d4 - d3) >= static_cast<RealType>(0) && (d5 - d6) >= static_cast<RealType>(0)) 
    {
        RealType t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * t;
    }

    RealType denomInv = static_cast<RealType>(1) / (va + vb + vc);
    RealType u = vb * denomInv;
    RealType w = vc * denomInv;
    return v0 + e01 * u + e02 * w;
}

__global__ void kernelDetectEntityCollision(
    const CudaFloat3* pointsA,
    const DeviceTreeNode* nodesA,
    const IntType* faceOrderA,
    const CudaInt3* facesA,
    IntType nodeCountA,
    CudaFloat3 posA,
    CudaFloat4 rotA,
    const CudaFloat3* pointsB,
    IntType pointCountB,
    CudaFloat3 posB,
    CudaFloat4 rotB,
    IntType entityIdxA,
    IntType entityIdxB,
    DeviceContact* contactBuffer,
    IntType* contactCounter,
    IntType contactLimit,
    RealType proximityThreshold)
{
    IntType vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= pointCountB) return;

    CudaFloat3 localPtB = pointsB[vid];
    CudaFloat3 worldPt = applyQuaternion(localPtB, rotB) + posB;
    CudaFloat4 rotAInv = make_float4(-rotA.x, -rotA.y, -rotA.z, rotA.w);
    CudaFloat3 queryInA = applyQuaternion(worldPt - posA, rotAInv);

    IntType traversalStack[64];
    IntType stackTop = 0;
    traversalStack[stackTop++] = 0;

    RealType minDistSq = proximityThreshold * proximityThreshold;
    CudaFloat3 nearestNormal = make_float3(0, 0, 0);
    CudaFloat3 nearestPoint = make_float3(0, 0, 0);
    bool foundContact = false;

    while (stackTop > 0) 
    {
        IntType currentNode = traversalStack[--stackTop];
        DeviceTreeNode node = nodesA[currentNode];

        RealType boxDistSq = static_cast<RealType>(0);
        for (IntType axis = 0; axis < 3; ++axis) 
        {
            RealType coord = (axis == 0) ? queryInA.x : ((axis == 1) ? queryInA.y : queryInA.z);
            RealType boxMin = (axis == 0) ? node.volume.lo.x : 
                              ((axis == 1) ? node.volume.lo.y : node.volume.lo.z);
            RealType boxMax = (axis == 0) ? node.volume.hi.x : 
                              ((axis == 1) ? node.volume.hi.y : node.volume.hi.z);

            if (coord < boxMin) boxDistSq += (boxMin - coord) * (boxMin - coord);
            else if (coord > boxMax) boxDistSq += (coord - boxMax) * (coord - boxMax);
        }
        if (boxDistSq > minDistSq) continue;

        if (node.terminal()) 
        {
            IntType primIdx = faceOrderA[node.leafStart];
            CudaInt3 face = facesA[primIdx];

            CudaFloat3 v0 = pointsA[face.x];
            CudaFloat3 v1 = pointsA[face.y];
            CudaFloat3 v2 = pointsA[face.z];

            CudaFloat3 closest = nearestPointOnTriangle(queryInA, v0, v1, v2);
            CudaFloat3 diff = queryInA - closest;
            RealType distSq = innerProduct(diff, diff);

            if (distSq < minDistSq && distSq > static_cast<RealType>(1e-12)) 
            {
                minDistSq = distSq;
                nearestPoint = closest;

                CudaFloat3 edge1 = v1 - v0;
                CudaFloat3 edge2 = v2 - v0;
                CudaFloat3 faceNormal = crossProduct(edge1, edge2);
                RealType normalLen = sqrtf(innerProduct(faceNormal, faceNormal));
                if (normalLen > static_cast<RealType>(1e-6)) 
                {
                    faceNormal = faceNormal * (static_cast<RealType>(1) / normalLen);
                    if (innerProduct(faceNormal, diff) < static_cast<RealType>(0)) 
                        faceNormal = faceNormal * static_cast<RealType>(-1);
                    nearestNormal = faceNormal;
                    foundContact = true;
                }
            }
        } 
        else 
        {
            if (node.leftChild >= 0 && stackTop < 63) traversalStack[stackTop++] = node.leftChild;
            if (node.rightChild >= 0 && stackTop < 63) traversalStack[stackTop++] = node.rightChild;
        }
    }

    if (foundContact) 
    {
        IntType idx = atomicAdd(contactCounter, 1);
        if (idx < contactLimit) 
        {
            DeviceContact contact;
            contact.entityIdxA = entityIdxA;
            contact.entityIdxB = entityIdxB;
            contact.worldPoint = worldPt;
            contact.worldNormal = applyQuaternion(nearestNormal, rotA);
            contact.penetration = sqrtf(minDistSq);
            contactBuffer[idx] = contact;
        }
    }
}

/* ========== NarrowPhaseDetector Implementation ========== */

NarrowPhaseDetector::NarrowPhaseDetector() 
{
    cudaMalloc(&m_dContactBuffer, m_contactLimit * sizeof(DeviceContact));
    cudaMalloc(&m_dContactCounter, sizeof(IntType));
    m_broadPhase = std::make_unique<BroadPhaseDetector>();
}

NarrowPhaseDetector::~NarrowPhaseDetector() 
{
    release();
}

void NarrowPhaseDetector::release() 
{
    if (m_dContactBuffer) cudaFree(m_dContactBuffer);
    if (m_dContactCounter) cudaFree(m_dContactCounter);
    if (m_dPlaneBuffer) cudaFree(m_dPlaneBuffer);
    m_dContactBuffer = nullptr;
    m_dContactCounter = nullptr;
    m_dPlaneBuffer = nullptr;
    m_broadPhase.reset();
}

void NarrowPhaseDetector::assignTree(DeviceTreeBuilder* treeBuilder) 
{
    m_activeTree = treeBuilder;
}

void NarrowPhaseDetector::performBroadPhase(const DynArray<Point3>& boundsMin,
                                             const DynArray<Point3>& boundsMax,
                                             DynArray<EntityPair>& candidatePairs) 
{
    if (m_broadPhase) 
    {
        m_broadPhase->findCandidatePairs(boundsMin, boundsMax, candidatePairs);
    }
}

void NarrowPhaseDetector::detectEntityEnvironment(
    IntType entityIdx,
    const EntityTransform& transform,
    const DevicePlane* planeArray,
    IntType planeCount,
    DynArray<DeviceContact>& contactResults)
{
    if (!m_activeTree || m_activeTree->totalPoints() == 0) return;

    if (m_planeBufferSize != planeCount) 
    {
        if (m_dPlaneBuffer) cudaFree(m_dPlaneBuffer);
        cudaMalloc(&m_dPlaneBuffer, planeCount * sizeof(DevicePlane));
        m_planeBufferSize = planeCount;
    }
    cudaMemcpy(m_dPlaneBuffer, planeArray, planeCount * sizeof(DevicePlane), cudaMemcpyHostToDevice);

    IntType* dProcessedFlags;
    IntType pointTotal = m_activeTree->totalPoints();
    cudaMalloc(&dProcessedFlags, pointTotal * sizeof(IntType));
    cudaMemset(dProcessedFlags, 0, pointTotal * sizeof(IntType));

    IntType zero = 0;
    cudaMemcpy(m_dContactCounter, &zero, sizeof(IntType), cudaMemcpyHostToDevice);

    CudaFloat3 entityPos = make_float3(transform.translation.x,
                                        transform.translation.y,
                                        transform.translation.z);
    CudaFloat4 entityRot = transform.rotation;

    IntType threadsPerBlock = 256;
    IntType numBlocks = (m_activeTree->totalFaces() + threadsPerBlock - 1) / threadsPerBlock;

    for (IntType p = 0; p < planeCount; ++p) 
    {
        cudaMemset(dProcessedFlags, 0, pointTotal * sizeof(IntType));

        kernelDetectPlaneCollision<<<numBlocks, threadsPerBlock>>>(
            m_activeTree->pointBuffer(),
            m_activeTree->nodeBuffer(),
            m_activeTree->faceOrderBuffer(),
            m_activeTree->faceBuffer(),
            m_activeTree->totalFaces(),
            entityPos,
            entityRot,
            planeArray[p],
            entityIdx,
            m_dContactBuffer,
            m_dContactCounter,
            m_contactLimit,
            dProcessedFlags,
            pointTotal);
    }

    cudaFree(dProcessedFlags);

    IntType contactCount;
    cudaMemcpy(&contactCount, m_dContactCounter, sizeof(IntType), cudaMemcpyDeviceToHost);
    contactCount = std::min(contactCount, m_contactLimit);

    contactResults.resize(contactCount);
    if (contactCount > 0) 
    {
        cudaMemcpy(contactResults.data(), m_dContactBuffer,
                   contactCount * sizeof(DeviceContact), cudaMemcpyDeviceToHost);
    }
}

void NarrowPhaseDetector::detectEntityEntity(
    IntType entityAIdx,
    const EntityTransform& transformA,
    DeviceTreeBuilder* treeA,
    IntType entityBIdx,
    const EntityTransform& transformB,
    DeviceTreeBuilder* treeB,
    DynArray<DeviceContact>& contactResults)
{
    if (!treeA || !treeB) return;

    IntType zero = 0;
    cudaMemcpy(m_dContactCounter, &zero, sizeof(IntType), cudaMemcpyHostToDevice);

    constexpr RealType kProximityThreshold = static_cast<RealType>(0.5);

    IntType threadsPerBlock = 256;
    IntType numBlocks = (treeB->totalPoints() + threadsPerBlock - 1) / threadsPerBlock;

    CudaFloat3 posA = make_float3(transformA.translation.x,
                                   transformA.translation.y,
                                   transformA.translation.z);
    CudaFloat3 posB = make_float3(transformB.translation.x,
                                   transformB.translation.y,
                                   transformB.translation.z);

    kernelDetectEntityCollision<<<numBlocks, threadsPerBlock>>>(
        treeA->pointBuffer(),
        treeA->nodeBuffer(),
        treeA->faceOrderBuffer(),
        treeA->faceBuffer(),
        treeA->totalNodes(),
        posA,
        transformA.rotation,
        treeB->pointBuffer(),
        treeB->totalPoints(),
        posB,
        transformB.rotation,
        entityAIdx,
        entityBIdx,
        m_dContactBuffer,
        m_dContactCounter,
        m_contactLimit,
        kProximityThreshold);

    numBlocks = (treeA->totalPoints() + threadsPerBlock - 1) / threadsPerBlock;

    kernelDetectEntityCollision<<<numBlocks, threadsPerBlock>>>(
        treeB->pointBuffer(),
        treeB->nodeBuffer(),
        treeB->faceOrderBuffer(),
        treeB->faceBuffer(),
        treeB->totalNodes(),
        posB,
        transformB.rotation,
        treeA->pointBuffer(),
        treeA->totalPoints(),
        posA,
        transformA.rotation,
        entityBIdx,
        entityAIdx,
        m_dContactBuffer,
        m_dContactCounter,
        m_contactLimit,
        kProximityThreshold);

    IntType contactCount;
    cudaMemcpy(&contactCount, m_dContactCounter, sizeof(IntType), cudaMemcpyDeviceToHost);
    contactCount = std::min(contactCount, m_contactLimit);

    contactResults.resize(contactCount);
    if (contactCount > 0) 
    {
        cudaMemcpy(contactResults.data(), m_dContactBuffer,
                   contactCount * sizeof(DeviceContact), cudaMemcpyDeviceToHost);
    }
}

}  // namespace gpu
}  // namespace phys3d
