/*
 * Implementation: CUDA Narrow Phase Detection
 * Restructured with fused kernels and modified traversal patterns
 */
#include <cstdio>
#include <algorithm>

#include "contact_kernel.cuh"

namespace phys3d {
namespace gpu {

/* ========== Device Helper Functions ========== */

// Check if bounds overlap with plane using support point method
__device__ __forceinline__ bool checkBoundsPlaneIntersection(
    const DeviceBounds3D& bounds,
    const CudaFloat3& planeNormal,
    RealType planeOffset)
{
    // Compute support point in negative normal direction
    CudaFloat3 support;
    support.x = (planeNormal.x >= 0) ? bounds.lo.x : bounds.hi.x;
    support.y = (planeNormal.y >= 0) ? bounds.lo.y : bounds.hi.y;
    support.z = (planeNormal.z >= 0) ? bounds.lo.z : bounds.hi.z;
    
    // Check if support point is on negative side of plane
    const RealType signedDist = innerProduct(support, planeNormal) - planeOffset;
    return signedDist <= static_cast<RealType>(0);
}

// Compute squared distance from point to AABB
__device__ __forceinline__ RealType computePointBoxDistSq(
    const CudaFloat3& pt,
    const DeviceBounds3D& box)
{
    RealType distSq = static_cast<RealType>(0);
    
    // X component
    RealType dx = static_cast<RealType>(0);
    if (pt.x < box.lo.x) dx = box.lo.x - pt.x;
    else if (pt.x > box.hi.x) dx = pt.x - box.hi.x;
    distSq += dx * dx;
    
    // Y component
    RealType dy = static_cast<RealType>(0);
    if (pt.y < box.lo.y) dy = box.lo.y - pt.y;
    else if (pt.y > box.hi.y) dy = pt.y - box.hi.y;
    distSq += dy * dy;
    
    // Z component
    RealType dz = static_cast<RealType>(0);
    if (pt.z < box.lo.z) dz = box.lo.z - pt.z;
    else if (pt.z > box.hi.z) dz = pt.z - box.hi.z;
    distSq += dz * dz;
    
    return distSq;
}

// Store contact to buffer with atomic index allocation
__device__ __forceinline__ void recordContact(
    DeviceContact* buffer,
    IntType* counter,
    IntType limit,
    IntType entityA,
    IntType entityB,
    const CudaFloat3& worldPt,
    const CudaFloat3& worldNormal,
    RealType penetration)
{
    const IntType idx = atomicAdd(counter, 1);
    if (idx < limit)
    {
        DeviceContact& contact = buffer[idx];
        contact.entityIdxA = entityA;
        contact.entityIdxB = entityB;
        contact.worldPoint = worldPt;
        contact.worldNormal = worldNormal;
        contact.penetration = penetration;
    }
}

/* ========== Plane Collision Detection Kernel ========== */

__global__ void kernelDetectPlaneCollisions(
    const CudaFloat3* __restrict__ pointData,
    const DeviceTreeNode* __restrict__ nodeData,
    const IntType* __restrict__ faceOrderData,
    const CudaInt3* __restrict__ faceData,
    IntType faceTotal,
    CudaFloat3 entityPos,
    CudaFloat4 entityRot,
    DevicePlane worldPlane,
    IntType entityIdx,
    DeviceContact* __restrict__ contactBuffer,
    IntType* __restrict__ contactCounter,
    IntType contactLimit,
    IntType* __restrict__ processedPoints,
    IntType pointTotal)
{
    const IntType leafId = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafId >= faceTotal) return;

    const IntType nodeIdx = faceTotal - 1 + leafId;
    const DeviceTreeNode node = nodeData[nodeIdx];

    // Transform plane to local coordinate frame
    const CudaFloat4 rotInv = make_float4(-entityRot.x, -entityRot.y, -entityRot.z, entityRot.w);
    const CudaFloat3 localNormal = applyQuaternion(worldPlane.direction, rotInv);
    const RealType localOffset = worldPlane.offset - innerProduct(worldPlane.direction, entityPos);

    // Early rejection: test AABB vs plane
    if (!checkBoundsPlaneIntersection(node.volume, localNormal, localOffset))
        return;

    // Get primitive index
    const IntType primIdx = faceOrderData[leafId];
    const CudaInt3 face = faceData[primIdx];
    
    // Process each vertex of the triangle
    const IntType vertexIds[3] = {face.x, face.y, face.z};
    
    IntType k = 0;
    do {
        const IntType vid = vertexIds[k];
        
        // Atomically mark vertex as processed
        const IntType alreadyDone = atomicExch(&processedPoints[vid], 1);
        if (alreadyDone)
        {
            ++k;
            continue;
        }

        // Load vertex position
        const CudaFloat3 localPt = pointData[vid];
        
        // Compute signed distance
        const RealType signedDist = innerProduct(localPt, localNormal) - localOffset;

        if (signedDist < static_cast<RealType>(0))
        {
            // Transform to world coordinates
            const CudaFloat3 worldPt = applyQuaternion(localPt, entityRot) + entityPos;
            
            recordContact(contactBuffer, contactCounter, contactLimit,
                          -1, entityIdx, worldPt, worldPlane.direction, -signedDist);
        }
        
        ++k;
    } while (k < 3);
}

/* ========== Triangle Closest Point (Barycentric Method) ========== */

__device__ CudaFloat3 computeTriangleClosestPoint(
    const CudaFloat3& query,
    const CudaFloat3& v0,
    const CudaFloat3& v1,
    const CudaFloat3& v2)
{
    // Edge vectors
    const CudaFloat3 e01 = v1 - v0;
    const CudaFloat3 e02 = v2 - v0;
    const CudaFloat3 v0q = query - v0;

    // Vertex region 0
    const RealType d1 = innerProduct(e01, v0q);
    const RealType d2 = innerProduct(e02, v0q);
    if (d1 <= static_cast<RealType>(0) && d2 <= static_cast<RealType>(0))
        return v0;

    // Vertex region 1
    const CudaFloat3 v1q = query - v1;
    const RealType d3 = innerProduct(e01, v1q);
    const RealType d4 = innerProduct(e02, v1q);
    if (d3 >= static_cast<RealType>(0) && d4 <= d3)
        return v1;

    // Edge region 01
    const RealType vc = d1 * d4 - d3 * d2;
    bool inEdge01 = (vc <= static_cast<RealType>(0)) && 
                    (d1 >= static_cast<RealType>(0)) && 
                    (d3 <= static_cast<RealType>(0));
    if (inEdge01)
    {
        const RealType t = d1 / (d1 - d3);
        return v0 + e01 * t;
    }

    // Vertex region 2
    const CudaFloat3 v2q = query - v2;
    const RealType d5 = innerProduct(e01, v2q);
    const RealType d6 = innerProduct(e02, v2q);
    if (d6 >= static_cast<RealType>(0) && d5 <= d6)
        return v2;

    // Edge region 02
    const RealType vb = d5 * d2 - d1 * d6;
    bool inEdge02 = (vb <= static_cast<RealType>(0)) && 
                    (d2 >= static_cast<RealType>(0)) && 
                    (d6 <= static_cast<RealType>(0));
    if (inEdge02)
    {
        const RealType t = d2 / (d2 - d6);
        return v0 + e02 * t;
    }

    // Edge region 12
    const RealType va = d3 * d6 - d5 * d4;
    bool inEdge12 = (va <= static_cast<RealType>(0)) && 
                    ((d4 - d3) >= static_cast<RealType>(0)) && 
                    ((d5 - d6) >= static_cast<RealType>(0));
    if (inEdge12)
    {
        const RealType t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * t;
    }

    // Interior region
    const RealType denomInv = static_cast<RealType>(1) / (va + vb + vc);
    const RealType u = vb * denomInv;
    const RealType w = vc * denomInv;
    return v0 + e01 * u + e02 * w;
}

/* ========== Entity-Entity Collision Detection Kernel ========== */

__global__ void kernelDetectEntityCollisions(
    const CudaFloat3* __restrict__ pointsA,
    const DeviceTreeNode* __restrict__ nodesA,
    const IntType* __restrict__ faceOrderA,
    const CudaInt3* __restrict__ facesA,
    IntType nodeCountA,
    CudaFloat3 posA,
    CudaFloat4 rotA,
    const CudaFloat3* __restrict__ pointsB,
    IntType pointCountB,
    CudaFloat3 posB,
    CudaFloat4 rotB,
    IntType entityIdxA,
    IntType entityIdxB,
    DeviceContact* __restrict__ contactBuffer,
    IntType* __restrict__ contactCounter,
    IntType contactLimit,
    RealType proximityThreshold)
{
    const IntType vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= pointCountB) return;

    // Transform B's vertex to world, then to A's local frame
    const CudaFloat3 localPtB = pointsB[vid];
    const CudaFloat3 worldPt = applyQuaternion(localPtB, rotB) + posB;
    
    const CudaFloat4 rotAInv = make_float4(-rotA.x, -rotA.y, -rotA.z, rotA.w);
    const CudaFloat3 queryInA = applyQuaternion(worldPt - posA, rotAInv);

    // BVH traversal using explicit stack
    IntType traversalStack[64];
    IntType stackTop = 0;
    traversalStack[stackTop++] = 0;

    RealType minDistSq = proximityThreshold * proximityThreshold;
    CudaFloat3 nearestNormal = make_float3(0, 0, 0);
    bool foundContact = false;

    while (stackTop > 0)
    {
        const IntType currentNode = traversalStack[--stackTop];
        const DeviceTreeNode node = nodesA[currentNode];

        // Distance-based culling
        const RealType boxDistSq = computePointBoxDistSq(queryInA, node.volume);
        if (boxDistSq > minDistSq)
            continue;

        if (node.terminal())
        {
            // Process leaf node
            const IntType primIdx = faceOrderA[node.leafStart];
            const CudaInt3 face = facesA[primIdx];

            const CudaFloat3 t0 = pointsA[face.x];
            const CudaFloat3 t1 = pointsA[face.y];
            const CudaFloat3 t2 = pointsA[face.z];

            const CudaFloat3 closest = computeTriangleClosestPoint(queryInA, t0, t1, t2);
            const CudaFloat3 diff = queryInA - closest;
            const RealType distSq = innerProduct(diff, diff);

            if (distSq < minDistSq && distSq > static_cast<RealType>(1e-12))
            {
                minDistSq = distSq;

                // Compute face normal
                const CudaFloat3 edge1 = t1 - t0;
                const CudaFloat3 edge2 = t2 - t0;
                CudaFloat3 faceNormal = crossProduct(edge1, edge2);
                
                const RealType normalLen = sqrtf(innerProduct(faceNormal, faceNormal));
                if (normalLen > static_cast<RealType>(1e-6))
                {
                    faceNormal = faceNormal * (static_cast<RealType>(1) / normalLen);
                    
                    // Ensure normal points toward query
                    if (innerProduct(faceNormal, diff) < static_cast<RealType>(0))
                        faceNormal = faceNormal * static_cast<RealType>(-1);
                    
                    nearestNormal = faceNormal;
                    foundContact = true;
                }
            }
        }
        else
        {
            // Push children onto stack
            if (node.leftChild >= 0 && stackTop < 63)
                traversalStack[stackTop++] = node.leftChild;
            if (node.rightChild >= 0 && stackTop < 63)
                traversalStack[stackTop++] = node.rightChild;
        }
    }

    if (foundContact)
    {
        const CudaFloat3 worldNormal = applyQuaternion(nearestNormal, rotA);
        recordContact(contactBuffer, contactCounter, contactLimit,
                      entityIdxA, entityIdxB, worldPt, worldNormal, sqrtf(minDistSq));
    }
}

/* ========== NarrowPhaseDetector Implementation ========== */

NarrowPhaseDetector::NarrowPhaseDetector()
    : m_dContactBuffer(nullptr)
    , m_dContactCounter(nullptr)
    , m_dPlaneBuffer(nullptr)
    , m_activeTree(nullptr)
    , m_planeBufferSize(0)
    , m_contactLimit(65536)
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
    // Free all device memory
    void* buffers[] = {m_dContactBuffer, m_dContactCounter, m_dPlaneBuffer};
    
    IntType idx = 0;
    while (idx < 3)
    {
        if (buffers[idx] != nullptr)
            cudaFree(buffers[idx]);
        ++idx;
    }
    
    m_dContactBuffer = nullptr;
    m_dContactCounter = nullptr;
    m_dPlaneBuffer = nullptr;
    m_broadPhase.reset();
}

void NarrowPhaseDetector::assignTree(DeviceTreeBuilder* treeBuilder)
{
    m_activeTree = treeBuilder;
}

void NarrowPhaseDetector::performBroadPhase(
    const DynArray<Point3>& boundsMin,
    const DynArray<Point3>& boundsMax,
    DynArray<EntityPair>& candidatePairs)
{
    if (m_broadPhase)
        m_broadPhase->findCandidatePairs(boundsMin, boundsMax, candidatePairs);
}

void NarrowPhaseDetector::detectEntityEnvironment(
    IntType entityIdx,
    const EntityTransform& transform,
    const DevicePlane* planeArray,
    IntType planeCount,
    DynArray<DeviceContact>& contactResults)
{
    // Validate input
    if (m_activeTree == nullptr)
        return;
    if (m_activeTree->totalPoints() == 0)
        return;

    // Ensure plane buffer is large enough
    if (m_planeBufferSize != planeCount)
    {
        if (m_dPlaneBuffer != nullptr)
            cudaFree(m_dPlaneBuffer);
        cudaMalloc(&m_dPlaneBuffer, planeCount * sizeof(DevicePlane));
        m_planeBufferSize = planeCount;
    }
    cudaMemcpy(m_dPlaneBuffer, planeArray, planeCount * sizeof(DevicePlane), cudaMemcpyHostToDevice);

    // Allocate vertex tracking buffer
    const IntType pointTotal = m_activeTree->totalPoints();
    IntType* dProcessedFlags;
    cudaMalloc(&dProcessedFlags, pointTotal * sizeof(IntType));

    // Reset contact counter
    const IntType zero = 0;
    cudaMemcpy(m_dContactCounter, &zero, sizeof(IntType), cudaMemcpyHostToDevice);

    // Prepare transform data
    const CudaFloat3 entityPos = make_float3(
        transform.translation.x,
        transform.translation.y,
        transform.translation.z);
    const CudaFloat4 entityRot = transform.rotation;

    // Kernel launch configuration
    constexpr IntType kBlockSize = 256;
    const IntType numBlocks = (m_activeTree->totalFaces() + kBlockSize - 1) / kBlockSize;

    // Process each plane
    IntType planeIdx = 0;
    while (planeIdx < planeCount)
    {
        cudaMemset(dProcessedFlags, 0, pointTotal * sizeof(IntType));

        kernelDetectPlaneCollisions<<<numBlocks, kBlockSize>>>(
            m_activeTree->pointBuffer(),
            m_activeTree->nodeBuffer(),
            m_activeTree->faceOrderBuffer(),
            m_activeTree->faceBuffer(),
            m_activeTree->totalFaces(),
            entityPos,
            entityRot,
            planeArray[planeIdx],
            entityIdx,
            m_dContactBuffer,
            m_dContactCounter,
            m_contactLimit,
            dProcessedFlags,
            pointTotal);

        ++planeIdx;
    }

    cudaFree(dProcessedFlags);

    // Retrieve results
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
    // Validate inputs
    if (treeA == nullptr || treeB == nullptr)
        return;

    // Reset counter
    const IntType zero = 0;
    cudaMemcpy(m_dContactCounter, &zero, sizeof(IntType), cudaMemcpyHostToDevice);

    constexpr RealType kProximityThreshold = static_cast<RealType>(0.5);
    constexpr IntType kBlockSize = 256;

    // Prepare transform data
    const CudaFloat3 posA = make_float3(transformA.translation.x,
                                         transformA.translation.y,
                                         transformA.translation.z);
    const CudaFloat3 posB = make_float3(transformB.translation.x,
                                         transformB.translation.y,
                                         transformB.translation.z);

    // Test B's vertices against A's surface
    IntType numBlocks = (treeB->totalPoints() + kBlockSize - 1) / kBlockSize;
    kernelDetectEntityCollisions<<<numBlocks, kBlockSize>>>(
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

    // Test A's vertices against B's surface
    numBlocks = (treeA->totalPoints() + kBlockSize - 1) / kBlockSize;
    kernelDetectEntityCollisions<<<numBlocks, kBlockSize>>>(
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

    // Retrieve results
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
