/**
 * @file collision_detector.cu
 * @brief CUDA implementation of GPU collision detection.
 */
#include <cstdio>
#include <algorithm>

#include "collision_detector.cuh"

namespace rigid {
namespace gpu {

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void detectBVHPlaneCollisionKernel(
    const Float3* vertices,
    const BVHNodeDevice* nodes,
    const Int* primIndices,
    const Int3* triangles,
    Int numTriangles,
    Float3 bodyPos,
    Float4 bodyOrientation,
    PlaneDevice planeWorld,
    Int bodyIdx,
    ContactDevice* contacts,
    Int* contactCount,
    Int maxContacts,
    Int* visitedVertices,
    Int numVertices)
{
    Int leafId = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafId >= numTriangles) return;

    Int nodeIdx = numTriangles - 1 + leafId;
    BVHNodeDevice node = nodes[nodeIdx];

    Float4 qInv = make_float4(-bodyOrientation.x, -bodyOrientation.y,
                              -bodyOrientation.z, bodyOrientation.w);
    Float3 planeLocalNormal = rotateByQuat(planeWorld.normal, qInv);
    Float planeLocalOffset = planeWorld.offset - dot(planeWorld.normal, bodyPos);

    if (!aabbIntersectsPlane(node.bounds, planeLocalNormal, planeLocalOffset))
        return;

    Int primIdx = primIndices[leafId];
    Int3 tri = triangles[primIdx];
    Int vertIds[3] = {tri.x, tri.y, tri.z};

    for (Int k = 0; k < 3; ++k) {
        Int vid = vertIds[k];
        Int wasVisited = atomicExch(&visitedVertices[vid], 1);
        if (wasVisited) continue;

        Float3 vLocal = vertices[vid];
        Float dist = dot(vLocal, planeLocalNormal) - planeLocalOffset;

        if (dist < 0.0f) {
            Float3 vWorld = rotateByQuat(vLocal, bodyOrientation) + bodyPos;
            Int idx = atomicAdd(contactCount, 1);
            if (idx < maxContacts) {
                ContactDevice contact;
                contact.bodyIndexA = -1;
                contact.bodyIndexB = bodyIdx;
                contact.position = vWorld;
                contact.normal = planeWorld.normal;
                contact.depth = -dist;
                contacts[idx] = contact;
            }
        }
    }
}

__device__ Float3 closestPointOnTriangle(Float3 p, Float3 a, Float3 b, Float3 c) {
    Float3 ab = b - a;
    Float3 ac = c - a;
    Float3 ap = p - a;

    Float d1 = dot(ab, ap);
    Float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    Float3 bp = p - b;
    Float d3 = dot(ab, bp);
    Float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    Float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        Float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    Float3 cp = p - c;
    Float d5 = dot(ab, cp);
    Float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    Float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        Float w = d2 / (d2 - d6);
        return a + ac * w;
    }

    Float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        Float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    Float denom = 1.0f / (va + vb + vc);
    Float v = vb * denom;
    Float w = vc * denom;
    return a + ab * v + ac * w;
}

__global__ void detectBodyBodyCollisionKernel(
    const Float3* verticesA,
    const BVHNodeDevice* nodesA,
    const Int* primIndicesA,
    const Int3* trianglesA,
    Int numNodesA,
    Float3 posA,
    Float4 orientationA,
    const Float3* verticesB,
    Int numVerticesB,
    Float3 posB,
    Float4 orientationB,
    Int bodyIdxA,
    Int bodyIdxB,
    ContactDevice* contacts,
    Int* contactCount,
    Int maxContacts,
    Float collisionThreshold)
{
    Int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerticesB) return;

    Float3 vLocalB = verticesB[vid];
    Float3 vWorld = rotateByQuat(vLocalB, orientationB) + posB;
    Float4 qInvA = make_float4(-orientationA.x, -orientationA.y,
                               -orientationA.z, orientationA.w);
    Float3 pLocalA = rotateByQuat(vWorld - posA, qInvA);

    Int stack[64];
    Int stackPtr = 0;
    stack[stackPtr++] = 0;

    Float minDistSq = collisionThreshold * collisionThreshold;
    Float3 closestNormal = make_float3(0, 0, 0);
    Float3 closestPoint = make_float3(0, 0, 0);
    bool found = false;

    while (stackPtr > 0) {
        Int nodeIdx = stack[--stackPtr];
        BVHNodeDevice node = nodesA[nodeIdx];

        Float distSq = 0.0f;
        for (Int i = 0; i < 3; ++i) {
            Float coord = (i == 0) ? pLocalA.x : ((i == 1) ? pLocalA.y : pLocalA.z);
            Float boxMin = (i == 0) ? node.bounds.min.x : 
                          ((i == 1) ? node.bounds.min.y : node.bounds.min.z);
            Float boxMax = (i == 0) ? node.bounds.max.x : 
                          ((i == 1) ? node.bounds.max.y : node.bounds.max.z);

            if (coord < boxMin) distSq += (boxMin - coord) * (boxMin - coord);
            else if (coord > boxMax) distSq += (coord - boxMax) * (coord - boxMax);
        }
        if (distSq > minDistSq) continue;

        if (node.isLeaf()) {
            Int primIdx = primIndicesA[node.firstPrim];
            Int3 tri = trianglesA[primIdx];

            Float3 p0 = verticesA[tri.x];
            Float3 p1 = verticesA[tri.y];
            Float3 p2 = verticesA[tri.z];

            Float3 closest = closestPointOnTriangle(pLocalA, p0, p1, p2);
            Float3 diff = pLocalA - closest;
            Float dSq = dot(diff, diff);

            if (dSq < minDistSq && dSq > 1e-12f) {
                minDistSq = dSq;
                closestPoint = closest;

                Float3 edge1 = p1 - p0;
                Float3 edge2 = p2 - p0;
                Float3 normal = cross(edge1, edge2);
                Float len = sqrtf(dot(normal, normal));
                if (len > 1e-6f) {
                    normal = normal * (1.0f / len);
                    if (dot(normal, diff) < 0) normal = normal * (-1.0f);
                    closestNormal = normal;
                    found = true;
                }
            }
        } else {
            if (node.left >= 0 && stackPtr < 63) stack[stackPtr++] = node.left;
            if (node.right >= 0 && stackPtr < 63) stack[stackPtr++] = node.right;
        }
    }

    if (found) {
        Int idx = atomicAdd(contactCount, 1);
        if (idx < maxContacts) {
            ContactDevice contact;
            contact.bodyIndexA = bodyIdxA;
            contact.bodyIndexB = bodyIdxB;
            contact.position = vWorld;
            contact.normal = rotateByQuat(closestNormal, orientationA);
            contact.depth = sqrtf(minDistSq);
            contacts[idx] = contact;
        }
    }
}

// ============================================================================
// CollisionDetectorGPU Implementation
// ============================================================================

CollisionDetectorGPU::CollisionDetectorGPU() {
    cudaMalloc(&dContacts_, maxContacts_ * sizeof(ContactDevice));
    cudaMalloc(&dContactCount_, sizeof(Int));
    broadphase_ = std::make_unique<BroadphaseGPU>();
}

CollisionDetectorGPU::~CollisionDetectorGPU() {
    free();
}

void CollisionDetectorGPU::free() {
    if (dContacts_) cudaFree(dContacts_);
    if (dContactCount_) cudaFree(dContactCount_);
    if (dPlanes_) cudaFree(dPlanes_);
    dContacts_ = nullptr;
    dContactCount_ = nullptr;
    dPlanes_ = nullptr;
    broadphase_.reset();
}

void CollisionDetectorGPU::setBVH(BVHBuilderGPU* bvhBuilder) {
    bvh_ = bvhBuilder;
}

void CollisionDetectorGPU::broadphaseDetect(const Vector<Vec3>& aabbMins,
                                             const Vector<Vec3>& aabbMaxs,
                                             Vector<CollisionPair>& outPairs) {
    if (broadphase_) {
        broadphase_->detectPairs(aabbMins, aabbMaxs, outPairs);
    }
}

void CollisionDetectorGPU::detectBodyEnvironment(
    Int bodyIdx,
    const BodyTransformDevice& transform,
    const PlaneDevice* planes,
    Int numPlanes,
    Vector<ContactDevice>& outContacts)
{
    if (!bvh_ || bvh_->vertexCount() == 0) return;

    if (numPlanes_ != numPlanes) {
        if (dPlanes_) cudaFree(dPlanes_);
        cudaMalloc(&dPlanes_, numPlanes * sizeof(PlaneDevice));
        numPlanes_ = numPlanes;
    }
    cudaMemcpy(dPlanes_, planes, numPlanes * sizeof(PlaneDevice),
               cudaMemcpyHostToDevice);

    Int* dVisited;
    Int numVerts = bvh_->vertexCount();
    cudaMalloc(&dVisited, numVerts * sizeof(Int));
    cudaMemset(dVisited, 0, numVerts * sizeof(Int));

    Int zero = 0;
    cudaMemcpy(dContactCount_, &zero, sizeof(Int), cudaMemcpyHostToDevice);

    Float3 bodyPos = make_float3(transform.position.x,
                                  transform.position.y,
                                  transform.position.z);
    Float4 bodyOri = transform.orientation;

    Int blockSize = 256;
    Int numBlocks = (bvh_->triangleCount() + blockSize - 1) / blockSize;

    for (Int p = 0; p < numPlanes; ++p) {
        cudaMemset(dVisited, 0, numVerts * sizeof(Int));

        detectBVHPlaneCollisionKernel<<<numBlocks, blockSize>>>(
            bvh_->deviceVertices(),
            bvh_->deviceNodes(),
            bvh_->devicePrimIndices(),
            bvh_->deviceTriangles(),
            bvh_->triangleCount(),
            bodyPos,
            bodyOri,
            planes[p],
            bodyIdx,
            dContacts_,
            dContactCount_,
            maxContacts_,
            dVisited,
            numVerts);
    }

    cudaFree(dVisited);

    Int contactCount;
    cudaMemcpy(&contactCount, dContactCount_, sizeof(Int), cudaMemcpyDeviceToHost);
    contactCount = std::min(contactCount, maxContacts_);

    outContacts.resize(contactCount);
    if (contactCount > 0) {
        cudaMemcpy(outContacts.data(), dContacts_,
                   contactCount * sizeof(ContactDevice), cudaMemcpyDeviceToHost);
    }
}

void CollisionDetectorGPU::detectBodyBody(
    Int bodyAIdx,
    const BodyTransformDevice& transformA,
    BVHBuilderGPU* bvhA,
    Int bodyBIdx,
    const BodyTransformDevice& transformB,
    BVHBuilderGPU* bvhB,
    Vector<ContactDevice>& outContacts)
{
    if (!bvhA || !bvhB) return;

    Int zero = 0;
    cudaMemcpy(dContactCount_, &zero, sizeof(Int), cudaMemcpyHostToDevice);

    constexpr Float kCollisionThreshold = 0.5f;

    Int blockSize = 256;
    Int numBlocks = (bvhB->vertexCount() + blockSize - 1) / blockSize;

    Float3 posA = make_float3(transformA.position.x,
                               transformA.position.y,
                               transformA.position.z);
    Float3 posB = make_float3(transformB.position.x,
                               transformB.position.y,
                               transformB.position.z);

    detectBodyBodyCollisionKernel<<<numBlocks, blockSize>>>(
        bvhA->deviceVertices(),
        bvhA->deviceNodes(),
        bvhA->devicePrimIndices(),
        bvhA->deviceTriangles(),
        bvhA->nodeCount(),
        posA,
        transformA.orientation,
        bvhB->deviceVertices(),
        bvhB->vertexCount(),
        posB,
        transformB.orientation,
        bodyAIdx,
        bodyBIdx,
        dContacts_,
        dContactCount_,
        maxContacts_,
        kCollisionThreshold);

    // Symmetric: B vs A
    numBlocks = (bvhA->vertexCount() + blockSize - 1) / blockSize;

    detectBodyBodyCollisionKernel<<<numBlocks, blockSize>>>(
        bvhB->deviceVertices(),
        bvhB->deviceNodes(),
        bvhB->devicePrimIndices(),
        bvhB->deviceTriangles(),
        bvhB->nodeCount(),
        posB,
        transformB.orientation,
        bvhA->deviceVertices(),
        bvhA->vertexCount(),
        posA,
        transformA.orientation,
        bodyBIdx,
        bodyAIdx,
        dContacts_,
        dContactCount_,
        maxContacts_,
        kCollisionThreshold);

    Int contactCount;
    cudaMemcpy(&contactCount, dContactCount_, sizeof(Int), cudaMemcpyDeviceToHost);
    contactCount = std::min(contactCount, maxContacts_);

    outContacts.resize(contactCount);
    if (contactCount > 0) {
        cudaMemcpy(outContacts.data(), dContacts_,
                   contactCount * sizeof(ContactDevice), cudaMemcpyDeviceToHost);
    }
}

}  // namespace gpu
}  // namespace rigid
