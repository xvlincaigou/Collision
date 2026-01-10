/**
 * @file bvh_builder.cu
 * @brief CUDA implementation of GPU BVH construction.
 */
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <cstdio>

#include "bvh_builder.cuh"

namespace rigid {
namespace gpu {

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeCentroidsBoundsKernel(
    const Float3* vertices,
    const Int3* triangles,
    Float3* centroids,
    AABBDevice* primBounds,
    Float3* sceneMin,
    Float3* sceneMax,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Int3 tri = triangles[idx];
    Float3 v0 = vertices[tri.x];
    Float3 v1 = vertices[tri.y];
    Float3 v2 = vertices[tri.z];

    centroids[idx] = make_float3(
        (v0.x + v1.x + v2.x) / 3.0f,
        (v0.y + v1.y + v2.y) / 3.0f,
        (v0.z + v1.z + v2.z) / 3.0f);

    AABBDevice bounds;
    bounds.min = make_float3(
        fminf(fminf(v0.x, v1.x), v2.x),
        fminf(fminf(v0.y, v1.y), v2.y),
        fminf(fminf(v0.z, v1.z), v2.z));
    bounds.max = make_float3(
        fmaxf(fmaxf(v0.x, v1.x), v2.x),
        fmaxf(fmaxf(v0.y, v1.y), v2.y),
        fmaxf(fmaxf(v0.z, v1.z), v2.z));
    primBounds[idx] = bounds;

    atomicMin(reinterpret_cast<int*>(&sceneMin->x), __float_as_int(bounds.min.x));
    atomicMin(reinterpret_cast<int*>(&sceneMin->y), __float_as_int(bounds.min.y));
    atomicMin(reinterpret_cast<int*>(&sceneMin->z), __float_as_int(bounds.min.z));
    atomicMax(reinterpret_cast<int*>(&sceneMax->x), __float_as_int(bounds.max.x));
    atomicMax(reinterpret_cast<int*>(&sceneMax->y), __float_as_int(bounds.max.y));
    atomicMax(reinterpret_cast<int*>(&sceneMax->z), __float_as_int(bounds.max.z));
}

__global__ void computeMortonCodesKernel(
    const Float3* centroids,
    UInt* mortonCodes,
    Float3 sceneMin,
    Float3 sceneExtent,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Float3 c = centroids[idx];

    float nx = (sceneExtent.x > 0) ? (c.x - sceneMin.x) / sceneExtent.x : 0.5f;
    float ny = (sceneExtent.y > 0) ? (c.y - sceneMin.y) / sceneExtent.y : 0.5f;
    float nz = (sceneExtent.z > 0) ? (c.z - sceneMin.z) / sceneExtent.z : 0.5f;

    mortonCodes[idx] = computeMortonCode3D(nx, ny, nz);
}

__global__ void buildRadixTreeKernel(
    const UInt* sortedMortonCodes,
    const int* sortedIndices,
    BVHNodeDevice* nodes,
    int* parentIndices,
    int* primIndices,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    int dLeft = delta(sortedMortonCodes, n, i, i - 1);
    int dRight = delta(sortedMortonCodes, n, i, i + 1);
    int d = (dRight > dLeft) ? 1 : -1;

    int deltaMin = delta(sortedMortonCodes, n, i, i - d);
    int lMax = 2;
    while (delta(sortedMortonCodes, n, i, i + lMax * d) > deltaMin)
        lMax *= 2;

    int l = 0;
    for (int t = lMax / 2; t >= 1; t /= 2)
        if (delta(sortedMortonCodes, n, i, i + (l + t) * d) > deltaMin)
            l = l + t;
    int j = i + l * d;

    int deltaNode = delta(sortedMortonCodes, n, i, j);
    int s = 0;
    int div = 2;
    int t = (l + div - 1) / div;
    while (t >= 1) {
        if (delta(sortedMortonCodes, n, i, i + (s + t) * d) > deltaNode)
            s = s + t;
        div *= 2;
        t = (l + div - 1) / div;
    }
    int gamma = i + s * d + min(d, 0);

    int leftIdx, rightIdx;
    int rangeLeft = min(i, j);
    int rangeRight = max(i, j);

    if (rangeLeft == gamma) {
        leftIdx = n - 1 + gamma;
        nodes[leftIdx].firstPrim = gamma;
        nodes[leftIdx].primCount = 1;
        nodes[leftIdx].left = -1;
        nodes[leftIdx].right = -1;
        primIndices[gamma] = sortedIndices[gamma];
        parentIndices[leftIdx] = i;
    } else {
        leftIdx = gamma;
        parentIndices[gamma] = i;
    }

    if (rangeRight == gamma + 1) {
        rightIdx = n - 1 + gamma + 1;
        nodes[rightIdx].firstPrim = gamma + 1;
        nodes[rightIdx].primCount = 1;
        nodes[rightIdx].left = -1;
        nodes[rightIdx].right = -1;
        primIndices[gamma + 1] = sortedIndices[gamma + 1];
        parentIndices[rightIdx] = i;
    } else {
        rightIdx = gamma + 1;
        parentIndices[gamma + 1] = i;
    }

    nodes[i].left = leftIdx;
    nodes[i].right = rightIdx;
    nodes[i].firstPrim = -1;
    nodes[i].primCount = 0;
}

__global__ void computeNodeBoundsKernel(
    BVHNodeDevice* nodes,
    const int* parentIndices,
    int* atomicCounters,
    const AABBDevice* primBounds,
    const int* primIndices,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int leafIdx = n - 1 + idx;
    int primIdx = primIndices[idx];
    nodes[leafIdx].bounds = primBounds[primIdx];

    int current = parentIndices[leafIdx];
    while (current >= 0) {
        int old = atomicAdd(&atomicCounters[current], 1);
        if (old == 0) return;  // Wait for sibling

        int left = nodes[current].left;
        int right = nodes[current].right;
        AABBDevice leftBounds = nodes[left].bounds;
        AABBDevice rightBounds = nodes[right].bounds;

        nodes[current].bounds.min = make_float3(
            fminf(leftBounds.min.x, rightBounds.min.x),
            fminf(leftBounds.min.y, rightBounds.min.y),
            fminf(leftBounds.min.z, rightBounds.min.z));
        nodes[current].bounds.max = make_float3(
            fmaxf(leftBounds.max.x, rightBounds.max.x),
            fmaxf(leftBounds.max.y, rightBounds.max.y),
            fmaxf(leftBounds.max.z, rightBounds.max.z));

        current = parentIndices[current];
    }
}

// ============================================================================
// BVHBuilderGPU Implementation
// ============================================================================

BVHBuilderGPU::BVHBuilderGPU() = default;

BVHBuilderGPU::~BVHBuilderGPU() {
    free();
}

void BVHBuilderGPU::free() {
    if (dVertices_) cudaFree(dVertices_);
    if (dTriangles_) cudaFree(dTriangles_);
    if (dCentroids_) cudaFree(dCentroids_);
    if (dPrimBounds_) cudaFree(dPrimBounds_);
    if (dMortonCodes_) cudaFree(dMortonCodes_);
    if (dSortedIndices_) cudaFree(dSortedIndices_);
    if (dPrimIndices_) cudaFree(dPrimIndices_);
    if (dNodes_) cudaFree(dNodes_);
    if (dParentIndices_) cudaFree(dParentIndices_);
    if (dAtomicCounters_) cudaFree(dAtomicCounters_);

    dVertices_ = nullptr;
    dTriangles_ = nullptr;
    dCentroids_ = nullptr;
    dPrimBounds_ = nullptr;
    dMortonCodes_ = nullptr;
    dSortedIndices_ = nullptr;
    dPrimIndices_ = nullptr;
    dNodes_ = nullptr;
    dParentIndices_ = nullptr;
    dAtomicCounters_ = nullptr;
}

void BVHBuilderGPU::build(const Vector<Vec3>& vertices,
                          const Vector<Triangle>& triangles) {
    free();

    numVertices_ = static_cast<Int>(vertices.size());
    numTriangles_ = static_cast<Int>(triangles.size());
    numInternalNodes_ = numTriangles_ - 1;
    numNodes_ = numTriangles_ + numInternalNodes_;

    if (numTriangles_ == 0) return;

    // Allocate device memory
    cudaMalloc(&dVertices_, numVertices_ * sizeof(Float3));
    cudaMalloc(&dTriangles_, numTriangles_ * sizeof(Int3));
    cudaMalloc(&dCentroids_, numTriangles_ * sizeof(Float3));
    cudaMalloc(&dPrimBounds_, numTriangles_ * sizeof(AABBDevice));
    cudaMalloc(&dMortonCodes_, numTriangles_ * sizeof(UInt));
    cudaMalloc(&dSortedIndices_, numTriangles_ * sizeof(int));
    cudaMalloc(&dPrimIndices_, numTriangles_ * sizeof(int));
    cudaMalloc(&dNodes_, numNodes_ * sizeof(BVHNodeDevice));
    cudaMalloc(&dParentIndices_, numNodes_ * sizeof(int));
    cudaMalloc(&dAtomicCounters_, numInternalNodes_ * sizeof(int));

    // Copy vertices to GPU
    std::vector<Float3> hVertices(numVertices_);
    for (int i = 0; i < numVertices_; ++i) {
        hVertices[i] = make_float3(vertices[i].x(), vertices[i].y(), vertices[i].z());
    }
    cudaMemcpy(dVertices_, hVertices.data(), numVertices_ * sizeof(Float3),
               cudaMemcpyHostToDevice);

    // Copy triangles to GPU
    std::vector<Int3> hTriangles(numTriangles_);
    for (int i = 0; i < numTriangles_; ++i) {
        hTriangles[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
    }
    cudaMemcpy(dTriangles_, hTriangles.data(), numTriangles_ * sizeof(Int3),
               cudaMemcpyHostToDevice);

    cudaMemset(dAtomicCounters_, 0, numInternalNodes_ * sizeof(int));
    cudaMemset(dParentIndices_, -1, numNodes_ * sizeof(int));

    computeCentroidsBounds();
    computeMortonCodes();
    sortMortonCodes();
    buildRadixTree();
    computeNodeBounds();

    cudaDeviceSynchronize();
}

void BVHBuilderGPU::computeCentroidsBounds() {
    Float3* dSceneMin;
    Float3* dSceneMax;
    cudaMalloc(&dSceneMin, sizeof(Float3));
    cudaMalloc(&dSceneMax, sizeof(Float3));

    Float3 initMin = make_float3(1e30f, 1e30f, 1e30f);
    Float3 initMax = make_float3(-1e30f, -1e30f, -1e30f);
    cudaMemcpy(dSceneMin, &initMin, sizeof(Float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dSceneMax, &initMax, sizeof(Float3), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numTriangles_ + blockSize - 1) / blockSize;

    computeCentroidsBoundsKernel<<<numBlocks, blockSize>>>(
        dVertices_, dTriangles_, dCentroids_, dPrimBounds_,
        dSceneMin, dSceneMax, numTriangles_);

    cudaFree(dSceneMin);
    cudaFree(dSceneMax);
}

void BVHBuilderGPU::computeMortonCodes() {
    std::vector<Float3> hCentroids(numTriangles_);
    cudaMemcpy(hCentroids.data(), dCentroids_, numTriangles_ * sizeof(Float3),
               cudaMemcpyDeviceToHost);

    Float3 sceneMin = make_float3(1e30f, 1e30f, 1e30f);
    Float3 sceneMax = make_float3(-1e30f, -1e30f, -1e30f);
    for (const auto& c : hCentroids) {
        sceneMin.x = fminf(sceneMin.x, c.x);
        sceneMin.y = fminf(sceneMin.y, c.y);
        sceneMin.z = fminf(sceneMin.z, c.z);
        sceneMax.x = fmaxf(sceneMax.x, c.x);
        sceneMax.y = fmaxf(sceneMax.y, c.y);
        sceneMax.z = fmaxf(sceneMax.z, c.z);
    }
    Float3 sceneExtent = make_float3(
        sceneMax.x - sceneMin.x,
        sceneMax.y - sceneMin.y,
        sceneMax.z - sceneMin.z);

    int blockSize = 256;
    int numBlocks = (numTriangles_ + blockSize - 1) / blockSize;

    computeMortonCodesKernel<<<numBlocks, blockSize>>>(
        dCentroids_, dMortonCodes_, sceneMin, sceneExtent, numTriangles_);
}

void BVHBuilderGPU::sortMortonCodes() {
    thrust::device_ptr<UInt> mortonPtr(dMortonCodes_);
    thrust::device_ptr<int> indicesPtr(dSortedIndices_);
    thrust::sequence(indicesPtr, indicesPtr + numTriangles_);
    thrust::sort_by_key(mortonPtr, mortonPtr + numTriangles_, indicesPtr);
}

void BVHBuilderGPU::buildRadixTree() {
    int blockSize = 256;
    int numBlocks = (numInternalNodes_ + blockSize - 1) / blockSize;

    buildRadixTreeKernel<<<numBlocks, blockSize>>>(
        dMortonCodes_, dSortedIndices_, dNodes_, dParentIndices_,
        dPrimIndices_, numTriangles_);
}

void BVHBuilderGPU::computeNodeBounds() {
    int blockSize = 256;
    int numBlocks = (numTriangles_ + blockSize - 1) / blockSize;

    computeNodeBoundsKernel<<<numBlocks, blockSize>>>(
        dNodes_, dParentIndices_, dAtomicCounters_,
        dPrimBounds_, dPrimIndices_, numTriangles_);
}

void BVHBuilderGPU::copyToHost(Vector<BVHNodeDevice>& outNodes,
                               Vector<Int>& outPrimIndices) const {
    outNodes.resize(numNodes_);
    outPrimIndices.resize(numTriangles_);
    cudaMemcpy(outNodes.data(), dNodes_, numNodes_ * sizeof(BVHNodeDevice),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(outPrimIndices.data(), dPrimIndices_, numTriangles_ * sizeof(Int),
               cudaMemcpyDeviceToHost);
}

}  // namespace gpu
}  // namespace rigid
