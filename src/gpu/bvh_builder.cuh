/**
 * @file bvh_builder.cuh
 * @brief GPU-accelerated BVH construction using CUDA.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "core/common.h"

#ifndef __CUDACC__
// Fallback for non-CUDA compilation
#include <limits>
inline int __clz(unsigned int x) { return (x == 0) ? 32 : __builtin_clz(x); }
#endif

namespace rigid {
namespace gpu {

// ============================================================================
// Device-Side Data Structures
// ============================================================================

/// Device-side AABB
struct AABBDevice {
    Float3 min;
    Float3 max;
};

/// Device-side BVH Node
struct BVHNodeDevice {
    AABBDevice bounds;
    Int left;
    Int right;
    Int firstPrim;
    Int primCount;

    __host__ __device__ bool isLeaf() const { return primCount > 0; }
};

// ============================================================================
// Morton Code Utilities
// ============================================================================

__device__ __host__ inline UInt expandBits(UInt v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline UInt computeMortonCode3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    UInt xx = expandBits(static_cast<UInt>(x));
    UInt yy = expandBits(static_cast<UInt>(y));
    UInt zz = expandBits(static_cast<UInt>(z));
    return (xx << 2) | (yy << 1) | zz;
}

/// Longest common prefix for radix tree construction
__device__ inline int delta(const UInt* sortedCodes, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    UInt ki = sortedCodes[i];
    UInt kj = sortedCodes[j];
    if (ki == kj) return 32 + __clz(i ^ j);
    return __clz(ki ^ kj);
}

// ============================================================================
// BVH Builder Class
// ============================================================================

/**
 * @class BVHBuilderGPU
 * @brief Constructs a BVH on the GPU using LBVH algorithm.
 */
class BVHBuilderGPU {
public:
    BVHBuilderGPU();
    ~BVHBuilderGPU();

    /// Build BVH from mesh data
    void build(const Vector<Vec3>& vertices, const Vector<Triangle>& triangles);

    /// Copy results back to host
    void copyToHost(Vector<BVHNodeDevice>& outNodes, Vector<Int>& outPrimIndices) const;

    /// Free GPU memory
    void free();

    // Device pointer accessors
    [[nodiscard]] BVHNodeDevice* deviceNodes() const { return dNodes_; }
    [[nodiscard]] Int* devicePrimIndices() const { return dPrimIndices_; }
    [[nodiscard]] Float3* deviceVertices() const { return dVertices_; }
    [[nodiscard]] Int3* deviceTriangles() const { return dTriangles_; }

    // Counts
    [[nodiscard]] Int nodeCount() const { return numNodes_; }
    [[nodiscard]] Int triangleCount() const { return numTriangles_; }
    [[nodiscard]] Int vertexCount() const { return numVertices_; }
    [[nodiscard]] Int rootIndex() const { return 0; }

private:
    void computeCentroidsBounds();
    void computeMortonCodes();
    void sortMortonCodes();
    void buildRadixTree();
    void computeNodeBounds();

    // Device memory
    Float3* dVertices_        = nullptr;
    Int3* dTriangles_         = nullptr;
    Float3* dCentroids_       = nullptr;
    AABBDevice* dPrimBounds_  = nullptr;
    UInt* dMortonCodes_       = nullptr;
    Int* dSortedIndices_      = nullptr;
    Int* dPrimIndices_        = nullptr;
    BVHNodeDevice* dNodes_    = nullptr;
    Int* dParentIndices_      = nullptr;
    Int* dAtomicCounters_     = nullptr;

    Int numVertices_      = 0;
    Int numTriangles_     = 0;
    Int numInternalNodes_ = 0;
    Int numNodes_         = 0;
};

}  // namespace gpu
}  // namespace rigid
