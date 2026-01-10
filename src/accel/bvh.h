/**
 * @file bvh.h
 * @brief Bounding Volume Hierarchy for efficient collision detection.
 */
#pragma once

#include "core/common.h"

#include <algorithm>
#include <cstdint>

namespace rigid {

// Forward declaration
namespace gpu {
class BVHBuilderGPU;
}

/**
 * @struct BVHNode
 * @brief A node in the BVH tree.
 */
struct BVHNode {
    AABB bounds;
    Int left       = -1;  ///< Left child index (-1 if none)
    Int right      = -1;  ///< Right child index (-1 if none)
    Int firstPrim  = -1;  ///< First primitive index (for leaves)
    Int primCount  = 0;   ///< Number of primitives (>0 means leaf)

    [[nodiscard]] bool isLeaf() const { return primCount > 0; }
};

/**
 * @class BVH
 * @brief Bounding Volume Hierarchy for triangle meshes.
 *
 * Supports both CPU and GPU construction/traversal.
 */
class BVH {
public:
    BVH();
    ~BVH();

    // Non-copyable but movable
    BVH(const BVH&) = delete;
    BVH& operator=(const BVH&) = delete;
    BVH(BVH&&) noexcept;
    BVH& operator=(BVH&&) noexcept;

    // ========================================================================
    // Construction
    // ========================================================================

    /// Build BVH on CPU
    void build(const Vector<Vec3>& vertices, const Vector<Triangle>& triangles);

    /// Build BVH on GPU (falls back to CPU if CUDA not available)
    void buildGPU(const Vector<Vec3>& vertices, const Vector<Triangle>& triangles);

    /// Clear all BVH data
    void clear();

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] bool isBuilt() const { return rootIndex_ >= 0; }
    [[nodiscard]] Int rootIndex() const { return rootIndex_; }
    [[nodiscard]] Int nodeCount() const { return static_cast<Int>(nodes_.size()); }

    [[nodiscard]] const BVHNode& node(Int index) const { return nodes_[index]; }
    [[nodiscard]] const Vector<BVHNode>& nodes() const { return nodes_; }
    [[nodiscard]] const Vector<Int>& primitiveIndices() const { return primIndices_; }

    // ========================================================================
    // GPU Support
    // ========================================================================

    [[nodiscard]] bool hasGPUData() const { return gpuBuilder_ != nullptr; }
    [[nodiscard]] gpu::BVHBuilderGPU* gpuBuilder() const { return gpuBuilder_.get(); }

private:
    /// Primitive record for BVH construction
    struct PrimRecord {
        AABB bounds;
        Vec3 centroid;
        Int triangleIndex = -1;
    };

    Int buildNode(Int first, Int last, Int depth);
    AABB accumulateBounds(Int first, Int last) const;
    void computeCentroidBounds(Int first, Int last, Vec3& minOut, Vec3& maxOut) const;
    Int partitionPrimitives(Int first, Int last, Int axis, Float splitValue);

    Vector<BVHNode> nodes_;
    Vector<Int> primIndices_;
    Vector<PrimRecord> prims_;

    Int rootIndex_       = -1;
    Int maxPrimsPerLeaf_ = 4;
    Int maxDepth_        = 64;
    Float centroidEps_   = 1e-4f;

    UniquePtr<gpu::BVHBuilderGPU> gpuBuilder_;
};

}  // namespace rigid
