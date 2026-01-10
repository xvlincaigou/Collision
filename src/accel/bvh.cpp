/**
 * @file bvh.cpp
 * @brief Implementation of the BVH class.
 */
#include "bvh.h"

#include <numeric>

#if defined(RIGID_USE_CUDA)
#include "gpu/bvh_builder.cuh"
#endif

namespace rigid {

BVH::BVH() = default;
BVH::~BVH() = default;

BVH::BVH(BVH&&) noexcept = default;
BVH& BVH::operator=(BVH&&) noexcept = default;

void BVH::clear() {
    nodes_.clear();
    primIndices_.clear();
    prims_.clear();
    rootIndex_ = -1;
    gpuBuilder_.reset();
}

void BVH::build(const Vector<Vec3>& vertices, const Vector<Triangle>& triangles) {
    clear();

    if (triangles.empty()) {
        return;
    }

    const Int primCount = static_cast<Int>(triangles.size());
    prims_.resize(primCount);
    primIndices_.resize(primCount);
    std::iota(primIndices_.begin(), primIndices_.end(), 0);

    // Compute primitive bounds and centroids
    for (Int i = 0; i < primCount; ++i) {
        const auto& tri = triangles[i];
        const Vec3& v0 = vertices[tri.x()];
        const Vec3& v1 = vertices[tri.y()];
        const Vec3& v2 = vertices[tri.z()];

        PrimRecord& rec = prims_[i];
        rec.bounds.reset();
        rec.bounds.expand(v0);
        rec.bounds.expand(v1);
        rec.bounds.expand(v2);
        rec.centroid = (v0 + v1 + v2) / 3.0f;
        rec.triangleIndex = i;
    }

    // Reserve space for nodes (at most 2n-1 nodes for n primitives)
    nodes_.reserve(primCount * 2);

    // Build tree recursively
    rootIndex_ = buildNode(0, primCount, 0);
}

Int BVH::buildNode(Int first, Int last, Int depth) {
    Int nodeIndex = static_cast<Int>(nodes_.size());
    nodes_.emplace_back();
    BVHNode& node = nodes_.back();

    node.bounds = accumulateBounds(first, last);
    node.firstPrim = first;
    node.primCount = last - first;
    node.left = -1;
    node.right = -1;

    // Check if should create leaf
    const bool forceLeaf = (node.primCount <= maxPrimsPerLeaf_) || (depth >= maxDepth_);
    if (forceLeaf) {
        return nodeIndex;
    }

    // Find split axis (largest centroid extent)
    Vec3 centroidMin, centroidMax;
    computeCentroidBounds(first, last, centroidMin, centroidMax);
    Vec3 centroidExtent = centroidMax - centroidMin;

    Int axis = 0;
    if (centroidExtent.y() > centroidExtent.x()) axis = 1;
    if (centroidExtent.z() > centroidExtent[axis]) axis = 2;

    // Check if centroids are too close
    if (centroidExtent[axis] <= centroidEps_) {
        return nodeIndex;
    }

    // Split at midpoint
    Float splitValue = centroidMin[axis] + centroidExtent[axis] * 0.5f;
    Int mid = partitionPrimitives(first, last, axis, splitValue);

    // Check if partition failed
    if (mid == first || mid == last) {
        return nodeIndex;
    }

    // Create internal node
    node.firstPrim = -1;
    node.primCount = 0;
    node.left = buildNode(first, mid, depth + 1);
    node.right = buildNode(mid, last, depth + 1);

    return nodeIndex;
}

AABB BVH::accumulateBounds(Int first, Int last) const {
    AABB result;
    for (Int i = first; i < last; ++i) {
        const AABB& primBounds = prims_[primIndices_[i]].bounds;
        result.merge(primBounds);
    }
    return result;
}

void BVH::computeCentroidBounds(Int first, Int last, Vec3& minOut, Vec3& maxOut) const {
    minOut = positiveInfinity();
    maxOut = negativeInfinity();

    for (Int i = first; i < last; ++i) {
        const Vec3& centroid = prims_[primIndices_[i]].centroid;
        minOut = minOut.cwiseMin(centroid);
        maxOut = maxOut.cwiseMax(centroid);
    }
}

Int BVH::partitionPrimitives(Int first, Int last, Int axis, Float splitValue) {
    auto beginIt = primIndices_.begin() + first;
    auto endIt = primIndices_.begin() + last;

    auto predicate = [&](Int primIndex) {
        return prims_[primIndex].centroid[axis] < splitValue;
    };

    auto midIt = std::partition(beginIt, endIt, predicate);
    return static_cast<Int>(midIt - primIndices_.begin());
}

#if defined(RIGID_USE_CUDA)

void BVH::buildGPU(const Vector<Vec3>& vertices, const Vector<Triangle>& triangles) {
    clear();

    if (triangles.empty()) {
        return;
    }

    gpuBuilder_ = std::make_unique<gpu::BVHBuilderGPU>();
    gpuBuilder_->build(vertices, triangles);

    // Copy results back to host
    Vector<gpu::BVHNodeDevice> gpuNodes;
    gpuBuilder_->copyToHost(gpuNodes, primIndices_);

    // Convert GPU nodes to CPU format
    nodes_.resize(gpuNodes.size());
    for (size_t i = 0; i < gpuNodes.size(); ++i) {
        const auto& gn = gpuNodes[i];
        nodes_[i].bounds.min_pt = Vec3(gn.bounds.min.x, gn.bounds.min.y, gn.bounds.min.z);
        nodes_[i].bounds.max_pt = Vec3(gn.bounds.max.x, gn.bounds.max.y, gn.bounds.max.z);
        nodes_[i].left = gn.left;
        nodes_[i].right = gn.right;
        nodes_[i].firstPrim = gn.firstPrim;
        nodes_[i].primCount = gn.primCount;
    }

    rootIndex_ = 0;
}

#else

void BVH::buildGPU(const Vector<Vec3>& vertices, const Vector<Triangle>& triangles) {
    // Fallback to CPU build
    build(vertices, triangles);
}

#endif

}  // namespace rigid
