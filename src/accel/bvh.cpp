/*
 * Implementation: SpatialTree class
 */
#include "bvh.h"

#include <numeric>

#if defined(RIGID_USE_CUDA)
#include "gpu/bvh_builder.cuh"
#endif

namespace phys3d {

SpatialTree::SpatialTree() = default;
SpatialTree::~SpatialTree() = default;

SpatialTree::SpatialTree(SpatialTree&&) noexcept = default;
SpatialTree& SpatialTree::operator=(SpatialTree&&) noexcept = default;

void SpatialTree::demolish() 
{
    m_nodeStorage.clear();
    m_faceOrder.clear();
    m_primitives.clear();
    m_rootIdx = -1;
    m_deviceBuilder.reset();
}

void SpatialTree::construct(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces) 
{
    demolish();

    if (faces.empty()) 
    {
        return;
    }

    const IntType faceTotal = static_cast<IntType>(faces.size());
    m_primitives.resize(faceTotal);
    m_faceOrder.resize(faceTotal);
    std::iota(m_faceOrder.begin(), m_faceOrder.end(), 0);

    for (IntType idx = 0; idx < faceTotal; ++idx) 
    {
        const auto& face = faces[idx];
        const Point3& p0 = pointCloud[face.x()];
        const Point3& p1 = pointCloud[face.y()];
        const Point3& p2 = pointCloud[face.z()];

        PrimitiveInfo& info = m_primitives[idx];
        info.extent.invalidate();
        info.extent.enclose(p0);
        info.extent.enclose(p1);
        info.extent.enclose(p2);
        info.center = (p0 + p1 + p2) / static_cast<RealType>(3);
        info.faceIdx = idx;
    }

    m_nodeStorage.reserve(faceTotal * 2);
    m_rootIdx = recursiveBuild(0, faceTotal, 0);
}

IntType SpatialTree::recursiveBuild(IntType rangeStart, IntType rangeEnd, IntType level) 
{
    IntType newNodeIdx = static_cast<IntType>(m_nodeStorage.size());
    m_nodeStorage.emplace_back();
    TreeNode& currentNode = m_nodeStorage.back();

    currentNode.volume = computeRangeBounds(rangeStart, rangeEnd);
    currentNode.leafStart = rangeStart;
    currentNode.leafCount = rangeEnd - rangeStart;
    currentNode.childLeft = -1;
    currentNode.childRight = -1;

    bool shouldTerminate = (currentNode.leafCount <= m_maxLeafSize) || (level >= m_maxTreeDepth);
    if (shouldTerminate) 
    {
        return newNodeIdx;
    }

    Point3 centerMin, centerMax;
    computeCenterBounds(rangeStart, rangeEnd, centerMin, centerMax);
    Point3 centerSpan = centerMax - centerMin;

    IntType splitAxis = 0;
    if (centerSpan.y() > centerSpan.x()) splitAxis = 1;
    if (centerSpan.z() > centerSpan[splitAxis]) splitAxis = 2;

    if (centerSpan[splitAxis] <= m_centerEpsilon) 
    {
        return newNodeIdx;
    }

    RealType splitVal = centerMin[splitAxis] + centerSpan[splitAxis] * static_cast<RealType>(0.5);
    IntType midPoint = partitionRange(rangeStart, rangeEnd, splitAxis, splitVal);

    if (midPoint == rangeStart || midPoint == rangeEnd) 
    {
        return newNodeIdx;
    }

    currentNode.leafStart = -1;
    currentNode.leafCount = 0;
    currentNode.childLeft = recursiveBuild(rangeStart, midPoint, level + 1);
    currentNode.childRight = recursiveBuild(midPoint, rangeEnd, level + 1);

    return newNodeIdx;
}

BoundingBox3D SpatialTree::computeRangeBounds(IntType rangeStart, IntType rangeEnd) const 
{
    BoundingBox3D result;
    for (IntType idx = rangeStart; idx < rangeEnd; ++idx) 
    {
        const BoundingBox3D& primBounds = m_primitives[m_faceOrder[idx]].extent;
        result.unite(primBounds);
    }
    return result;
}

void SpatialTree::computeCenterBounds(IntType rangeStart, IntType rangeEnd, Point3& minC, Point3& maxC) const 
{
    minC = makeInfinityVec();
    maxC = makeNegInfinityVec();

    for (IntType idx = rangeStart; idx < rangeEnd; ++idx) 
    {
        const Point3& c = m_primitives[m_faceOrder[idx]].center;
        minC = minC.cwiseMin(c);
        maxC = maxC.cwiseMax(c);
    }
}

IntType SpatialTree::partitionRange(IntType rangeStart, IntType rangeEnd, IntType axis, RealType splitPos) 
{
    auto startIter = m_faceOrder.begin() + rangeStart;
    auto endIter = m_faceOrder.begin() + rangeEnd;

    auto comparator = [&](IntType faceIdx) {
        return m_primitives[faceIdx].center[axis] < splitPos;
    };

    auto midIter = std::partition(startIter, endIter, comparator);
    return static_cast<IntType>(midIter - m_faceOrder.begin());
}

#if defined(RIGID_USE_CUDA)

void SpatialTree::constructOnDevice(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces) 
{
    demolish();

    if (faces.empty()) 
    {
        return;
    }

    m_deviceBuilder = std::make_unique<gpu::DeviceTreeBuilder>();
    m_deviceBuilder->execute(pointCloud, faces);

    DynArray<gpu::DeviceTreeNode> deviceNodes;
    m_deviceBuilder->transferToHost(deviceNodes, m_faceOrder);

    m_nodeStorage.resize(deviceNodes.size());
    for (size_t k = 0; k < deviceNodes.size(); ++k) 
    {
        const auto& dn = deviceNodes[k];
        m_nodeStorage[k].volume.corner_lo = Point3(dn.volume.lo.x, dn.volume.lo.y, dn.volume.lo.z);
        m_nodeStorage[k].volume.corner_hi = Point3(dn.volume.hi.x, dn.volume.hi.y, dn.volume.hi.z);
        m_nodeStorage[k].childLeft = dn.leftChild;
        m_nodeStorage[k].childRight = dn.rightChild;
        m_nodeStorage[k].leafStart = dn.leafStart;
        m_nodeStorage[k].leafCount = dn.leafCount;
    }

    m_rootIdx = 0;
}

#else

void SpatialTree::constructOnDevice(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces) 
{
    construct(pointCloud, faces);
}

#endif

}  // namespace phys3d
