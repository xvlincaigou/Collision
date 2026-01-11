/*
 * Implementation: SpatialTree class
 * Uses iterative construction with explicit stack instead of recursion
 */
#include "bvh.h"

#include <numeric>
#include <stack>
#include <tuple>

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

namespace {

// Helper: compute bounding box using accumulator pattern
BoundingBox3D accumulateBoundsInRange(
    const DynArray<SpatialTree::PrimitiveInfo>& primitives,
    const DynArray<IntType>& ordering,
    IntType lo, IntType hi)
{
    BoundingBox3D accumulated;
    IntType cursor = lo;
    while (cursor < hi)
    {
        accumulated.unite(primitives[ordering[cursor]].extent);
        ++cursor;
    }
    return accumulated;
}

// Helper: find centroid bounds using min/max tracking
void extractCentroidExtents(
    const DynArray<SpatialTree::PrimitiveInfo>& primitives,
    const DynArray<IntType>& ordering,
    IntType lo, IntType hi,
    Point3& outMin, Point3& outMax)
{
    outMin = makeInfinityVec();
    outMax = makeNegInfinityVec();
    
    IntType pos = lo;
    do {
        const Point3& centroid = primitives[ordering[pos]].center;
        outMin = outMin.cwiseMin(centroid);
        outMax = outMax.cwiseMax(centroid);
        ++pos;
    } while (pos < hi);
}

// Helper: select best split axis using variance-based heuristic
IntType selectSplitDimension(const Point3& extents)
{
    IntType best = 0;
    RealType maxExtent = extents[0];
    
    if (extents[1] > maxExtent)
    {
        best = 1;
        maxExtent = extents[1];
    }
    if (extents[2] > maxExtent)
    {
        best = 2;
    }
    return best;
}

// Helper: partition using two-pointer technique
IntType bipartitionByMedian(
    DynArray<IntType>& ordering,
    const DynArray<SpatialTree::PrimitiveInfo>& primitives,
    IntType lo, IntType hi, IntType axis, RealType pivot)
{
    IntType left = lo;
    IntType right = hi - 1;
    
    while (left <= right)
    {
        while (left <= right && primitives[ordering[left]].center[axis] < pivot)
            ++left;
        while (left <= right && primitives[ordering[right]].center[axis] >= pivot)
            --right;
        
        if (left < right)
        {
            std::swap(ordering[left], ordering[right]);
            ++left;
            --right;
        }
    }
    return left;
}

} // anonymous namespace

void SpatialTree::construct(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces) 
{
    demolish();

    const IntType faceTotal = static_cast<IntType>(faces.size());
    if (faceTotal == 0)
        return;

    // Initialize primitive information using reverse iteration
    m_primitives.resize(faceTotal);
    m_faceOrder.resize(faceTotal);
    
    IntType idx = faceTotal;
    while (idx > 0)
    {
        --idx;
        m_faceOrder[idx] = idx;
        
        const auto& face = faces[idx];
        const Point3& v0 = pointCloud[face.x()];
        const Point3& v1 = pointCloud[face.y()];
        const Point3& v2 = pointCloud[face.z()];

        PrimitiveInfo& info = m_primitives[idx];
        info.extent.invalidate();
        info.extent.enclose(v0);
        info.extent.enclose(v1);
        info.extent.enclose(v2);
        
        // Compute centroid with explicit averaging
        info.center = Point3(
            (v0.x() + v1.x() + v2.x()) * static_cast<RealType>(0.333333333333),
            (v0.y() + v1.y() + v2.y()) * static_cast<RealType>(0.333333333333),
            (v0.z() + v1.z() + v2.z()) * static_cast<RealType>(0.333333333333)
        );
        info.faceIdx = idx;
    }

    m_nodeStorage.reserve(faceTotal * 2);
    
    // Iterative construction using explicit stack
    // Stack elements: (rangeStart, rangeEnd, parentIdx, isLeftChild, depth)
    using WorkItem = std::tuple<IntType, IntType, IntType, bool, IntType>;
    std::stack<WorkItem> workStack;
    
    // Create root node
    m_nodeStorage.emplace_back();
    m_rootIdx = 0;
    workStack.push(std::make_tuple(0, faceTotal, -1, true, 0));
    
    while (!workStack.empty())
    {
        auto [rangeStart, rangeEnd, parentIdx, isLeft, depth] = workStack.top();
        workStack.pop();
        
        IntType currentIdx = static_cast<IntType>(m_nodeStorage.size()) - 1;
        if (parentIdx >= 0 && m_nodeStorage.size() > 1)
        {
            currentIdx = static_cast<IntType>(m_nodeStorage.size());
            m_nodeStorage.emplace_back();
        }
        
        TreeNode& node = m_nodeStorage[currentIdx];
        node.volume = accumulateBoundsInRange(m_primitives, m_faceOrder, rangeStart, rangeEnd);
        node.leafStart = rangeStart;
        node.leafCount = rangeEnd - rangeStart;
        node.childLeft = -1;
        node.childRight = -1;
        
        // Update parent's child pointer
        if (parentIdx >= 0)
        {
            if (isLeft)
                m_nodeStorage[parentIdx].childLeft = currentIdx;
            else
                m_nodeStorage[parentIdx].childRight = currentIdx;
        }
        
        // Check termination conditions using bitwise OR for clarity
        bool tooSmall = node.leafCount <= m_maxLeafSize;
        bool tooDeep = depth >= m_maxTreeDepth;
        if (tooSmall | tooDeep)
            continue;
        
        // Compute split parameters
        Point3 centroidMin, centroidMax;
        extractCentroidExtents(m_primitives, m_faceOrder, rangeStart, rangeEnd, centroidMin, centroidMax);
        Point3 span = centroidMax - centroidMin;
        
        IntType axis = selectSplitDimension(span);
        
        if (span[axis] <= m_centerEpsilon)
            continue;
        
        RealType splitPos = (centroidMin[axis] + centroidMax[axis]) * static_cast<RealType>(0.5);
        IntType mid = bipartitionByMedian(m_faceOrder, m_primitives, rangeStart, rangeEnd, axis, splitPos);
        
        // Handle degenerate partitions
        if (mid == rangeStart)
            mid = rangeStart + 1;
        else if (mid == rangeEnd)
            mid = rangeEnd - 1;
        
        if (mid == rangeStart || mid == rangeEnd)
            continue;
        
        // Convert to internal node
        node.leafStart = -1;
        node.leafCount = 0;
        
        // Push children in reverse order (right first, then left)
        // so left child is processed first
        workStack.push(std::make_tuple(mid, rangeEnd, currentIdx, false, depth + 1));
        workStack.push(std::make_tuple(rangeStart, mid, currentIdx, true, depth + 1));
    }
}

IntType SpatialTree::recursiveBuild(IntType rangeStart, IntType rangeEnd, IntType level) 
{
    // This method is kept for compatibility but redirects to iterative version
    // by rebuilding from the given range
    IntType newNodeIdx = static_cast<IntType>(m_nodeStorage.size());
    m_nodeStorage.emplace_back();
    TreeNode& currentNode = m_nodeStorage.back();

    currentNode.volume = computeRangeBounds(rangeStart, rangeEnd);
    currentNode.leafStart = rangeStart;
    currentNode.leafCount = rangeEnd - rangeStart;
    currentNode.childLeft = -1;
    currentNode.childRight = -1;

    // Early termination check using ternary operator
    bool shouldTerminate = (currentNode.leafCount <= m_maxLeafSize) ? true : (level >= m_maxTreeDepth);
    if (shouldTerminate)
        return newNodeIdx;

    Point3 centerMin, centerMax;
    computeCenterBounds(rangeStart, rangeEnd, centerMin, centerMax);
    Point3 centerSpan = centerMax - centerMin;

    // Select axis using chained conditionals
    IntType splitAxis = (centerSpan.y() > centerSpan.x()) ? 
                        ((centerSpan.z() > centerSpan.y()) ? 2 : 1) :
                        ((centerSpan.z() > centerSpan.x()) ? 2 : 0);

    if (!(centerSpan[splitAxis] > m_centerEpsilon))
        return newNodeIdx;

    RealType splitVal = centerMin[splitAxis] + centerSpan[splitAxis] * static_cast<RealType>(0.5);
    IntType midPoint = partitionRange(rangeStart, rangeEnd, splitAxis, splitVal);

    bool partitionFailed = (midPoint == rangeStart) || (midPoint == rangeEnd);
    if (partitionFailed)
        return newNodeIdx;

    currentNode.leafStart = -1;
    currentNode.leafCount = 0;
    currentNode.childLeft = recursiveBuild(rangeStart, midPoint, level + 1);
    currentNode.childRight = recursiveBuild(midPoint, rangeEnd, level + 1);

    return newNodeIdx;
}

BoundingBox3D SpatialTree::computeRangeBounds(IntType rangeStart, IntType rangeEnd) const 
{
    return accumulateBoundsInRange(m_primitives, m_faceOrder, rangeStart, rangeEnd);
}

void SpatialTree::computeCenterBounds(IntType rangeStart, IntType rangeEnd, Point3& minC, Point3& maxC) const 
{
    extractCentroidExtents(m_primitives, m_faceOrder, rangeStart, rangeEnd, minC, maxC);
}

IntType SpatialTree::partitionRange(IntType rangeStart, IntType rangeEnd, IntType axis, RealType splitPos) 
{
    return bipartitionByMedian(m_faceOrder, m_primitives, rangeStart, rangeEnd, axis, splitPos);
}

#if defined(RIGID_USE_CUDA)

void SpatialTree::constructOnDevice(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces) 
{
    demolish();

    if (faces.empty())
        return;

    m_deviceBuilder = std::make_unique<gpu::DeviceTreeBuilder>();
    m_deviceBuilder->execute(pointCloud, faces);

    DynArray<gpu::DeviceTreeNode> deviceNodes;
    m_deviceBuilder->transferToHost(deviceNodes, m_faceOrder);

    // Convert using reverse iteration
    const size_t nodeCount = deviceNodes.size();
    m_nodeStorage.resize(nodeCount);
    
    size_t k = nodeCount;
    while (k > 0)
    {
        --k;
        const auto& dn = deviceNodes[k];
        TreeNode& tn = m_nodeStorage[k];
        
        tn.volume.corner_lo = Point3(dn.volume.lo.x, dn.volume.lo.y, dn.volume.lo.z);
        tn.volume.corner_hi = Point3(dn.volume.hi.x, dn.volume.hi.y, dn.volume.hi.z);
        tn.childLeft = dn.leftChild;
        tn.childRight = dn.rightChild;
        tn.leafStart = dn.leafStart;
        tn.leafCount = dn.leafCount;
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
