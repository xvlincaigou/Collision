/*
 * Hierarchical Bounding Volume Acceleration Structure
 */
#ifndef PHYS3D_SPATIAL_TREE_HPP
#define PHYS3D_SPATIAL_TREE_HPP

#include "core/types.h"

#include <algorithm>
#include <cstdint>

namespace phys3d {

namespace gpu { class DeviceTreeBuilder; }

/*
 * TreeNode - Single node in the spatial hierarchy
 */
struct TreeNode 
{
    BoundingBox3D volume;
    IntType childLeft   = -1;
    IntType childRight  = -1;
    IntType leafStart   = -1;
    IntType leafCount   = 0;

    [[nodiscard]] bool isTerminal() const { return leafCount > 0; }
};

/*
 * SpatialTree - Binary tree of bounding volumes for mesh triangles
 */
class SpatialTree 
{
public:
    SpatialTree();
    ~SpatialTree();

    SpatialTree(const SpatialTree&) = delete;
    SpatialTree& operator=(const SpatialTree&) = delete;
    SpatialTree(SpatialTree&&) noexcept;
    SpatialTree& operator=(SpatialTree&&) noexcept;

    /* Construction */
    void construct(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces);
    void constructOnDevice(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces);
    void demolish();

    /* Queries */
    [[nodiscard]] bool constructed() const { return m_rootIdx >= 0; }
    [[nodiscard]] IntType rootIdx() const { return m_rootIdx; }
    [[nodiscard]] IntType nodeTotal() const { return static_cast<IntType>(m_nodeStorage.size()); }

    [[nodiscard]] const TreeNode& nodeAt(IntType idx) const { return m_nodeStorage[idx]; }
    [[nodiscard]] const DynArray<TreeNode>& allNodes() const { return m_nodeStorage; }
    [[nodiscard]] const DynArray<IntType>& faceOrdering() const { return m_faceOrder; }

    /* Device Data */
    [[nodiscard]] bool hasDeviceData() const { return m_deviceBuilder != nullptr; }
    [[nodiscard]] gpu::DeviceTreeBuilder* deviceBuilder() const { return m_deviceBuilder.get(); }

public:
    struct PrimitiveInfo 
    {
        BoundingBox3D extent;
        Point3 center;
        IntType faceIdx = -1;
    };

private:

    IntType recursiveBuild(IntType rangeStart, IntType rangeEnd, IntType level);
    BoundingBox3D computeRangeBounds(IntType rangeStart, IntType rangeEnd) const;
    void computeCenterBounds(IntType rangeStart, IntType rangeEnd, Point3& minC, Point3& maxC) const;
    IntType partitionRange(IntType rangeStart, IntType rangeEnd, IntType axis, RealType splitPos);

    DynArray<TreeNode> m_nodeStorage;
    DynArray<IntType> m_faceOrder;
    DynArray<PrimitiveInfo> m_primitives;

    IntType m_rootIdx         = -1;
    IntType m_maxLeafSize     = 4;
    IntType m_maxTreeDepth    = 64;
    RealType m_centerEpsilon  = static_cast<RealType>(1e-4);

    SolePtr<gpu::DeviceTreeBuilder> m_deviceBuilder;
};

}  // namespace phys3d

namespace rigid {
    using BVHNode = phys3d::TreeNode;
    using BVH = phys3d::SpatialTree;
}

#endif // PHYS3D_SPATIAL_TREE_HPP
