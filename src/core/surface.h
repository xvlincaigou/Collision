/*
 * Triangle Surface Geometry with Acceleration Structure Support
 */
#ifndef PHYS3D_GEOMETRY_SURFACE_HPP
#define PHYS3D_GEOMETRY_SURFACE_HPP

#include "types.h"

namespace phys3d {

class SpatialTree;  // Forward declaration

/*
 * TriangleSurface - Stores triangulated geometry data
 */
class TriangleSurface 
{
public:
    TriangleSurface() = default;
    explicit TriangleSurface(TextType identifier);
    ~TriangleSurface();

    TriangleSurface(const TriangleSurface&) = delete;
    TriangleSurface& operator=(const TriangleSurface&) = delete;
    TriangleSurface(TriangleSurface&&) noexcept;
    TriangleSurface& operator=(TriangleSurface&&) noexcept;

    /* I/O Methods */
    bool parseWavefrontFile(const TextType& filepath, bool constructTree = true);
    void purge();

    /* Queries */
    [[nodiscard]] const TextType& identifier() const { return m_identifier; }
    [[nodiscard]] const DynArray<Point3>& pointCloud() const { return m_pointCloud; }
    [[nodiscard]] const DynArray<Triplet3i>& faceIndices() const { return m_faceIndices; }
    [[nodiscard]] const DynArray<Point3>& faceNormals() const { return m_faceNormals; }
    [[nodiscard]] const BoundingBox3D& localExtent() const { return m_localExtent; }

    [[nodiscard]] IntType pointCount() const { return static_cast<IntType>(m_pointCloud.size()); }
    [[nodiscard]] IntType faceCount() const { return static_cast<IntType>(m_faceIndices.size()); }
    [[nodiscard]] bool isBlank() const { return m_pointCloud.empty(); }

    /* Spatial Tree Access */
    [[nodiscard]] bool hasAccelerator() const;
    [[nodiscard]] const SpatialTree& accelerator() const;
    void reconstructAccelerator();
    void flagAcceleratorStale() { m_treeStale = true; }

private:
    void recomputeExtent();

    TextType m_identifier;
    DynArray<Point3> m_pointCloud;
    DynArray<Triplet3i> m_faceIndices;
    DynArray<Point3> m_faceNormals;

    BoundingBox3D m_localExtent;
    bool m_extentStale = true;

    SolePtr<SpatialTree> m_spatialTree;
    bool m_treeStale = true;
};

}  // namespace phys3d

namespace rigid {
    using Mesh = phys3d::TriangleSurface;
    using BVH = phys3d::SpatialTree;
}

#endif // PHYS3D_GEOMETRY_SURFACE_HPP
