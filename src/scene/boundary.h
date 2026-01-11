/*
 * Simulation Domain Boundaries
 */
#ifndef PHYS3D_BOUNDARIES_HPP
#define PHYS3D_BOUNDARIES_HPP

#include "core/types.h"

namespace phys3d {

/*
 * Boundaries - Axis-aligned box defined by 6 bounding planes
 */
class Boundaries 
{
public:
    enum PlaneId : IntType 
    {
        kMinX = 0,
        kMaxX,
        kMinY,
        kMaxY,
        kMinZ,
        kMaxZ,
        kPlaneTotal
    };

    Boundaries();

    void defineBounds(const Point3& lowerCorner, const Point3& upperCorner);

    [[nodiscard]] HalfSpace3D& planeAt(PlaneId id) { return m_planes[id]; }
    [[nodiscard]] const HalfSpace3D& planeAt(PlaneId id) const { return m_planes[id]; }

    static constexpr IntType planeCount() { return kPlaneTotal; }

    [[nodiscard]] const Point3& lowerCorner() const { return m_lowerCorner; }
    [[nodiscard]] const Point3& upperCorner() const { return m_upperCorner; }

private:
    HalfSpace3D m_planes[kPlaneTotal];
    Point3 m_lowerCorner;
    Point3 m_upperCorner;
};

}  // namespace phys3d

namespace rigid {
    using Environment = phys3d::Boundaries;
}

#endif // PHYS3D_BOUNDARIES_HPP
