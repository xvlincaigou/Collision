/*
 * Implementation: Boundaries class
 */
#include "environment.h"

namespace phys3d {

Boundaries::Boundaries() 
{
    defineBounds(Point3(static_cast<RealType>(-10), static_cast<RealType>(-10), static_cast<RealType>(-10)),
                 Point3(static_cast<RealType>(10), static_cast<RealType>(10), static_cast<RealType>(10)));
}

void Boundaries::defineBounds(const Point3& lowerCorner, const Point3& upperCorner) 
{
    m_lowerCorner = lowerCorner;
    m_upperCorner = upperCorner;

    m_planes[kMinX].direction = Point3(static_cast<RealType>(1), static_cast<RealType>(0), static_cast<RealType>(0));
    m_planes[kMinX].distance = lowerCorner.x();

    m_planes[kMaxX].direction = Point3(static_cast<RealType>(-1), static_cast<RealType>(0), static_cast<RealType>(0));
    m_planes[kMaxX].distance = -upperCorner.x();

    m_planes[kMinY].direction = Point3(static_cast<RealType>(0), static_cast<RealType>(1), static_cast<RealType>(0));
    m_planes[kMinY].distance = lowerCorner.y();

    m_planes[kMaxY].direction = Point3(static_cast<RealType>(0), static_cast<RealType>(-1), static_cast<RealType>(0));
    m_planes[kMaxY].distance = -upperCorner.y();

    m_planes[kMinZ].direction = Point3(static_cast<RealType>(0), static_cast<RealType>(0), static_cast<RealType>(1));
    m_planes[kMinZ].distance = lowerCorner.z();

    m_planes[kMaxZ].direction = Point3(static_cast<RealType>(0), static_cast<RealType>(0), static_cast<RealType>(-1));
    m_planes[kMaxZ].distance = -upperCorner.z();
}

}  // namespace phys3d
