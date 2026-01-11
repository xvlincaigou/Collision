/*
 * Contact Information for Collision Response
 */
#ifndef PHYS3D_CONTACT_INFO_HPP
#define PHYS3D_CONTACT_INFO_HPP

#include "core/common.h"

namespace phys3d {

/*
 * ContactPoint - Stores collision contact data
 */
struct ContactPoint 
{
    Point3   location;
    Point3   direction;
    RealType depth;
    IntType  entityA;
    IntType  entityB;

    ContactPoint()
        : location(Point3::Zero())
        , direction(Point3::Zero())
        , depth(static_cast<RealType>(0))
        , entityA(-1)
        , entityB(-1) 
    {}
};

}  // namespace phys3d

namespace rigid {
    using Contact = phys3d::ContactPoint;
}

#endif // PHYS3D_CONTACT_INFO_HPP
