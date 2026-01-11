/*
 * Physical Material and Mass Properties
 */
#ifndef PHYS3D_MATERIAL_PROPS_HPP
#define PHYS3D_MATERIAL_PROPS_HPP

#include "core/common.h"

namespace phys3d {

/*
 * MaterialProperties - Defines physical attributes of an entity
 */
struct MaterialProperties 
{
    RealType totalMass       = static_cast<RealType>(1);
    RealType inverseMass     = static_cast<RealType>(1);
    RealType linearDrag      = static_cast<RealType>(0.4);
    RealType angularDrag     = static_cast<RealType>(0.1);
    RealType bounciness      = static_cast<RealType>(0);
    RealType surfaceFriction = static_cast<RealType>(0.5);

    Matrix33 localInertia    = Matrix33::Identity();
    Matrix33 localInertiaInv = Matrix33::Identity();
    Point3   massCentroid    = Point3::Zero();

    /* Mutators */
    void assignMass(RealType value) 
    {
        totalMass = ensurePositive(value);
        inverseMass = static_cast<RealType>(1) / totalMass;
    }

    void assignLocalInertia(const Matrix33& value) 
    {
        localInertia = value;
        localInertiaInv = safeInvert(localInertia);
    }

    void assignMassCentroid(const Point3& value) 
    {
        massCentroid = value;
    }

    void prepare() 
    {
        assignMass(totalMass);
        assignLocalInertia(localInertia);
    }

    /* Validation */
    [[nodiscard]] bool wellDefined() const 
    {
        bool massOk = totalMass > static_cast<RealType>(0) && std::isfinite(totalMass);
        bool inertiaOk = localInertia.allFinite() && localInertiaInv.allFinite();
        return massOk && inertiaOk;
    }

    [[nodiscard]] bool isMovable() const 
    {
        return inverseMass > static_cast<RealType>(0);
    }

    /* Factory Methods */
    static MaterialProperties createImmovable() 
    {
        MaterialProperties props;
        props.totalMass = std::numeric_limits<RealType>::infinity();
        props.inverseMass = static_cast<RealType>(0);
        return props;
    }

    static MaterialProperties createDefault(RealType mass = static_cast<RealType>(1)) 
    {
        MaterialProperties props;
        props.assignMass(mass);
        return props;
    }
};

}  // namespace phys3d

namespace rigid {
    using BodyProperties = phys3d::MaterialProperties;
}

#endif // PHYS3D_MATERIAL_PROPS_HPP
