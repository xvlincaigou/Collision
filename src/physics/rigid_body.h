/*
 * Dynamic Entity with Geometry and Physical State
 */
#ifndef PHYS3D_DYNAMIC_ENTITY_HPP
#define PHYS3D_DYNAMIC_ENTITY_HPP

#include "body_properties.h"
#include "core/common.h"
#include "core/mesh.h"

namespace phys3d {

/*
 * EntityState - Kinematic state of a dynamic entity
 */
struct EntityState 
{
    Point3    translation  = Point3::Zero();
    Rotation4 orientation  = Rotation4::Identity();
    Point3    velocity     = Point3::Zero();
    Point3    angularRate  = Point3::Zero();
    RealType  scaleFactor  = static_cast<RealType>(1);

    [[nodiscard]] Matrix33 orientationMatrix() const 
    {
        return orientation.toRotationMatrix();
    }

    [[nodiscard]] Point3 transformToWorld(const Point3& localPt) const 
    {
        return orientation * (localPt * scaleFactor) + translation;
    }

    [[nodiscard]] Point3 transformToLocal(const Point3& worldPt) const 
    {
        Point3 local = orientation.inverse() * (worldPt - translation);
        return (scaleFactor > static_cast<RealType>(0)) ? local / scaleFactor : local;
    }

    [[nodiscard]] Point3 applyScale(const Point3& localPt) const 
    {
        return localPt * scaleFactor;
    }
};

/*
 * DynamicEntity - A physical object in the simulation
 */
class DynamicEntity 
{
public:
    DynamicEntity() = default;
    explicit DynamicEntity(TextType identifier);

    /* Geometry */
    bool attachSurface(const TextType& resourcePath, bool buildTree = true);
    void assignSurface(JointPtr<TriangleSurface> surface);

    [[nodiscard]] bool hasSurface() const { return m_geometry != nullptr; }
    [[nodiscard]] TriangleSurface& surface() { return *m_geometry; }
    [[nodiscard]] const TriangleSurface& surface() const { return *m_geometry; }

    /* Properties */
    [[nodiscard]] MaterialProperties& material() { return m_material; }
    [[nodiscard]] const MaterialProperties& material() const { return m_material; }

    [[nodiscard]] EntityState& kinematic() { return m_kinematic; }
    [[nodiscard]] const EntityState& kinematic() const { return m_kinematic; }

    void assignMaterial(const MaterialProperties& props);
    void assignKinematic(const EntityState& state);

    [[nodiscard]] const TextType& identifier() const { return m_identifier; }

    /* Dynamics */
    [[nodiscard]] bool isMovable() const { return m_material.isMovable(); }

    void accumulateForce(const Point3& force, const Point3& worldPoint);
    void accumulateTorque(const Point3& torque);
    void resetAccumulators();

    [[nodiscard]] const Point3& totalForce() const { return m_forceSum; }
    [[nodiscard]] const Point3& totalTorque() const { return m_torqueSum; }

    [[nodiscard]] const Matrix33& worldInertiaInverse();

    /* Bounds */
    [[nodiscard]] BoundingBox3D& worldExtent();

private:
    void recomputeWorldExtent();
    void refreshWorldInertia();

    TextType m_identifier;
    JointPtr<TriangleSurface> m_geometry;
    MaterialProperties m_material;
    EntityState m_kinematic;

    Point3 m_forceSum   = Point3::Zero();
    Point3 m_torqueSum  = Point3::Zero();

    BoundingBox3D m_worldExtent;
    Matrix33 m_worldInertiaInv = Matrix33::Identity();

    bool m_extentStale   = true;
    bool m_inertiaStale  = true;
};

}  // namespace phys3d

namespace rigid {
    using BodyState = phys3d::EntityState;
    using RigidBody = phys3d::DynamicEntity;
}

#endif // PHYS3D_DYNAMIC_ENTITY_HPP
