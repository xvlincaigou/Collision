/*
 * Dynamic Entity with Geometry and Physical State
 * Redesigned with cached state management and component-based layout
 */
#ifndef PHYS3D_DYNAMIC_ENTITY_HPP
#define PHYS3D_DYNAMIC_ENTITY_HPP

#include "material.h"
#include "core/types.h"
#include "core/surface.h"

namespace phys3d {

/*
 * EntityState - Kinematic state with integrated transform operations
 */
struct EntityState 
{
    // Position and orientation
    Point3    translation  = Point3::Zero();
    Rotation4 orientation  = Rotation4::Identity();
    
    // Motion
    Point3    velocity     = Point3::Zero();
    Point3    angularRate  = Point3::Zero();
    
    // Uniform scale
    RealType  scaleFactor  = static_cast<RealType>(1);

    // Pre-compute rotation matrix
    [[nodiscard]] Matrix33 orientationMatrix() const 
    {
        // Explicit quaternion to matrix conversion
        const RealType qx = orientation.x();
        const RealType qy = orientation.y();
        const RealType qz = orientation.z();
        const RealType qw = orientation.w();
        
        const RealType xx = qx * qx;
        const RealType yy = qy * qy;
        const RealType zz = qz * qz;
        const RealType xy = qx * qy;
        const RealType xz = qx * qz;
        const RealType xw = qx * qw;
        const RealType yz = qy * qz;
        const RealType yw = qy * qw;
        const RealType zw = qz * qw;
        
        Matrix33 result;
        result(0,0) = static_cast<RealType>(1) - static_cast<RealType>(2) * (yy + zz);
        result(0,1) = static_cast<RealType>(2) * (xy - zw);
        result(0,2) = static_cast<RealType>(2) * (xz + yw);
        result(1,0) = static_cast<RealType>(2) * (xy + zw);
        result(1,1) = static_cast<RealType>(1) - static_cast<RealType>(2) * (xx + zz);
        result(1,2) = static_cast<RealType>(2) * (yz - xw);
        result(2,0) = static_cast<RealType>(2) * (xz - yw);
        result(2,1) = static_cast<RealType>(2) * (yz + xw);
        result(2,2) = static_cast<RealType>(1) - static_cast<RealType>(2) * (xx + yy);
        return result;
    }

    [[nodiscard]] Point3 transformToWorld(const Point3& localPt) const 
    {
        // Apply scale, then rotation, then translation
        const Point3 scaled(
            localPt.x() * scaleFactor,
            localPt.y() * scaleFactor,
            localPt.z() * scaleFactor
        );
        const Point3 rotated = orientation * scaled;
        return Point3(
            rotated.x() + translation.x(),
            rotated.y() + translation.y(),
            rotated.z() + translation.z()
        );
    }

    [[nodiscard]] Point3 transformToLocal(const Point3& worldPt) const 
    {
        Point3 delta(
            worldPt.x() - translation.x(),
            worldPt.y() - translation.y(),
            worldPt.z() - translation.z()
        );
        const Point3 rotated = orientation.inverse() * delta;
        
        const RealType invScale = (scaleFactor > static_cast<RealType>(0)) 
                                   ? static_cast<RealType>(1) / scaleFactor 
                                   : static_cast<RealType>(1);
        return Point3(
            rotated.x() * invScale,
            rotated.y() * invScale,
            rotated.z() * invScale
        );
    }

    [[nodiscard]] Point3 applyScale(const Point3& localPt) const 
    {
        return Point3(
            localPt.x() * scaleFactor,
            localPt.y() * scaleFactor,
            localPt.z() * scaleFactor
        );
    }
};

/*
 * CacheControl - Tracks validity of computed quantities
 */
struct CacheControl
{
    bool extentValid   = false;
    bool inertiaValid  = false;
    
    void invalidateAll() 
    { 
        extentValid = false; 
        inertiaValid = false; 
    }
    
    void invalidateExtent()  { extentValid = false; }
    void invalidateInertia() { inertiaValid = false; }
};

/*
 * ForceAccumulator - Stores accumulated forces and torques
 */
struct ForceAccumulator
{
    Point3 force  = Point3::Zero();
    Point3 torque = Point3::Zero();
    
    void reset() 
    {
        force.x() = static_cast<RealType>(0);
        force.y() = static_cast<RealType>(0);
        force.z() = static_cast<RealType>(0);
        torque.x() = static_cast<RealType>(0);
        torque.y() = static_cast<RealType>(0);
        torque.z() = static_cast<RealType>(0);
    }
    
    void addForceAtPoint(const Point3& f, const Point3& worldPt, const Point3& bodyCenter)
    {
        // Add force
        force.x() += f.x();
        force.y() += f.y();
        force.z() += f.z();
        
        // Compute and add torque: r × f
        const Point3 lever(
            worldPt.x() - bodyCenter.x(),
            worldPt.y() - bodyCenter.y(),
            worldPt.z() - bodyCenter.z()
        );
        torque.x() += lever.y() * f.z() - lever.z() * f.y();
        torque.y() += lever.z() * f.x() - lever.x() * f.z();
        torque.z() += lever.x() * f.y() - lever.y() * f.x();
    }
    
    void addPureTorque(const Point3& t)
    {
        torque.x() += t.x();
        torque.y() += t.y();
        torque.z() += t.z();
    }
};

/*
 * DynamicEntity - A physical object in the simulation
 * Uses component-based design for cleaner separation
 */
class DynamicEntity 
{
public:
    DynamicEntity() = default;
    explicit DynamicEntity(TextType identifier);

    /* ===== Geometry Management ===== */
    
    bool attachSurface(const TextType& resourcePath, bool buildTree = true);
    void assignSurface(JointPtr<TriangleSurface> surface);

    [[nodiscard]] bool hasSurface() const 
    { 
        return m_geometry.operator bool(); 
    }
    
    [[nodiscard]] TriangleSurface& surface() 
    { 
        return *m_geometry; 
    }
    
    [[nodiscard]] const TriangleSurface& surface() const 
    { 
        return *m_geometry; 
    }

    /* ===== Property Accessors ===== */
    
    [[nodiscard]] MaterialProperties& material() 
    { 
        return m_material; 
    }
    
    [[nodiscard]] const MaterialProperties& material() const 
    { 
        return m_material; 
    }

    [[nodiscard]] EntityState& kinematic() 
    { 
        return m_kinematic; 
    }
    
    [[nodiscard]] const EntityState& kinematic() const 
    { 
        return m_kinematic; 
    }

    void assignMaterial(const MaterialProperties& props);
    void assignKinematic(const EntityState& state);

    [[nodiscard]] const TextType& identifier() const 
    { 
        return m_identifier; 
    }

    /* ===== Dynamics ===== */
    
    [[nodiscard]] bool isMovable() const 
    { 
        return m_material.isMovable(); 
    }

    void accumulateForce(const Point3& force, const Point3& worldPoint);
    void accumulateTorque(const Point3& torque);
    void resetAccumulators();

    [[nodiscard]] const Point3& totalForce() const 
    { 
        return m_forces.force; 
    }
    
    [[nodiscard]] const Point3& totalTorque() const 
    { 
        return m_forces.torque; 
    }

    [[nodiscard]] const Matrix33& worldInertiaInverse();

    /* ===== Spatial Bounds ===== */
    
    [[nodiscard]] BoundingBox3D& worldExtent();

private:
    /* ===== Internal Update Methods ===== */
    void recomputeWorldExtent();
    void refreshWorldInertia();
    
    // Core identity
    TextType m_identifier;
    
    // Geometry component
    JointPtr<TriangleSurface> m_geometry;
    
    // Physical properties
    MaterialProperties m_material;
    
    // Kinematic state
    EntityState m_kinematic;
    
    // Force/torque accumulator
    ForceAccumulator m_forces;
    
    // Cached computed values
    BoundingBox3D m_worldExtent;
    Matrix33 m_worldInertiaInv = Matrix33::Identity();
    
    // Cache validity tracking
    CacheControl m_cache;
};

}  // namespace phys3d

namespace rigid {
    using BodyState = phys3d::EntityState;
    using RigidBody = phys3d::DynamicEntity;
}

#endif // PHYS3D_DYNAMIC_ENTITY_HPP
