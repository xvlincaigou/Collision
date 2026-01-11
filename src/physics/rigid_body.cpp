/*
 * Implementation: DynamicEntity class
 */
#include "rigid_body.h"
#include "core/mesh_cache.h"

namespace phys3d {

DynamicEntity::DynamicEntity(TextType identifier) 
    : m_identifier(std::move(identifier)) 
{}

bool DynamicEntity::attachSurface(const TextType& resourcePath, bool buildTree) 
{
    auto surfacePtr = SurfaceResourcePool::global().fetch(resourcePath, buildTree);
    if (!surfacePtr) 
    {
        return false;
    }

    m_geometry = std::move(surfacePtr);
    m_extentStale = true;
    m_inertiaStale = true;
    return true;
}

void DynamicEntity::assignSurface(JointPtr<TriangleSurface> surface) 
{
    m_geometry = std::move(surface);
    m_extentStale = true;
    m_inertiaStale = true;
}

void DynamicEntity::assignMaterial(const MaterialProperties& props) 
{
    m_material = props;
    m_inertiaStale = true;
}

void DynamicEntity::assignKinematic(const EntityState& state) 
{
    m_kinematic = state;
    m_extentStale = true;
    m_inertiaStale = true;
}

void DynamicEntity::accumulateForce(const Point3& force, const Point3& worldPoint) 
{
    m_forceSum += force;
    Point3 lever = worldPoint - m_kinematic.translation;
    m_torqueSum += lever.cross(force);
}

void DynamicEntity::accumulateTorque(const Point3& torque) 
{
    m_torqueSum += torque;
}

void DynamicEntity::resetAccumulators() 
{
    m_forceSum.setZero();
    m_torqueSum.setZero();
}

const Matrix33& DynamicEntity::worldInertiaInverse() 
{
    if (m_inertiaStale) 
    {
        refreshWorldInertia();
    }
    return m_worldInertiaInv;
}

BoundingBox3D& DynamicEntity::worldExtent() 
{
    if (m_extentStale) 
    {
        recomputeWorldExtent();
    }
    return m_worldExtent;
}

void DynamicEntity::recomputeWorldExtent() 
{
    m_worldExtent.invalidate();

    if (!m_geometry) 
    {
        m_extentStale = false;
        return;
    }

    const auto& pointCloud = m_geometry->pointCloud();
    if (pointCloud.empty()) 
    {
        m_extentStale = false;
        return;
    }

    Matrix33 rotMat = m_kinematic.orientationMatrix();
    RealType scaleFactor = m_kinematic.scaleFactor;
    
    for (const Point3& localPt : pointCloud) 
    {
        Point3 scaledPt = (localPt - m_material.massCentroid) * scaleFactor;
        Point3 worldPt = rotMat * scaledPt + m_kinematic.translation;
        m_worldExtent.enclose(worldPt);
    }

    m_extentStale = false;
}

void DynamicEntity::refreshWorldInertia() 
{
    Matrix33 rotMat = m_kinematic.orientationMatrix();
    RealType scaleSq = m_kinematic.scaleFactor * m_kinematic.scaleFactor;
    RealType invScaleSq = (scaleSq > static_cast<RealType>(0)) ? 
                          (static_cast<RealType>(1) / scaleSq) : 
                          static_cast<RealType>(1);
    m_worldInertiaInv = rotMat * (m_material.localInertiaInv * invScaleSq) * rotMat.transpose();
    m_inertiaStale = false;
}

}  // namespace phys3d
