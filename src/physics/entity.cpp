/*
 * Implementation: DynamicEntity class
 * Restructured with component-based operations
 */
#include "entity.h"
#include "core/resource_pool.h"

namespace phys3d {

namespace {

// Helper: Compute world-space AABB for a point cloud with transform
void computeTransformedBounds(
    const DynArray<Point3>& points,
    const Point3& centroid,
    const Matrix33& rotation,
    const Point3& translation,
    RealType scale,
    BoundingBox3D& outBounds)
{
    outBounds.invalidate();
    
    const size_t count = points.size();
    if (count == 0)
        return;
    
    // Process in chunks for better cache behavior
    size_t idx = 0;
    while (idx < count)
    {
        // Compute offset from centroid
        const Point3& pt = points[idx];
        const RealType ox = (pt.x() - centroid.x()) * scale;
        const RealType oy = (pt.y() - centroid.y()) * scale;
        const RealType oz = (pt.z() - centroid.z()) * scale;
        
        // Apply rotation (using explicit matrix multiplication)
        const RealType rx = rotation(0,0) * ox + rotation(0,1) * oy + rotation(0,2) * oz;
        const RealType ry = rotation(1,0) * ox + rotation(1,1) * oy + rotation(1,2) * oz;
        const RealType rz = rotation(2,0) * ox + rotation(2,1) * oy + rotation(2,2) * oz;
        
        // Apply translation
        const Point3 worldPt(
            rx + translation.x(),
            ry + translation.y(),
            rz + translation.z()
        );
        
        outBounds.enclose(worldPt);
        ++idx;
    }
}

// Helper: Transform inertia tensor to world frame
Matrix33 transformInertiaToWorld(
    const Matrix33& localInertiaInv,
    const Matrix33& rotation,
    RealType scale)
{
    const RealType scaleSq = scale * scale;
    
    // Handle degenerate scale
    RealType invScaleSq;
    if (scaleSq > static_cast<RealType>(1e-10))
        invScaleSq = static_cast<RealType>(1) / scaleSq;
    else
        invScaleSq = static_cast<RealType>(1);
    
    // I_world^{-1} = R * (I_local^{-1} / scale^2) * R^T
    const Matrix33 scaledLocal = localInertiaInv * invScaleSq;
    
    // Compute R * scaledLocal
    Matrix33 temp;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            RealType sum = static_cast<RealType>(0);
            int k = 0;
            do {
                sum += rotation(i, k) * scaledLocal(k, j);
                ++k;
            } while (k < 3);
            temp(i, j) = sum;
        }
    }
    
    // Compute temp * R^T
    Matrix33 result;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            RealType sum = static_cast<RealType>(0);
            int k = 0;
            do {
                sum += temp(i, k) * rotation(j, k);  // R^T(k,j) = R(j,k)
                ++k;
            } while (k < 3);
            result(i, j) = sum;
        }
    }
    
    return result;
}

} // anonymous namespace

/* ========== Constructor ========== */

DynamicEntity::DynamicEntity(TextType identifier) 
    : m_identifier(std::move(identifier)) 
{
    m_cache.invalidateAll();
}

/* ========== Geometry Management ========== */

bool DynamicEntity::attachSurface(const TextType& resourcePath, bool buildTree) 
{
    auto surfacePtr = SurfaceResourcePool::global().fetch(resourcePath, buildTree);
    
    bool success = surfacePtr.operator bool();
    if (!success)
        return false;

    m_geometry = std::move(surfacePtr);
    m_cache.invalidateAll();
    return true;
}

void DynamicEntity::assignSurface(JointPtr<TriangleSurface> surface) 
{
    m_geometry = std::move(surface);
    m_cache.invalidateAll();
}

/* ========== Property Assignment ========== */

void DynamicEntity::assignMaterial(const MaterialProperties& props) 
{
    m_material = props;
    m_cache.invalidateInertia();
}

void DynamicEntity::assignKinematic(const EntityState& state) 
{
    m_kinematic = state;
    m_cache.invalidateAll();
}

/* ========== Force/Torque Accumulation ========== */

void DynamicEntity::accumulateForce(const Point3& force, const Point3& worldPoint) 
{
    m_forces.addForceAtPoint(force, worldPoint, m_kinematic.translation);
}

void DynamicEntity::accumulateTorque(const Point3& torque) 
{
    m_forces.addPureTorque(torque);
}

void DynamicEntity::resetAccumulators() 
{
    m_forces.reset();
}

/* ========== Lazy Computed Properties ========== */

const Matrix33& DynamicEntity::worldInertiaInverse() 
{
    if (!m_cache.inertiaValid) 
    {
        refreshWorldInertia();
    }
    return m_worldInertiaInv;
}

BoundingBox3D& DynamicEntity::worldExtent() 
{
    if (!m_cache.extentValid) 
    {
        recomputeWorldExtent();
    }
    return m_worldExtent;
}

/* ========== Internal Update Methods ========== */

void DynamicEntity::recomputeWorldExtent() 
{
    // Check for valid geometry
    bool hasGeometry = m_geometry.operator bool();
    if (!hasGeometry) 
    {
        m_worldExtent.invalidate();
        m_cache.extentValid = true;
        return;
    }

    const auto& pointCloud = m_geometry->pointCloud();
    bool hasPoints = !pointCloud.empty();
    if (!hasPoints) 
    {
        m_worldExtent.invalidate();
        m_cache.extentValid = true;
        return;
    }

    const Matrix33 rotMat = m_kinematic.orientationMatrix();
    
    computeTransformedBounds(
        pointCloud,
        m_material.massCentroid,
        rotMat,
        m_kinematic.translation,
        m_kinematic.scaleFactor,
        m_worldExtent
    );

    m_cache.extentValid = true;
}

void DynamicEntity::refreshWorldInertia() 
{
    const Matrix33 rotMat = m_kinematic.orientationMatrix();
    
    m_worldInertiaInv = transformInertiaToWorld(
        m_material.localInertiaInv,
        rotMat,
        m_kinematic.scaleFactor
    );
    
    m_cache.inertiaValid = true;
}

}  // namespace phys3d
