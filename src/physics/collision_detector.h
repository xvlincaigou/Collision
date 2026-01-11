/*
 * Collision Detection System
 */
#ifndef PHYS3D_COLLISION_SYSTEM_HPP
#define PHYS3D_COLLISION_SYSTEM_HPP

#include "contact.h"
#include "core/common.h"
#include "scene/scene.h"
#include "accel/bvh.h"

namespace phys3d {

namespace gpu {
class NarrowPhaseDetector;
struct DeviceContact;
struct EntityPair;
}

/*
 * CollisionSystem - Detects collisions between entities and environment
 */
class CollisionSystem 
{
public:
    CollisionSystem();
    ~CollisionSystem();

    void reset() { m_contactList.clear(); }

    void performDetection(World& world);

    [[nodiscard]] const DynArray<ContactPoint>& contactList() const { return m_contactList; }

    void enableDeviceAcceleration(bool enable) { m_useDevice = enable; }
    [[nodiscard]] bool deviceAccelerationEnabled() const { return m_useDevice; }

private:
    void detectEntityEnvironment(IntType entityIdx, DynamicEntity& entity, Boundaries& env);
    void detectEntityEntity(IntType idxA, DynamicEntity& entityA, IntType idxB, DynamicEntity& entityB);
    void detectPointsAgainstSurface(IntType dynamicIdx, DynamicEntity& dynamic,
                                     IntType staticIdx, DynamicEntity& stationary);

    void traverseTreeAgainstPlane(const TreeNode& node, const SpatialTree& tree,
                                   const HalfSpace3D& localPlane,
                                   const DynArray<Point3>& pointCloud,
                                   const DynArray<Triplet3i>& faces,
                                   const Matrix33& rotMat, const Point3& translation,
                                   IntType entityIdx, const Point3& worldPlaneNormal,
                                   DynArray<bool>& visitedPoints);

    void traverseTreeAgainstPoint(const TreeNode& node, const SpatialTree& tree,
                                   const Point3& queryLocal,
                                   const DynArray<Point3>& pointCloud,
                                   const DynArray<Triplet3i>& faces,
                                   RealType& minDistSq, Point3& closestNormal,
                                   Point3& closestPos, bool& foundContact);

    void performDeviceDetection(World& world);
    void convertDeviceContacts(const DynArray<gpu::DeviceContact>& deviceContacts);
    void deviceBroadPhase(World& world, DynArray<std::pair<IntType, IntType>>& candidatePairs);

    DynArray<ContactPoint> m_contactList;
    bool m_useDevice = true;

    SolePtr<gpu::NarrowPhaseDetector> m_deviceDetector;
};

/* Triangle collision helper */
inline bool testTrianglePenetration(
    const Point3& queryLocal,
    const Point3& v0, const Point3& v1, const Point3& v2,
    RealType& minDistSq,
    Point3& outNormal,
    Point3& outPoint)
{
    Point3 e1 = v1 - v0;
    Point3 e2 = v2 - v0;
    Point3 faceNormal = e1.cross(e2).normalized();

    RealType planeDist = (queryLocal - v0).dot(faceNormal);
    constexpr RealType kPenetrationLimit = static_cast<RealType>(-0.5);

    if (planeDist < static_cast<RealType>(0) && planeDist > kPenetrationLimit) 
    {
        RealType distSq = planeDist * planeDist;
        if (distSq < minDistSq) 
        {
            Point3 projected = queryLocal - faceNormal * planeDist;
            Point3 projRel = projected - v0;

            RealType d00 = e1.dot(e1);
            RealType d01 = e1.dot(e2);
            RealType d11 = e2.dot(e2);
            RealType d20 = projRel.dot(e1);
            RealType d21 = projRel.dot(e2);
            RealType denom = d00 * d11 - d01 * d01;

            if (std::abs(denom) > static_cast<RealType>(1e-8)) 
            {
                RealType denomInv = static_cast<RealType>(1) / denom;
                RealType u = (d11 * d20 - d01 * d21) * denomInv;
                RealType w = (d00 * d21 - d01 * d20) * denomInv;
                RealType v = static_cast<RealType>(1) - u - w;

                if (v >= static_cast<RealType>(0) && u >= static_cast<RealType>(0) && w >= static_cast<RealType>(0)) 
                {
                    minDistSq = distSq;
                    outNormal = faceNormal;
                    outPoint = projected;
                    return true;
                }
            }
        }
    }
    return false;
}

}  // namespace phys3d

namespace rigid {
    using CollisionDetector = phys3d::CollisionSystem;
    inline bool checkTriangleCollision(const phys3d::Point3& p, 
                                        const phys3d::Point3& v0, 
                                        const phys3d::Point3& v1, 
                                        const phys3d::Point3& v2,
                                        phys3d::RealType& d, 
                                        phys3d::Point3& n, 
                                        phys3d::Point3& pt) {
        return phys3d::testTrianglePenetration(p, v0, v1, v2, d, n, pt);
    }
}

#endif // PHYS3D_COLLISION_SYSTEM_HPP
