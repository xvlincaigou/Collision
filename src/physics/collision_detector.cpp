/*
 * Implementation: CollisionSystem class
 */
#include "collision_detector.h"

#include <cmath>
#include <iostream>

#if defined(RIGID_USE_CUDA)
#include "gpu/collision_detector.cuh"
#include "gpu/broadphase.cuh"
#endif

namespace phys3d {

/* ========== Constructor/Destructor ========== */

#if defined(RIGID_USE_CUDA)

CollisionSystem::CollisionSystem() : m_useDevice(true) 
{
    m_deviceDetector = std::make_unique<gpu::NarrowPhaseDetector>();
}

CollisionSystem::~CollisionSystem() = default;

#else

CollisionSystem::CollisionSystem() : m_useDevice(false) {}
CollisionSystem::~CollisionSystem() = default;

#endif

/* ========== Main Detection Entry Point ========== */

void CollisionSystem::performDetection(World& world) 
{
    if (m_useDevice) 
    {
        performDeviceDetection(world);
        return;
    }

    reset();

    Boundaries& env = world.boundaries();
    IntType entityCount = world.entityCount();

    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entity = world.entity(i);
        if (!entity || !entity->hasSurface()) continue;
        detectEntityEnvironment(i, *entity, env);
    }

    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entityA = world.entity(i);
        if (!entityA || !entityA->hasSurface()) continue;

        for (IntType j = i + 1; j < entityCount; ++j) 
        {
            DynamicEntity* entityB = world.entity(j);
            if (!entityB || !entityB->hasSurface()) continue;

            if (entityA->worldExtent().overlaps(entityB->worldExtent())) 
            {
                detectEntityEntity(i, *entityA, j, *entityB);
            }
        }
    }
}

/* ========== Entity vs Environment ========== */

void CollisionSystem::detectEntityEnvironment(IntType entityIdx, DynamicEntity& entity, Boundaries& env) 
{
    TriangleSurface& surface = entity.surface();
    const EntityState& state = entity.kinematic();
    Matrix33 rotMat = state.orientationMatrix();
    Point3 translation = state.translation;

    const DynArray<Point3>& pointCloud = surface.pointCloud();
    static DynArray<bool> visitedPoints;
    visitedPoints.resize(pointCloud.size());

    for (IntType pIdx = 0; pIdx < Boundaries::planeCount(); ++pIdx) 
    {
        HalfSpace3D& worldPlane = env.planeAt(static_cast<Boundaries::PlaneId>(pIdx));

        const BoundingBox3D& wb = entity.worldExtent();
        Point3 supportMin;
        for (int k = 0; k < 3; ++k) 
        {
            supportMin[k] = (worldPlane.direction[k] >= static_cast<RealType>(0)) ? 
                            wb.corner_lo[k] : wb.corner_hi[k];
        }
        RealType distMin = supportMin.dot(worldPlane.direction) - worldPlane.distance;

        if (distMin > static_cast<RealType>(1e-4)) continue;

        HalfSpace3D localPlane;
        localPlane.direction = rotMat.transpose() * worldPlane.direction;
        localPlane.distance = worldPlane.distance - worldPlane.direction.dot(translation);

        if (surface.hasAccelerator()) 
        {
            std::fill(visitedPoints.begin(), visitedPoints.end(), false);
            traverseTreeAgainstPlane(
                surface.accelerator().nodeAt(surface.accelerator().rootIdx()),
                surface.accelerator(), localPlane, pointCloud, surface.faceIndices(),
                rotMat, translation, entityIdx, worldPlane.direction, visitedPoints);
        } 
        else 
        {
            for (const auto& localPt : pointCloud) 
            {
                RealType distLocal = localPt.dot(localPlane.direction) - localPlane.distance;
                if (distLocal < static_cast<RealType>(0)) 
                {
                    ContactPoint contact;
                    contact.entityA = -1;
                    contact.entityB = entityIdx;
                    contact.depth = -distLocal;
                    contact.direction = worldPlane.direction;
                    contact.location = rotMat * localPt + translation;
                    m_contactList.push_back(contact);
                }
            }
        }
    }
}

/* ========== Entity vs Entity ========== */

void CollisionSystem::detectEntityEntity(IntType idxA, DynamicEntity& entityA,
                                          IntType idxB, DynamicEntity& entityB) 
{
    detectPointsAgainstSurface(idxA, entityA, idxB, entityB);
    detectPointsAgainstSurface(idxB, entityB, idxA, entityA);
}

void CollisionSystem::detectPointsAgainstSurface(IntType dynamicIdx, DynamicEntity& dynamic,
                                                  IntType staticIdx, DynamicEntity& stationary) 
{
    const auto& dynamicPoints = dynamic.surface().pointCloud();
    const BoundingBox3D& staticBounds = stationary.worldExtent();

    Matrix33 rotDyn = dynamic.kinematic().orientationMatrix();
    Point3 posDyn = dynamic.kinematic().translation;
    Matrix33 rotStat = stationary.kinematic().orientationMatrix();
    Point3 posStat = stationary.kinematic().translation;
    Matrix33 rotStatInv = rotStat.transpose();

    const TriangleSurface& staticSurf = stationary.surface();
    const auto& staticPoints = staticSurf.pointCloud();
    const auto& staticFaces = staticSurf.faceIndices();

    constexpr RealType kProximityThreshold = static_cast<RealType>(0.5);

    for (const auto& localPt : dynamicPoints) 
    {
        Point3 worldPt = rotDyn * localPt + posDyn;

        if (!staticBounds.enclosesPoint(worldPt)) 
        {
            continue;
        }

        Point3 queryInStatic = rotStatInv * (worldPt - posStat);

        RealType minDistSq = kProximityThreshold * kProximityThreshold;
        Point3 closestNormal = Point3::Zero();
        Point3 closestPoint = Point3::Zero();
        bool foundContact = false;

        if (staticSurf.hasAccelerator()) 
        {
            traverseTreeAgainstPoint(
                staticSurf.accelerator().nodeAt(staticSurf.accelerator().rootIdx()),
                staticSurf.accelerator(), queryInStatic, staticPoints, staticFaces,
                minDistSq, closestNormal, closestPoint, foundContact);
        } 
        else 
        {
            const BoundingBox3D& lb = staticSurf.localExtent();
            if (lb.enclosesPoint(queryInStatic)) 
            {
                for (const auto& face : staticFaces) 
                {
                    const Point3& p0 = staticPoints[face[0]];
                    const Point3& p1 = staticPoints[face[1]];
                    const Point3& p2 = staticPoints[face[2]];
                    if (testTrianglePenetration(queryInStatic, p0, p1, p2,
                                                 minDistSq, closestNormal, closestPoint)) 
                    {
                        foundContact = true;
                    }
                }
            }
        }

        if (foundContact) 
        {
            ContactPoint contact;
            contact.entityA = staticIdx;
            contact.entityB = dynamicIdx;
            contact.depth = std::sqrt(minDistSq);
            contact.direction = rotStat * closestNormal;
            contact.location = worldPt;
            m_contactList.push_back(contact);
        }
    }
}

/* ========== Tree Traversal ========== */

void CollisionSystem::traverseTreeAgainstPlane(
    const TreeNode& node, const SpatialTree& tree,
    const HalfSpace3D& localPlane,
    const DynArray<Point3>& pointCloud,
    const DynArray<Triplet3i>& faces,
    const Matrix33& rotMat, const Point3& translation,
    IntType entityIdx, const Point3& worldPlaneNormal,
    DynArray<bool>& visitedPoints)
{
    Point3 center = node.volume.midpoint();
    Point3 extent = node.volume.halfSize();

    RealType r = extent.x() * std::abs(localPlane.direction.x()) +
                 extent.y() * std::abs(localPlane.direction.y()) +
                 extent.z() * std::abs(localPlane.direction.z());
    RealType s = localPlane.direction.dot(center) - localPlane.distance;

    if (s > r) return;

    if (node.isTerminal()) 
    {
        const auto& indices = tree.faceOrdering();
        IntType end = node.leafStart + node.leafCount;

        for (IntType i = node.leafStart; i < end; ++i) 
        {
            IntType faceIdx = indices[i];
            const Triplet3i& face = faces[faceIdx];

            for (int k = 0; k < 3; ++k) 
            {
                IntType vIdx = face[k];
                if (visitedPoints[vIdx]) continue;
                visitedPoints[vIdx] = true;

                const Point3& localPt = pointCloud[vIdx];
                RealType dist = localPt.dot(localPlane.direction) - localPlane.distance;

                if (dist < static_cast<RealType>(0)) 
                {
                    ContactPoint contact;
                    contact.entityA = -1;
                    contact.entityB = entityIdx;
                    contact.depth = -dist;
                    contact.direction = worldPlaneNormal;
                    contact.location = rotMat * localPt + translation;
                    m_contactList.push_back(contact);
                }
            }
        }
    } 
    else 
    {
        if (node.childLeft != -1) 
        {
            traverseTreeAgainstPlane(tree.nodeAt(node.childLeft), tree, localPlane,
                                     pointCloud, faces, rotMat, translation,
                                     entityIdx, worldPlaneNormal, visitedPoints);
        }
        if (node.childRight != -1) 
        {
            traverseTreeAgainstPlane(tree.nodeAt(node.childRight), tree, localPlane,
                                     pointCloud, faces, rotMat, translation,
                                     entityIdx, worldPlaneNormal, visitedPoints);
        }
    }
}

void CollisionSystem::traverseTreeAgainstPoint(
    const TreeNode& node, const SpatialTree& tree,
    const Point3& queryLocal,
    const DynArray<Point3>& pointCloud,
    const DynArray<Triplet3i>& faces,
    RealType& minDistSq, Point3& closestNormal,
    Point3& closestPos, bool& foundContact)
{
    RealType boxDistSq = static_cast<RealType>(0);
    for (int i = 0; i < 3; ++i) 
    {
        if (queryLocal[i] < node.volume.corner_lo[i]) 
        {
            RealType d = node.volume.corner_lo[i] - queryLocal[i];
            boxDistSq += d * d;
        } 
        else if (queryLocal[i] > node.volume.corner_hi[i]) 
        {
            RealType d = queryLocal[i] - node.volume.corner_hi[i];
            boxDistSq += d * d;
        }
    }

    if (boxDistSq > minDistSq) return;

    if (node.isTerminal()) 
    {
        const auto& indices = tree.faceOrdering();
        IntType end = node.leafStart + node.leafCount;

        for (IntType i = node.leafStart; i < end; ++i) 
        {
            const Triplet3i& face = faces[indices[i]];
            const Point3& p0 = pointCloud[face[0]];
            const Point3& p1 = pointCloud[face[1]];
            const Point3& p2 = pointCloud[face[2]];

            if (testTrianglePenetration(queryLocal, p0, p1, p2,
                                         minDistSq, closestNormal, closestPos)) 
            {
                foundContact = true;
            }
        }
    } 
    else 
    {
        if (node.childLeft != -1) 
        {
            traverseTreeAgainstPoint(tree.nodeAt(node.childLeft), tree, queryLocal,
                                     pointCloud, faces, minDistSq,
                                     closestNormal, closestPos, foundContact);
        }
        if (node.childRight != -1) 
        {
            traverseTreeAgainstPoint(tree.nodeAt(node.childRight), tree, queryLocal,
                                     pointCloud, faces, minDistSq,
                                     closestNormal, closestPos, foundContact);
        }
    }
}

/* ========== GPU Implementation ========== */

#if defined(RIGID_USE_CUDA)

void CollisionSystem::performDeviceDetection(World& world) 
{
    reset();

    Boundaries& env = world.boundaries();
    IntType entityCount = world.entityCount();

    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entity = world.entity(i);
        if (!entity || !entity->hasSurface()) continue;

        const SpatialTree& tree = entity->surface().accelerator();
        if (!tree.hasDeviceData()) continue;

        m_deviceDetector->assignTree(tree.deviceBuilder());

        gpu::EntityTransform transform;
        const EntityState& state = entity->kinematic();
        transform.translation = make_float3(state.translation.x(),
                                             state.translation.y(),
                                             state.translation.z());
        transform.rotation = make_float4(state.orientation.x(),
                                          state.orientation.y(),
                                          state.orientation.z(),
                                          state.orientation.w());

        DynArray<gpu::DevicePlane> planes(Boundaries::planeCount());
        for (int p = 0; p < Boundaries::planeCount(); ++p) 
        {
            const HalfSpace3D& plane = env.planeAt(static_cast<Boundaries::PlaneId>(p));
            planes[p].direction = make_float3(plane.direction.x(),
                                               plane.direction.y(),
                                               plane.direction.z());
            planes[p].offset = plane.distance;
        }

        DynArray<gpu::DeviceContact> deviceContacts;
        m_deviceDetector->detectEntityEnvironment(i, transform, planes.data(),
                                                   static_cast<IntType>(planes.size()),
                                                   deviceContacts);
        convertDeviceContacts(deviceContacts);
    }

    DynArray<std::pair<IntType, IntType>> candidatePairs;
    deviceBroadPhase(world, candidatePairs);

    for (const auto& pair : candidatePairs) 
    {
        IntType i = pair.first;
        IntType j = pair.second;

        DynamicEntity* entityA = world.entity(i);
        DynamicEntity* entityB = world.entity(j);

        if (!entityA || !entityB) continue;
        if (!entityA->hasSurface() || !entityB->hasSurface()) continue;

        const SpatialTree& treeA = entityA->surface().accelerator();
        const SpatialTree& treeB = entityB->surface().accelerator();
        if (!treeA.hasDeviceData() || !treeB.hasDeviceData()) continue;

        gpu::EntityTransform transformA, transformB;
        const EntityState& stateA = entityA->kinematic();
        const EntityState& stateB = entityB->kinematic();

        transformA.translation = make_float3(stateA.translation.x(),
                                              stateA.translation.y(),
                                              stateA.translation.z());
        transformA.rotation = make_float4(stateA.orientation.x(),
                                           stateA.orientation.y(),
                                           stateA.orientation.z(),
                                           stateA.orientation.w());
        transformB.translation = make_float3(stateB.translation.x(),
                                              stateB.translation.y(),
                                              stateB.translation.z());
        transformB.rotation = make_float4(stateB.orientation.x(),
                                           stateB.orientation.y(),
                                           stateB.orientation.z(),
                                           stateB.orientation.w());

        DynArray<gpu::DeviceContact> deviceContacts;
        m_deviceDetector->detectEntityEntity(
            i, transformA, treeA.deviceBuilder(),
            j, transformB, treeB.deviceBuilder(),
            deviceContacts);
        convertDeviceContacts(deviceContacts);
    }
}

void CollisionSystem::convertDeviceContacts(const DynArray<gpu::DeviceContact>& deviceContacts) 
{
    for (const auto& dc : deviceContacts) 
    {
        ContactPoint contact;
        contact.entityA = dc.entityIdxA;
        contact.entityB = dc.entityIdxB;
        contact.location = Point3(dc.worldPoint.x, dc.worldPoint.y, dc.worldPoint.z);
        contact.direction = Point3(dc.worldNormal.x, dc.worldNormal.y, dc.worldNormal.z);
        contact.depth = dc.penetration;
        m_contactList.push_back(contact);
    }
}

void CollisionSystem::deviceBroadPhase(World& world, DynArray<std::pair<IntType, IntType>>& candidatePairs) 
{
    IntType entityCount = world.entityCount();
    if (entityCount < 2) 
    {
        candidatePairs.clear();
        return;
    }

    DynArray<Point3> boundsMin(entityCount);
    DynArray<Point3> boundsMax(entityCount);

    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entity = world.entity(i);
        if (entity && entity->hasSurface()) 
        {
            BoundingBox3D& wb = entity->worldExtent();
            boundsMin[i] = wb.corner_lo;
            boundsMax[i] = wb.corner_hi;
        } 
        else 
        {
            boundsMin[i] = Point3(1e30f, 1e30f, 1e30f);
            boundsMax[i] = Point3(-1e30f, -1e30f, -1e30f);
        }
    }

    DynArray<gpu::EntityPair> devicePairs;
    m_deviceDetector->performBroadPhase(boundsMin, boundsMax, devicePairs);

    candidatePairs.resize(devicePairs.size());
    for (size_t i = 0; i < devicePairs.size(); ++i) 
    {
        candidatePairs[i] = std::make_pair(devicePairs[i].entityA, devicePairs[i].entityB);
    }
}

#else

void CollisionSystem::performDeviceDetection(World& world) 
{
    m_useDevice = false;
    performDetection(world);
}

void CollisionSystem::convertDeviceContacts(const DynArray<gpu::DeviceContact>&) {}

void CollisionSystem::deviceBroadPhase(World&, DynArray<std::pair<IntType, IntType>>&) {}

#endif

}  // namespace phys3d
