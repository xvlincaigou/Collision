/*
 * Implementation: CollisionSystem class
 * Uses iterative BVH traversal and restructured detection pipeline
 */
#include "collision_detector.h"

#include <cmath>
#include <iostream>
#include <stack>
#include <array>

#if defined(RIGID_USE_CUDA)
#include "gpu/collision_detector.cuh"
#include "gpu/broadphase.cuh"
#endif

namespace phys3d {

namespace {

// Constants
constexpr RealType kPlaneEpsilon = static_cast<RealType>(1e-4);
constexpr RealType kProximityThreshold = static_cast<RealType>(0.5);
constexpr IntType kMaxStackDepth = 64;

// Helper: Check if AABB intersects half-space
bool boxIntersectsHalfSpace(const BoundingBox3D& box, const HalfSpace3D& plane)
{
    // Find support point in negative normal direction
    Point3 support;
    support.x() = (plane.direction.x() >= static_cast<RealType>(0)) ? box.corner_lo.x() : box.corner_hi.x();
    support.y() = (plane.direction.y() >= static_cast<RealType>(0)) ? box.corner_lo.y() : box.corner_hi.y();
    support.z() = (plane.direction.z() >= static_cast<RealType>(0)) ? box.corner_lo.z() : box.corner_hi.z();
    
    return plane.evaluate(support) <= kPlaneEpsilon;
}

// Helper: Transform plane to local space
HalfSpace3D transformPlaneToLocal(const HalfSpace3D& worldPlane, const Matrix33& rotMat, const Point3& translation)
{
    HalfSpace3D localPlane;
    localPlane.direction = rotMat.transpose() * worldPlane.direction;
    localPlane.distance = worldPlane.distance - worldPlane.direction.dot(translation);
    return localPlane;
}

// Helper: Check AABB-plane overlap using center/extent form
bool nodeIntersectsPlane(const BoundingBox3D& box, const HalfSpace3D& plane)
{
    const Point3 center = box.midpoint();
    const Point3 extent = box.halfSize();
    
    const RealType projectedRadius = 
        extent.x() * std::abs(plane.direction.x()) +
        extent.y() * std::abs(plane.direction.y()) +
        extent.z() * std::abs(plane.direction.z());
    
    const RealType signedDist = plane.evaluate(center);
    
    return signedDist <= projectedRadius;
}

// Helper: Compute squared distance from point to AABB
RealType pointToBoxDistanceSq(const Point3& pt, const BoundingBox3D& box)
{
    RealType distSq = static_cast<RealType>(0);
    
    // X axis
    if (pt.x() < box.corner_lo.x())
    {
        const RealType d = box.corner_lo.x() - pt.x();
        distSq += d * d;
    }
    else if (pt.x() > box.corner_hi.x())
    {
        const RealType d = pt.x() - box.corner_hi.x();
        distSq += d * d;
    }
    
    // Y axis
    if (pt.y() < box.corner_lo.y())
    {
        const RealType d = box.corner_lo.y() - pt.y();
        distSq += d * d;
    }
    else if (pt.y() > box.corner_hi.y())
    {
        const RealType d = pt.y() - box.corner_hi.y();
        distSq += d * d;
    }
    
    // Z axis
    if (pt.z() < box.corner_lo.z())
    {
        const RealType d = box.corner_lo.z() - pt.z();
        distSq += d * d;
    }
    else if (pt.z() > box.corner_hi.z())
    {
        const RealType d = pt.z() - box.corner_hi.z();
        distSq += d * d;
    }
    
    return distSq;
}

// Helper: Create contact from vertex-plane penetration
ContactPoint createPlaneContact(
    const Point3& localVertex,
    const Matrix33& rotMat,
    const Point3& translation,
    IntType entityIdx,
    const Point3& worldNormal,
    RealType penetration)
{
    ContactPoint contact;
    contact.entityA = -1;
    contact.entityB = entityIdx;
    contact.direction = worldNormal;
    contact.depth = penetration;
    contact.location = rotMat * localVertex + translation;
    return contact;
}

// Helper: Create contact from mesh proximity
ContactPoint createMeshContact(
    const Point3& worldPoint,
    const Point3& localNormal,
    const Matrix33& rotMat,
    IntType staticIdx,
    IntType dynamicIdx,
    RealType distance)
{
    ContactPoint contact;
    contact.entityA = staticIdx;
    contact.entityB = dynamicIdx;
    contact.location = worldPoint;
    contact.direction = rotMat * localNormal;
    contact.depth = distance;
    return contact;
}

} // anonymous namespace

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
    const IntType entityCount = world.entityCount();

    // Phase 1: Entity-Environment collisions
    IntType i = 0;
    while (i < entityCount)
    {
        DynamicEntity* entity = world.entity(i);
        bool valid = (entity != nullptr) && entity->hasSurface();
        if (valid)
            detectEntityEnvironment(i, *entity, env);
        ++i;
    }

    // Phase 2: Entity-Entity collisions (upper triangular iteration)
    i = 0;
    while (i < entityCount)
    {
        DynamicEntity* entityA = world.entity(i);
        bool aValid = (entityA != nullptr) && entityA->hasSurface();
        
        if (aValid)
        {
            IntType j = i + 1;
            while (j < entityCount)
            {
                DynamicEntity* entityB = world.entity(j);
                bool bValid = (entityB != nullptr) && entityB->hasSurface();
                
                if (bValid)
                {
                    const bool overlapping = entityA->worldExtent().overlaps(entityB->worldExtent());
                    if (overlapping)
                        detectEntityEntity(i, *entityA, j, *entityB);
                }
                ++j;
            }
        }
        ++i;
    }
}

/* ========== Entity vs Environment (Iterative BVH Traversal) ========== */

void CollisionSystem::detectEntityEnvironment(IntType entityIdx, DynamicEntity& entity, Boundaries& env) 
{
    TriangleSurface& surface = entity.surface();
    const EntityState& state = entity.kinematic();
    const Matrix33 rotMat = state.orientationMatrix();
    const Point3& translation = state.translation;

    const DynArray<Point3>& pointCloud = surface.pointCloud();
    static DynArray<bool> visitedPoints;
    visitedPoints.assign(pointCloud.size(), false);

    // Process each boundary plane
    IntType planeIdx = 0;
    while (planeIdx < Boundaries::planeCount())
    {
        HalfSpace3D& worldPlane = env.planeAt(static_cast<Boundaries::PlaneId>(planeIdx));

        // Quick AABB rejection
        if (!boxIntersectsHalfSpace(entity.worldExtent(), worldPlane))
        {
            ++planeIdx;
            continue;
        }

        const HalfSpace3D localPlane = transformPlaneToLocal(worldPlane, rotMat, translation);

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
            // Brute force: check all vertices
            IntType vIdx = static_cast<IntType>(pointCloud.size());
            while (vIdx > 0)
            {
                --vIdx;
                const Point3& localPt = pointCloud[vIdx];
                const RealType dist = localPlane.evaluate(localPt);
                
                if (dist < static_cast<RealType>(0))
                {
                    m_contactList.push_back(createPlaneContact(
                        localPt, rotMat, translation, entityIdx, worldPlane.direction, -dist));
                }
            }
        }
        
        ++planeIdx;
    }
}

/* ========== Entity vs Entity ========== */

void CollisionSystem::detectEntityEntity(IntType idxA, DynamicEntity& entityA,
                                          IntType idxB, DynamicEntity& entityB) 
{
    // Symmetric detection: A's vertices vs B's mesh, then B's vertices vs A's mesh
    detectPointsAgainstSurface(idxA, entityA, idxB, entityB);
    detectPointsAgainstSurface(idxB, entityB, idxA, entityA);
}

void CollisionSystem::detectPointsAgainstSurface(IntType dynamicIdx, DynamicEntity& dynamic,
                                                  IntType staticIdx, DynamicEntity& stationary) 
{
    const auto& dynamicPoints = dynamic.surface().pointCloud();
    const BoundingBox3D& staticBounds = stationary.worldExtent();

    const Matrix33 rotDyn = dynamic.kinematic().orientationMatrix();
    const Point3& posDyn = dynamic.kinematic().translation;
    const Matrix33 rotStat = stationary.kinematic().orientationMatrix();
    const Point3& posStat = stationary.kinematic().translation;
    const Matrix33 rotStatInv = rotStat.transpose();

    const TriangleSurface& staticSurf = stationary.surface();
    const auto& staticPoints = staticSurf.pointCloud();
    const auto& staticFaces = staticSurf.faceIndices();

    const RealType thresholdSq = kProximityThreshold * kProximityThreshold;

    // Process vertices in reverse order
    IntType vIdx = static_cast<IntType>(dynamicPoints.size());
    while (vIdx > 0)
    {
        --vIdx;
        const Point3& localPt = dynamicPoints[vIdx];
        const Point3 worldPt = rotDyn * localPt + posDyn;

        // Quick AABB rejection
        if (!staticBounds.enclosesPoint(worldPt))
            continue;

        const Point3 queryInStatic = rotStatInv * (worldPt - posStat);

        RealType minDistSq = thresholdSq;
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
            // Brute force fallback
            if (staticSurf.localExtent().enclosesPoint(queryInStatic))
            {
                IntType faceIdx = static_cast<IntType>(staticFaces.size());
                while (faceIdx > 0)
                {
                    --faceIdx;
                    const auto& face = staticFaces[faceIdx];
                    
                    if (testTrianglePenetration(queryInStatic,
                                                 staticPoints[face[0]],
                                                 staticPoints[face[1]],
                                                 staticPoints[face[2]],
                                                 minDistSq, closestNormal, closestPoint))
                    {
                        foundContact = true;
                    }
                }
            }
        }

        if (foundContact)
        {
            m_contactList.push_back(createMeshContact(
                worldPt, closestNormal, rotStat, staticIdx, dynamicIdx, std::sqrt(minDistSq)));
        }
    }
}

/* ========== Iterative BVH Traversal for Plane Detection ========== */

void CollisionSystem::traverseTreeAgainstPlane(
    const TreeNode& rootNode, const SpatialTree& tree,
    const HalfSpace3D& localPlane,
    const DynArray<Point3>& pointCloud,
    const DynArray<Triplet3i>& faces,
    const Matrix33& rotMat, const Point3& translation,
    IntType entityIdx, const Point3& worldPlaneNormal,
    DynArray<bool>& visitedPoints)
{
    // Use explicit stack for iterative traversal
    std::array<IntType, kMaxStackDepth> nodeStack;
    IntType stackTop = 0;
    nodeStack[stackTop++] = tree.rootIdx();
    
    while (stackTop > 0)
    {
        const IntType nodeIdx = nodeStack[--stackTop];
        const TreeNode& node = tree.nodeAt(nodeIdx);
        
        // Test AABB vs plane
        if (!nodeIntersectsPlane(node.volume, localPlane))
            continue;
        
        if (node.isTerminal())
        {
            // Process leaf: check all vertices in triangles
            const auto& indices = tree.faceOrdering();
            IntType primIdx = node.leafStart;
            const IntType primEnd = node.leafStart + node.leafCount;
            
            while (primIdx < primEnd)
            {
                const IntType faceIdx = indices[primIdx];
                const Triplet3i& face = faces[faceIdx];
                
                // Check each vertex
                IntType k = 0;
                do {
                    const IntType vIdx = face[k];
                    
                    if (!visitedPoints[vIdx])
                    {
                        visitedPoints[vIdx] = true;
                        
                        const Point3& localPt = pointCloud[vIdx];
                        const RealType dist = localPlane.evaluate(localPt);
                        
                        if (dist < static_cast<RealType>(0))
                        {
                            m_contactList.push_back(createPlaneContact(
                                localPt, rotMat, translation, entityIdx, worldPlaneNormal, -dist));
                        }
                    }
                    ++k;
                } while (k < 3);
                
                ++primIdx;
            }
        }
        else
        {
            // Push children (right first for depth-first left-to-right order)
            if (node.childRight != -1 && stackTop < kMaxStackDepth)
                nodeStack[stackTop++] = node.childRight;
            if (node.childLeft != -1 && stackTop < kMaxStackDepth)
                nodeStack[stackTop++] = node.childLeft;
        }
    }
}

/* ========== Iterative BVH Traversal for Point Query ========== */

void CollisionSystem::traverseTreeAgainstPoint(
    const TreeNode& rootNode, const SpatialTree& tree,
    const Point3& queryLocal,
    const DynArray<Point3>& pointCloud,
    const DynArray<Triplet3i>& faces,
    RealType& minDistSq, Point3& closestNormal,
    Point3& closestPoint, bool& foundContact)
{
    // Iterative traversal with priority by distance
    std::array<IntType, kMaxStackDepth> nodeStack;
    IntType stackTop = 0;
    nodeStack[stackTop++] = tree.rootIdx();
    
    while (stackTop > 0)
    {
        const IntType nodeIdx = nodeStack[--stackTop];
        const TreeNode& node = tree.nodeAt(nodeIdx);
        
        // Cull nodes farther than current minimum
        const RealType boxDistSq = pointToBoxDistanceSq(queryLocal, node.volume);
        if (boxDistSq > minDistSq)
            continue;
        
        if (node.isTerminal())
        {
            // Test all triangles in leaf
            const auto& indices = tree.faceOrdering();
            IntType primIdx = node.leafStart;
            const IntType primEnd = node.leafStart + node.leafCount;
            
            while (primIdx < primEnd)
            {
                const Triplet3i& face = faces[indices[primIdx]];
                
                if (testTrianglePenetration(queryLocal,
                                             pointCloud[face[0]],
                                             pointCloud[face[1]],
                                             pointCloud[face[2]],
                                             minDistSq, closestNormal, closestPoint))
                {
                    foundContact = true;
                }
                ++primIdx;
            }
        }
        else
        {
            // Push children
            if (node.childRight != -1 && stackTop < kMaxStackDepth)
                nodeStack[stackTop++] = node.childRight;
            if (node.childLeft != -1 && stackTop < kMaxStackDepth)
                nodeStack[stackTop++] = node.childLeft;
        }
    }
}

/* ========== GPU Implementation ========== */

#if defined(RIGID_USE_CUDA)

void CollisionSystem::performDeviceDetection(World& world) 
{
    reset();

    Boundaries& env = world.boundaries();
    const IntType entityCount = world.entityCount();

    // Entity-Environment detection
    IntType i = 0;
    while (i < entityCount)
    {
        DynamicEntity* entity = world.entity(i);
        bool canProcess = (entity != nullptr) && entity->hasSurface();
        
        if (canProcess)
        {
            const SpatialTree& tree = entity->surface().accelerator();
            
            if (tree.hasDeviceData())
            {
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
                IntType p = 0;
                while (p < Boundaries::planeCount())
                {
                    const HalfSpace3D& plane = env.planeAt(static_cast<Boundaries::PlaneId>(p));
                    planes[p].direction = make_float3(plane.direction.x(),
                                                       plane.direction.y(),
                                                       plane.direction.z());
                    planes[p].offset = plane.distance;
                    ++p;
                }

                DynArray<gpu::DeviceContact> deviceContacts;
                m_deviceDetector->detectEntityEnvironment(i, transform, planes.data(),
                                                           static_cast<IntType>(planes.size()),
                                                           deviceContacts);
                convertDeviceContacts(deviceContacts);
            }
        }
        ++i;
    }

    // Entity-Entity detection with GPU broadphase
    DynArray<std::pair<IntType, IntType>> candidatePairs;
    deviceBroadPhase(world, candidatePairs);

    size_t pairIdx = 0;
    while (pairIdx < candidatePairs.size())
    {
        const auto& pair = candidatePairs[pairIdx];
        const IntType idxA = pair.first;
        const IntType idxB = pair.second;

        DynamicEntity* entityA = world.entity(idxA);
        DynamicEntity* entityB = world.entity(idxB);

        bool pairValid = (entityA != nullptr) && (entityB != nullptr);
        pairValid = pairValid && entityA->hasSurface() && entityB->hasSurface();

        if (pairValid)
        {
            const SpatialTree& treeA = entityA->surface().accelerator();
            const SpatialTree& treeB = entityB->surface().accelerator();
            
            if (treeA.hasDeviceData() && treeB.hasDeviceData())
            {
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
                    idxA, transformA, treeA.deviceBuilder(),
                    idxB, transformB, treeB.deviceBuilder(),
                    deviceContacts);
                convertDeviceContacts(deviceContacts);
            }
        }
        ++pairIdx;
    }
}

void CollisionSystem::convertDeviceContacts(const DynArray<gpu::DeviceContact>& deviceContacts) 
{
    size_t idx = deviceContacts.size();
    while (idx > 0)
    {
        --idx;
        const auto& dc = deviceContacts[idx];
        
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
    const IntType entityCount = world.entityCount();
    
    if (entityCount < 2)
    {
        candidatePairs.clear();
        return;
    }

    DynArray<Point3> boundsMin(entityCount);
    DynArray<Point3> boundsMax(entityCount);

    IntType i = entityCount;
    while (i > 0)
    {
        --i;
        DynamicEntity* entity = world.entity(i);
        
        if (entity != nullptr && entity->hasSurface())
        {
            BoundingBox3D& wb = entity->worldExtent();
            boundsMin[i] = wb.corner_lo;
            boundsMax[i] = wb.corner_hi;
        }
        else
        {
            constexpr RealType kInfinity = static_cast<RealType>(1e30);
            boundsMin[i] = Point3(kInfinity, kInfinity, kInfinity);
            boundsMax[i] = Point3(-kInfinity, -kInfinity, -kInfinity);
        }
    }

    DynArray<gpu::EntityPair> devicePairs;
    m_deviceDetector->performBroadPhase(boundsMin, boundsMax, devicePairs);

    candidatePairs.resize(devicePairs.size());
    size_t pairIdx = devicePairs.size();
    while (pairIdx > 0)
    {
        --pairIdx;
        candidatePairs[pairIdx] = std::make_pair(devicePairs[pairIdx].entityA, devicePairs[pairIdx].entityB);
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
