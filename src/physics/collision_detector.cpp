/**
 * @file collision_detector.cpp
 * @brief Implementation of the CollisionDetector class.
 */
#include "collision_detector.h"

#include <cmath>
#include <iostream>

#if defined(RIGID_USE_CUDA)
#include "gpu/collision_detector.cuh"
#include "gpu/broadphase.cuh"
#endif

namespace rigid {

// ============================================================================
// Constructor/Destructor
// ============================================================================

#if defined(RIGID_USE_CUDA)

CollisionDetector::CollisionDetector() : useGPU_(true) {
    gpuDetector_ = std::make_unique<gpu::CollisionDetectorGPU>();
}

CollisionDetector::~CollisionDetector() = default;

#else

CollisionDetector::CollisionDetector() : useGPU_(false) {}
CollisionDetector::~CollisionDetector() = default;

#endif

// ============================================================================
// Main Detection Entry Point
// ============================================================================

void CollisionDetector::detectCollisions(Scene& scene) {
    if (useGPU_) {
        detectCollisionsGPU(scene);
        return;
    }

    clear();

    Environment& env = scene.environment();
    Int bodyCount = scene.bodyCount();

    // Body vs Environment
    for (Int i = 0; i < bodyCount; ++i) {
        RigidBody* body = scene.body(i);
        if (!body || !body->hasMesh()) continue;
        detectBodyEnvironment(i, *body, env);
    }

    // Body vs Body (O(N^2) broadphase)
    for (Int i = 0; i < bodyCount; ++i) {
        RigidBody* bodyA = scene.body(i);
        if (!bodyA || !bodyA->hasMesh()) continue;

        for (Int j = i + 1; j < bodyCount; ++j) {
            RigidBody* bodyB = scene.body(j);
            if (!bodyB || !bodyB->hasMesh()) continue;

            // Broadphase: AABB intersection
            if (bodyA->worldBounds().intersects(bodyB->worldBounds())) {
                detectBodyBody(i, *bodyA, j, *bodyB);
            }
        }
    }
}

// ============================================================================
// Body vs Environment
// ============================================================================

void CollisionDetector::detectBodyEnvironment(Int bodyIdx, RigidBody& body,
                                               Environment& env) {
    Mesh& mesh = body.mesh();
    const BodyState& state = body.state();
    Mat3 rot = state.rotationMatrix();
    Vec3 pos = state.position;

    const Vector<Vec3>& vertices = mesh.vertices();
    static Vector<bool> visitedVerts;
    visitedVerts.resize(vertices.size());

    for (Int pIdx = 0; pIdx < Environment::boundaryCount(); ++pIdx) {
        Plane& planeWorld = env.plane(static_cast<Environment::BoundaryId>(pIdx));

        // Quick AABB vs plane test
        const AABB& wb = body.worldBounds();
        Vec3 supportMin;
        for (int k = 0; k < 3; ++k) {
            supportMin[k] = (planeWorld.normal[k] >= 0) ? wb.min_pt[k] : wb.max_pt[k];
        }
        Float distMin = supportMin.dot(planeWorld.normal) - planeWorld.offset;

        if (distMin > 1e-4f) continue;  // No collision possible

        // Transform plane to local space
        Plane planeLocal;
        planeLocal.normal = rot.transpose() * planeWorld.normal;
        planeLocal.offset = planeWorld.offset - planeWorld.normal.dot(pos);

        if (mesh.hasBVH()) {
            std::fill(visitedVerts.begin(), visitedVerts.end(), false);
            traverseBVHPlane(
                mesh.bvh().node(mesh.bvh().rootIndex()),
                mesh.bvh(), planeLocal, vertices, mesh.triangles(),
                rot, pos, bodyIdx, planeWorld.normal, visitedVerts);
        } else {
            // Brute force fallback
            for (const auto& vLocal : vertices) {
                Float distLocal = vLocal.dot(planeLocal.normal) - planeLocal.offset;
                if (distLocal < 0.0f) {
                    Contact contact;
                    contact.bodyIndexA = -1;  // Environment
                    contact.bodyIndexB = bodyIdx;
                    contact.depth = -distLocal;
                    contact.normal = planeWorld.normal;
                    contact.position = rot * vLocal + pos;
                    contacts_.push_back(contact);
                }
            }
        }
    }
}

// ============================================================================
// Body vs Body
// ============================================================================

void CollisionDetector::detectBodyBody(Int idxA, RigidBody& bodyA,
                                        Int idxB, RigidBody& bodyB) {
    // Symmetric check: vertices of A vs mesh B, and vice versa
    detectVertexMesh(idxA, bodyA, idxB, bodyB);
    detectVertexMesh(idxB, bodyB, idxA, bodyA);
}

void CollisionDetector::detectVertexMesh(Int dynamicIdx, RigidBody& dynamic,
                                          Int staticIdx, RigidBody& statik) {
    const auto& vertsDyn = dynamic.mesh().vertices();
    const AABB& boundsStatic = statik.worldBounds();

    Mat3 rotDyn = dynamic.state().rotationMatrix();
    Vec3 posDyn = dynamic.state().position;
    Mat3 rotStat = statik.state().rotationMatrix();
    Vec3 posStat = statik.state().position;
    Mat3 rotStatInv = rotStat.transpose();

    const Mesh& meshStat = statik.mesh();
    const auto& vertsStat = meshStat.vertices();
    const auto& trisStat = meshStat.triangles();

    constexpr Float kCollisionThreshold = 0.5f;

    for (const auto& vLocal : vertsDyn) {
        Vec3 vWorld = rotDyn * vLocal + posDyn;

        // Quick world AABB check
        if (!boundsStatic.contains(vWorld)) {
            continue;
        }

        // Transform to static body's local space
        Vec3 pLocalStat = rotStatInv * (vWorld - posStat);

        Float minDistSq = kCollisionThreshold * kCollisionThreshold;
        Vec3 closestNormalLocal = Vec3::Zero();
        Vec3 closestPtLocal = Vec3::Zero();
        bool found = false;

        if (meshStat.hasBVH()) {
            traverseBVHPoint(
                meshStat.bvh().node(meshStat.bvh().rootIndex()),
                meshStat.bvh(), pLocalStat, vertsStat, trisStat,
                minDistSq, closestNormalLocal, closestPtLocal, found);
        } else {
            // Brute force fallback
            const AABB& lb = meshStat.localBounds();
            if (lb.contains(pLocalStat)) {
                for (const auto& tri : trisStat) {
                    const Vec3& p0 = vertsStat[tri[0]];
                    const Vec3& p1 = vertsStat[tri[1]];
                    const Vec3& p2 = vertsStat[tri[2]];
                    if (checkTriangleCollision(pLocalStat, p0, p1, p2,
                                               minDistSq, closestNormalLocal,
                                               closestPtLocal)) {
                        found = true;
                    }
                }
            }
        }

        if (found) {
            Contact contact;
            contact.bodyIndexA = staticIdx;
            contact.bodyIndexB = dynamicIdx;
            contact.depth = std::sqrt(minDistSq);
            contact.normal = rotStat * closestNormalLocal;
            contact.position = vWorld;
            contacts_.push_back(contact);
        }
    }
}

// ============================================================================
// BVH Traversal
// ============================================================================

void CollisionDetector::traverseBVHPlane(
    const BVHNode& node, const BVH& bvh,
    const Plane& planeLocal,
    const Vector<Vec3>& vertices,
    const Vector<Triangle>& triangles,
    const Mat3& bodyRot, const Vec3& bodyPos,
    Int bodyIdx, const Vec3& planeWorldNormal,
    Vector<bool>& visitedVerts)
{
    // Test AABB vs plane
    Vec3 center = node.bounds.center();
    Vec3 extents = node.bounds.extents();

    Float r = extents.x() * std::abs(planeLocal.normal.x()) +
              extents.y() * std::abs(planeLocal.normal.y()) +
              extents.z() * std::abs(planeLocal.normal.z());
    Float s = planeLocal.normal.dot(center) - planeLocal.offset;

    if (s > r) return;  // AABB is entirely on positive side

    if (node.isLeaf()) {
        const auto& indices = bvh.primitiveIndices();
        Int end = node.firstPrim + node.primCount;

        for (Int i = node.firstPrim; i < end; ++i) {
            Int triIdx = indices[i];
            const Triangle& tri = triangles[triIdx];

            for (int k = 0; k < 3; ++k) {
                Int vIdx = tri[k];
                if (visitedVerts[vIdx]) continue;
                visitedVerts[vIdx] = true;

                const Vec3& vLocal = vertices[vIdx];
                Float dist = vLocal.dot(planeLocal.normal) - planeLocal.offset;

                if (dist < 0.0f) {
                    Contact contact;
                    contact.bodyIndexA = -1;
                    contact.bodyIndexB = bodyIdx;
                    contact.depth = -dist;
                    contact.normal = planeWorldNormal;
                    contact.position = bodyRot * vLocal + bodyPos;
                    contacts_.push_back(contact);
                }
            }
        }
    } else {
        if (node.left != -1) {
            traverseBVHPlane(bvh.node(node.left), bvh, planeLocal,
                             vertices, triangles, bodyRot, bodyPos,
                             bodyIdx, planeWorldNormal, visitedVerts);
        }
        if (node.right != -1) {
            traverseBVHPlane(bvh.node(node.right), bvh, planeLocal,
                             vertices, triangles, bodyRot, bodyPos,
                             bodyIdx, planeWorldNormal, visitedVerts);
        }
    }
}

void CollisionDetector::traverseBVHPoint(
    const BVHNode& node, const BVH& bvh,
    const Vec3& pointLocal,
    const Vector<Vec3>& vertices,
    const Vector<Triangle>& triangles,
    Float& minDistSq, Vec3& closestNormal,
    Vec3& closestPos, bool& found)
{
    // Compute squared distance to AABB
    Float distSq = 0.0f;
    for (int i = 0; i < 3; ++i) {
        if (pointLocal[i] < node.bounds.min_pt[i]) {
            Float d = node.bounds.min_pt[i] - pointLocal[i];
            distSq += d * d;
        } else if (pointLocal[i] > node.bounds.max_pt[i]) {
            Float d = pointLocal[i] - node.bounds.max_pt[i];
            distSq += d * d;
        }
    }

    if (distSq > minDistSq) return;

    if (node.isLeaf()) {
        const auto& indices = bvh.primitiveIndices();
        Int end = node.firstPrim + node.primCount;

        for (Int i = node.firstPrim; i < end; ++i) {
            const Triangle& tri = triangles[indices[i]];
            const Vec3& p0 = vertices[tri[0]];
            const Vec3& p1 = vertices[tri[1]];
            const Vec3& p2 = vertices[tri[2]];

            if (checkTriangleCollision(pointLocal, p0, p1, p2,
                                       minDistSq, closestNormal, closestPos)) {
                found = true;
            }
        }
    } else {
        if (node.left != -1) {
            traverseBVHPoint(bvh.node(node.left), bvh, pointLocal,
                             vertices, triangles, minDistSq,
                             closestNormal, closestPos, found);
        }
        if (node.right != -1) {
            traverseBVHPoint(bvh.node(node.right), bvh, pointLocal,
                             vertices, triangles, minDistSq,
                             closestNormal, closestPos, found);
        }
    }
}

// ============================================================================
// GPU Implementation
// ============================================================================

#if defined(RIGID_USE_CUDA)

void CollisionDetector::detectCollisionsGPU(Scene& scene) {
    clear();

    Environment& env = scene.environment();
    Int bodyCount = scene.bodyCount();

    // Body vs Environment
    for (Int i = 0; i < bodyCount; ++i) {
        RigidBody* body = scene.body(i);
        if (!body || !body->hasMesh()) continue;

        const BVH& bvh = body->mesh().bvh();
        if (!bvh.hasGPUData()) continue;

        gpuDetector_->setBVH(bvh.gpuBuilder());

        gpu::BodyTransformDevice transform;
        const BodyState& state = body->state();
        transform.position = make_float3(state.position.x(),
                                         state.position.y(),
                                         state.position.z());
        transform.orientation = make_float4(state.orientation.x(),
                                            state.orientation.y(),
                                            state.orientation.z(),
                                            state.orientation.w());

        Vector<gpu::PlaneDevice> planes(Environment::boundaryCount());
        for (int p = 0; p < Environment::boundaryCount(); ++p) {
            const Plane& plane = env.plane(static_cast<Environment::BoundaryId>(p));
            planes[p].normal = make_float3(plane.normal.x(),
                                           plane.normal.y(),
                                           plane.normal.z());
            planes[p].offset = plane.offset;
        }

        Vector<gpu::ContactDevice> gpuContacts;
        gpuDetector_->detectBodyEnvironment(i, transform, planes.data(),
                                            static_cast<Int>(planes.size()),
                                            gpuContacts);
        convertGPUContacts(gpuContacts);
    }

    // Body vs Body with GPU broadphase
    Vector<std::pair<Int, Int>> collisionPairs;
    broadphaseGPU(scene, collisionPairs);

    for (const auto& pair : collisionPairs) {
        Int i = pair.first;
        Int j = pair.second;

        RigidBody* bodyA = scene.body(i);
        RigidBody* bodyB = scene.body(j);

        if (!bodyA || !bodyB) continue;
        if (!bodyA->hasMesh() || !bodyB->hasMesh()) continue;

        const BVH& bvhA = bodyA->mesh().bvh();
        const BVH& bvhB = bodyB->mesh().bvh();
        if (!bvhA.hasGPUData() || !bvhB.hasGPUData()) continue;

        gpu::BodyTransformDevice transformA, transformB;
        const BodyState& stateA = bodyA->state();
        const BodyState& stateB = bodyB->state();

        transformA.position = make_float3(stateA.position.x(),
                                          stateA.position.y(),
                                          stateA.position.z());
        transformA.orientation = make_float4(stateA.orientation.x(),
                                             stateA.orientation.y(),
                                             stateA.orientation.z(),
                                             stateA.orientation.w());
        transformB.position = make_float3(stateB.position.x(),
                                          stateB.position.y(),
                                          stateB.position.z());
        transformB.orientation = make_float4(stateB.orientation.x(),
                                             stateB.orientation.y(),
                                             stateB.orientation.z(),
                                             stateB.orientation.w());

        Vector<gpu::ContactDevice> gpuContacts;
        gpuDetector_->detectBodyBody(
            i, transformA, bvhA.gpuBuilder(),
            j, transformB, bvhB.gpuBuilder(),
            gpuContacts);
        convertGPUContacts(gpuContacts);
    }
}

void CollisionDetector::convertGPUContacts(
    const Vector<gpu::ContactDevice>& gpuContacts)
{
    for (const auto& gc : gpuContacts) {
        Contact contact;
        contact.bodyIndexA = gc.bodyIndexA;
        contact.bodyIndexB = gc.bodyIndexB;
        contact.position = Vec3(gc.position.x, gc.position.y, gc.position.z);
        contact.normal = Vec3(gc.normal.x, gc.normal.y, gc.normal.z);
        contact.depth = gc.depth;
        contacts_.push_back(contact);
    }
}

void CollisionDetector::broadphaseGPU(Scene& scene,
                                       Vector<std::pair<Int, Int>>& pairs)
{
    Int bodyCount = scene.bodyCount();
    if (bodyCount < 2) {
        pairs.clear();
        return;
    }

    Vector<Vec3> aabbMins(bodyCount);
    Vector<Vec3> aabbMaxs(bodyCount);

    for (Int i = 0; i < bodyCount; ++i) {
        RigidBody* body = scene.body(i);
        if (body && body->hasMesh()) {
            AABB& wb = body->worldBounds();
            aabbMins[i] = wb.min_pt;
            aabbMaxs[i] = wb.max_pt;
        } else {
            aabbMins[i] = Vec3(1e30f, 1e30f, 1e30f);
            aabbMaxs[i] = Vec3(-1e30f, -1e30f, -1e30f);
        }
    }

    Vector<gpu::CollisionPair> gpuPairs;
    gpuDetector_->broadphaseDetect(aabbMins, aabbMaxs, gpuPairs);

    pairs.resize(gpuPairs.size());
    for (size_t i = 0; i < gpuPairs.size(); ++i) {
        pairs[i] = std::make_pair(gpuPairs[i].bodyA, gpuPairs[i].bodyB);
    }
}

#else

void CollisionDetector::detectCollisionsGPU(Scene& scene) {
    // Fallback to CPU
    useGPU_ = false;
    detectCollisions(scene);
}

void CollisionDetector::convertGPUContacts(const Vector<gpu::ContactDevice>&) {}

void CollisionDetector::broadphaseGPU(Scene&, Vector<std::pair<Int, Int>>&) {}

#endif

}  // namespace rigid
