/**
 * @file simulator.cpp
 * @brief Implementation of the Simulator class.
 */
#include "simulator.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace rigid {

void Simulator::initialize() {
    integrator_.initialize(scene_);
    frameCount_ = 0;
}

void Simulator::reset() {
    scene_.clearBodies();
    frameCount_ = 0;
}

void Simulator::step() {
    integrator_.step(scene_);
    ++frameCount_;
}

RigidBody& Simulator::addBody(const String& name, const String& meshPath,
                              Float mass) {
    return addBody(name, meshPath, mass, 1.0f, 0.5f, 0.5f);
}

RigidBody& Simulator::addBody(const String& name, const String& meshPath,
                              Float mass, Float scale, Float restitution, Float friction) {
    RigidBody& body = scene_.createBody(name);

    auto meshPtr = MeshCache::instance().acquire(meshPath, true);
    body.setMesh(meshPtr);

    BodyProperties props;
    props.setMass(mass);
    props.restitution = restitution;
    props.friction = friction;
    props.finalize();
    body.setProperties(props);

    // Set initial scale
    BodyState state = body.state();
    state.scale = scale;
    body.setState(state);

    return body;
}

void Simulator::setEnvironmentBounds(const Vec3& minCorner, const Vec3& maxCorner) {
    scene_.setEnvironmentBounds(minCorner, maxCorner);
}

void Simulator::exportFrame(const String& filename) {
    exportSceneToOBJ(scene_, filename);
}

// ============================================================================
// Scene Export
// ============================================================================

bool exportSceneToOBJ(Scene& scene, const String& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    const Int n = scene.bodyCount();

    // Precompute vertex counts and offsets
    Vector<size_t> vertexCounts(n, 0);
    Vector<size_t> vertexOffsets(n, 0);
    Vector<bool> valid(n, false);

    size_t runningOffset = 1;  // OBJ indices start at 1
    for (Int i = 0; i < n; ++i) {
        RigidBody* body = scene.body(i);
        if (!body || !body->hasMesh()) {
            vertexOffsets[i] = runningOffset;
            continue;
        }

        const Mesh& mesh = body->mesh();
        valid[i] = true;
        vertexCounts[i] = mesh.vertexCount();
        vertexOffsets[i] = runningOffset;
        runningOffset += vertexCounts[i];
    }

    // Generate OBJ blocks in parallel
    Vector<String> blocks(n);

    tbb::parallel_for(tbb::blocked_range<Int>(0, n),
        [&](const tbb::blocked_range<Int>& range) {
            for (Int i = range.begin(); i != range.end(); ++i) {
                if (!valid[i]) continue;

                RigidBody* body = scene.body(i);
                if (!body || !body->hasMesh()) continue;

                const Mesh& mesh = body->mesh();
                const BodyState& state = body->state();
                const auto& verts = mesh.vertices();
                const auto& tris = mesh.triangles();
                const size_t offset = vertexOffsets[i];

                std::ostringstream ss;
                ss.setf(std::ios::fmtflags(0), std::ios::floatfield);
                ss.precision(9);

                // Object name
                ss << "o " << mesh.name() << "_" << i << "\n";

                // Vertices in world space
                for (const auto& vLocal : verts) {
                    Vec3 vWorld = state.localToWorld(vLocal);
                    ss << "v " << vWorld.x() << " "
                       << vWorld.y() << " "
                       << vWorld.z() << "\n";
                }

                // Faces (OBJ is 1-based)
                for (const auto& t : tris) {
                    ss << "f " << (t[0] + offset) << " "
                       << (t[1] + offset) << " "
                       << (t[2] + offset) << "\n";
                }

                blocks[i] = ss.str();
            }
        });

    // Write all blocks to file
    for (Int i = 0; i < n; ++i) {
        if (valid[i]) {
            file << blocks[i];
        }
    }

    return true;
}

}  // namespace rigid
