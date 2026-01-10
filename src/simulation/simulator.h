/**
 * @file simulator.h
 * @brief High-level simulation controller.
 */
#pragma once

#include "core/common.h"
#include "core/mesh_cache.h"
#include "physics/integrator.h"
#include "scene/scene.h"

namespace rigid {

/**
 * @class Simulator
 * @brief High-level interface for running rigid body simulations.
 */
class Simulator {
public:
    Simulator() = default;

    /// Initialize the simulator
    void initialize();

    /// Reset to initial state
    void reset();

    /// Advance simulation by one time step
    void step();

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Add a rigid body with mesh and mass
    RigidBody& addBody(const String& name, const String& meshPath,
                       Float mass = 1.0f);

    /// Add a rigid body with full customization
    RigidBody& addBody(const String& name, const String& meshPath,
                       Float mass, Float scale, Float restitution, Float friction);

    /// Set the simulation environment bounds
    void setEnvironmentBounds(const Vec3& minCorner, const Vec3& maxCorner);

    /// Set solver parameters
    void setMaxIterations(Int iter) { integrator_.setMaxIterations(iter); }
    void setTolerance(Float tol) { integrator_.setTolerance(tol); }
    void setTimeStep(Float dt) { integrator_.setTimeStep(dt); }

    // ========================================================================
    // Export
    // ========================================================================

    /// Export current frame to OBJ file
    void exportFrame(const String& filename);

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] Scene& scene() { return scene_; }
    [[nodiscard]] const Scene& scene() const { return scene_; }

    [[nodiscard]] Int currentFrame() const { return frameCount_; }

private:
    Scene scene_;
    Integrator integrator_;
    Int frameCount_ = 0;
};

// ============================================================================
// Scene Export Utility
// ============================================================================

/**
 * @brief Export a scene to an OBJ file.
 * @param scene The scene to export.
 * @param filename Output filename.
 * @return True if successful.
 */
bool exportSceneToOBJ(Scene& scene, const String& filename);

}  // namespace rigid
