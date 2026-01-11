/*
 * High-Level Simulation Controller
 */
#ifndef PHYS3D_SIMULATION_CONTROLLER_HPP
#define PHYS3D_SIMULATION_CONTROLLER_HPP

#include "core/common.h"
#include "core/mesh_cache.h"
#include "physics/integrator.h"
#include "scene/scene.h"

namespace phys3d {

/*
 * SimulationController - Main interface for running physics simulations
 */
class SimulationController 
{
public:
    struct EntityDescriptor 
    {
        TextType identifier;
        TextType geometryPath;
        RealType mass = static_cast<RealType>(1);
        RealType scale = static_cast<RealType>(1);
        RealType bounciness = static_cast<RealType>(0.5);
        RealType friction = static_cast<RealType>(0.5);

        static EntityDescriptor Standard(const TextType& id, const TextType& path, RealType m = static_cast<RealType>(1)) 
        {
            EntityDescriptor desc;
            desc.identifier = id;
            desc.geometryPath = path;
            desc.mass = m;
            return desc;
        }
    };

    SimulationController();

    void initialize();
    void tick();
    void reset();

    DynamicEntity& createEntity(const TextType& identifier, const TextType& geometryPath,
                                 RealType mass = static_cast<RealType>(1));

    DynamicEntity& createEntity(const TextType& identifier, const TextType& geometryPath,
                                 RealType mass, RealType scale, RealType bounciness, RealType friction);

    DynamicEntity& createEntity(const EntityDescriptor& descriptor);

    void setBoundaryLimits(const Point3& lowerBound, const Point3& upperBound);

    void setIterationLimit(IntType limit) { m_integrator.setIterationLimit(limit); }
    void setConvergenceThreshold(RealType thresh) { m_integrator.setConvergenceThreshold(thresh); }
    void setDeltaTime(RealType dt) { m_integrator.setDeltaTime(dt); }

    /* Export */
    void exportFrame(const TextType& outputPath);

    /* Accessors */
    [[nodiscard]] World& world() { return m_world; }
    [[nodiscard]] const World& world() const { return m_world; }

    [[nodiscard]] IntType currentFrame() const { return m_frameNumber; }

private:
    DynamicEntity& instantiateEntity(const TextType& identifier, const TextType& geometryPath);
    void configureMaterial(DynamicEntity& entity, RealType mass, RealType bounciness, RealType friction);
    void configureScale(DynamicEntity& entity, RealType scale);
    void resetFrameCounter();

    World m_world;
    TimeIntegrator m_integrator;
    IntType m_frameNumber = 0;
};

}  // namespace phys3d

namespace rigid {
    using Simulator = phys3d::SimulationController;
}

#endif // PHYS3D_SIMULATION_CONTROLLER_HPP
