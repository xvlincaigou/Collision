/*
 * Simulation World Container
 */
#ifndef PHYS3D_WORLD_HPP
#define PHYS3D_WORLD_HPP

#include "core/common.h"
#include "environment.h"
#include "physics/rigid_body.h"

namespace phys3d {

/*
 * World - Contains all simulation objects
 */
class World 
{
public:
    World() = default;

    /* Entity Management */
    DynamicEntity& spawnEntity(const TextType& identifier);

    [[nodiscard]] DynamicEntity* entity(IntType idx);
    [[nodiscard]] const DynamicEntity* entity(IntType idx) const;

    [[nodiscard]] DynArray<SolePtr<DynamicEntity>>& entities() { return m_entities; }
    [[nodiscard]] const DynArray<SolePtr<DynamicEntity>>& entities() const { return m_entities; }

    [[nodiscard]] IntType entityCount() const { return static_cast<IntType>(m_entities.size()); }

    void destroyAllEntities() { m_entities.clear(); }

    [[nodiscard]] bool hasNoEntities() const { return m_entities.empty(); }

    /* Boundaries */
    [[nodiscard]] Boundaries& boundaries() { return m_boundaries; }
    [[nodiscard]] const Boundaries& boundaries() const { return m_boundaries; }

    void configureBounds(const Point3& lowerCorner, const Point3& upperCorner) 
    {
        m_boundaries.defineBounds(lowerCorner, upperCorner);
    }

private:
    DynArray<SolePtr<DynamicEntity>> m_entities;
    Boundaries m_boundaries;
};

}  // namespace phys3d

namespace rigid {
    using Scene = phys3d::World;
}

#endif // PHYS3D_WORLD_HPP
