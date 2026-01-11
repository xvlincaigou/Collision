/*
 * Implementation: World class
 */
#include "scene.h"

namespace phys3d {

DynamicEntity& World::spawnEntity(const TextType& identifier) 
{
    m_entities.emplace_back(std::make_unique<DynamicEntity>(identifier));
    return *m_entities.back();
}

DynamicEntity* World::entity(IntType idx) 
{
    if (idx < 0 || idx >= static_cast<IntType>(m_entities.size())) 
    {
        return nullptr;
    }
    return m_entities[idx].get();
}

const DynamicEntity* World::entity(IntType idx) const 
{
    if (idx < 0 || idx >= static_cast<IntType>(m_entities.size())) 
    {
        return nullptr;
    }
    return m_entities[idx].get();
}

}  // namespace phys3d
