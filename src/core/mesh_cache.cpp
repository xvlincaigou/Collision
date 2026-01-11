/*
 * Implementation: SurfaceResourcePool class
 */
#include "mesh_cache.h"
#include "mesh.h"
#include "accel/bvh.h"

namespace phys3d {

SurfaceResourcePool& SurfaceResourcePool::global() 
{
    static SurfaceResourcePool singletonInstance;
    return singletonInstance;
}

JointPtr<TriangleSurface> SurfaceResourcePool::fetch(const TextType& resourcePath, bool constructTree) 
{
    std::lock_guard<std::mutex> scopedLock(m_guard);

    auto iter = m_storage.find(resourcePath);
    if (iter != m_storage.end()) 
    {
        if (auto existingSurface = iter->second.lock()) 
        {
            return existingSurface;
        }
    }

    auto freshSurface = loadFromDisk(resourcePath, constructTree);
    if (freshSurface) 
    {
        m_storage[resourcePath] = freshSurface;
    }

    return freshSurface;
}

void SurfaceResourcePool::purgeAll() 
{
    std::lock_guard<std::mutex> scopedLock(m_guard);
    m_storage.clear();
}

void SurfaceResourcePool::removeExpired() 
{
    std::lock_guard<std::mutex> scopedLock(m_guard);

    auto iter = m_storage.begin();
    while (iter != m_storage.end()) 
    {
        if (iter->second.expired()) 
        {
            iter = m_storage.erase(iter);
        } 
        else 
        {
            ++iter;
        }
    }
}

size_t SurfaceResourcePool::entryCount() const 
{
    std::lock_guard<std::mutex> scopedLock(m_guard);
    return m_storage.size();
}

JointPtr<TriangleSurface> SurfaceResourcePool::loadFromDisk(const TextType& resourcePath, bool constructTree) 
{
    auto newSurface = std::make_shared<TriangleSurface>();
    if (!newSurface->parseWavefrontFile(resourcePath, constructTree)) 
    {
        return nullptr;
    }
    return newSurface;
}

}  // namespace phys3d
