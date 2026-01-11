/*
 * Thread-Safe Resource Pool for Triangle Surfaces
 */
#ifndef PHYS3D_RESOURCE_POOL_HPP
#define PHYS3D_RESOURCE_POOL_HPP

#include "types.h"

#include <mutex>
#include <unordered_map>

namespace phys3d {

class TriangleSurface;

/*
 * SurfaceResourcePool - Singleton cache with automatic cleanup
 * Uses weak references to allow unused surfaces to be freed
 */
class SurfaceResourcePool 
{
public:
    static SurfaceResourcePool& global();

    JointPtr<TriangleSurface> fetch(const TextType& resourcePath, bool constructTree = true);

    void purgeAll();
    void removeExpired();
    [[nodiscard]] size_t entryCount() const;

private:
    SurfaceResourcePool() = default;
    ~SurfaceResourcePool() = default;

    SurfaceResourcePool(const SurfaceResourcePool&) = delete;
    SurfaceResourcePool& operator=(const SurfaceResourcePool&) = delete;

    JointPtr<TriangleSurface> loadFromDisk(const TextType& resourcePath, bool constructTree);

    using ResourceMap = std::unordered_map<TextType, std::weak_ptr<TriangleSurface>>;

    mutable std::mutex m_guard;
    ResourceMap m_storage;
};

}  // namespace phys3d

namespace rigid {
    using MeshCache = phys3d::SurfaceResourcePool;
}

#endif // PHYS3D_RESOURCE_POOL_HPP
