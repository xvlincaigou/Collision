/**
 * @file mesh_cache.h
 * @brief Thread-safe mesh caching system using weak pointers.
 */
#pragma once

#include "common.h"

#include <mutex>
#include <unordered_map>

namespace rigid {

class Mesh;

/**
 * @class MeshCache
 * @brief Singleton cache for mesh objects with automatic cleanup.
 *
 * Uses weak_ptr to allow meshes to be freed when no longer referenced.
 * Thread-safe for concurrent access.
 */
class MeshCache {
public:
    /// Get the global singleton instance
    static MeshCache& instance();

    /**
     * @brief Acquire a mesh from cache or load from file.
     * @param path Path to the mesh file.
     * @param buildBVH Whether to build BVH when loading.
     * @return Shared pointer to the mesh, or nullptr on failure.
     */
    SharedPtr<Mesh> acquire(const String& path, bool buildBVH = true);

    /// Clear all cached meshes
    void clear();

    /// Remove expired entries from cache
    void cleanup();

    /// Get the number of currently cached meshes
    [[nodiscard]] size_t size() const;

private:
    MeshCache() = default;
    ~MeshCache() = default;

    // Non-copyable
    MeshCache(const MeshCache&) = delete;
    MeshCache& operator=(const MeshCache&) = delete;

    SharedPtr<Mesh> loadMesh(const String& path, bool buildBVH);

    using CacheMap = std::unordered_map<String, std::weak_ptr<Mesh>>;

    mutable std::mutex mutex_;
    CacheMap cache_;
};

}  // namespace rigid
