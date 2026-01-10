/**
 * @file mesh_cache.cpp
 * @brief Implementation of the MeshCache class.
 */
#include "mesh_cache.h"
#include "mesh.h"
#include "accel/bvh.h"

namespace rigid {

MeshCache& MeshCache::instance() {
    static MeshCache inst;
    return inst;
}

SharedPtr<Mesh> MeshCache::acquire(const String& path, bool buildBVH) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already cached and still valid
    auto it = cache_.find(path);
    if (it != cache_.end()) {
        if (auto mesh = it->second.lock()) {
            return mesh;
        }
        // Entry expired, will be replaced
    }

    // Load new mesh
    auto mesh = loadMesh(path, buildBVH);
    if (mesh) {
        cache_[path] = mesh;
    }

    return mesh;
}

void MeshCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

void MeshCache::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = cache_.begin(); it != cache_.end();) {
        if (it->second.expired()) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t MeshCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

SharedPtr<Mesh> MeshCache::loadMesh(const String& path, bool buildBVH) {
    auto mesh = std::make_shared<Mesh>();
    if (!mesh->loadFromOBJ(path, buildBVH)) {
        return nullptr;
    }
    return mesh;
}

}  // namespace rigid
