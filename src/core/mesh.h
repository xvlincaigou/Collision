/**
 * @file mesh.h
 * @brief Triangle mesh class with BVH acceleration support.
 */
#pragma once

#include "common.h"

namespace rigid {

// Forward declaration
class BVH;

/**
 * @class Mesh
 * @brief Represents a triangle mesh with vertices, faces, and optional BVH.
 */
class Mesh {
public:
    Mesh() = default;
    explicit Mesh(String name);
    ~Mesh();

    // Non-copyable but movable
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&&) noexcept;
    Mesh& operator=(Mesh&&) noexcept;

    // ========================================================================
    // I/O Operations
    // ========================================================================

    /**
     * @brief Load mesh from an OBJ file.
     * @param path Path to the OBJ file.
     * @param buildBVH Whether to build BVH after loading.
     * @return True if loading succeeded.
     */
    bool loadFromOBJ(const String& path, bool buildBVH = true);

    /// Clear all mesh data
    void clear();

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const String& name() const { return name_; }
    [[nodiscard]] const Vector<Vec3>& vertices() const { return vertices_; }
    [[nodiscard]] const Vector<Triangle>& triangles() const { return triangles_; }
    [[nodiscard]] const Vector<Vec3>& normals() const { return normals_; }
    [[nodiscard]] const AABB& localBounds() const { return localBounds_; }

    [[nodiscard]] Int vertexCount() const { return static_cast<Int>(vertices_.size()); }
    [[nodiscard]] Int triangleCount() const { return static_cast<Int>(triangles_.size()); }
    [[nodiscard]] bool isEmpty() const { return vertices_.empty(); }

    // ========================================================================
    // BVH Management
    // ========================================================================

    /// Check if BVH is built and valid
    [[nodiscard]] bool hasBVH() const;

    /// Get the BVH (returns empty BVH if not built)
    [[nodiscard]] const BVH& bvh() const;

    /// Rebuild the BVH from current geometry
    void rebuildBVH();

    /// Mark BVH as needing rebuild
    void invalidateBVH() { bvhDirty_ = true; }

private:
    void updateBounds();

    String name_;
    Vector<Vec3> vertices_;
    Vector<Triangle> triangles_;
    Vector<Vec3> normals_;

    AABB localBounds_;
    bool boundsDirty_ = true;

    UniquePtr<BVH> bvh_;
    bool bvhDirty_ = true;
};

}  // namespace rigid
