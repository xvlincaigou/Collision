/**
 * @file mesh.cpp
 * @brief Implementation of the Mesh class.
 */
#include "mesh.h"
#include "accel/bvh.h"

#include <fstream>
#include <sstream>

namespace rigid {

Mesh::Mesh(String name) : name_(std::move(name)) {}

Mesh::~Mesh() = default;

Mesh::Mesh(Mesh&&) noexcept = default;
Mesh& Mesh::operator=(Mesh&&) noexcept = default;

bool Mesh::loadFromOBJ(const String& path, bool buildBVH) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    clear();

    String line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        String tag;
        if (!(iss >> tag)) {
            continue;
        }

        if (tag == "v") {
            // Vertex position
            Float x, y, z;
            iss >> x >> y >> z;
            vertices_.emplace_back(x, y, z);
        } else if (tag == "vn") {
            // Vertex normal
            Float x, y, z;
            iss >> x >> y >> z;
            normals_.emplace_back(x, y, z);
        } else if (tag == "f") {
            // Face (triangulate if needed)
            Vector<Int> indices;
            String token;

            while (iss >> token) {
                std::istringstream tokenStream(token);
                String indexStr;
                std::getline(tokenStream, indexStr, '/');

                if (!indexStr.empty()) {
                    // OBJ indices are 1-based
                    indices.push_back(std::stoi(indexStr) - 1);
                }
            }

            // Triangulate polygon (fan triangulation)
            if (indices.size() >= 3) {
                for (size_t i = 1; i + 1 < indices.size(); ++i) {
                    triangles_.emplace_back(indices[0], indices[i], indices[i + 1]);
                }
            }
        }
    }

    updateBounds();
    bvhDirty_ = true;

    if (buildBVH) {
        rebuildBVH();
    }

    return true;
}

void Mesh::clear() {
    vertices_.clear();
    triangles_.clear();
    normals_.clear();
    localBounds_.reset();
    boundsDirty_ = true;
    bvh_.reset();
    bvhDirty_ = true;
}

bool Mesh::hasBVH() const {
    return bvh_ && bvh_->isBuilt();
}

const BVH& Mesh::bvh() const {
    static const BVH emptyBVH;
    return bvh_ ? *bvh_ : emptyBVH;
}

void Mesh::rebuildBVH() {
    if (boundsDirty_) {
        updateBounds();
    }

    if (!bvh_) {
        bvh_ = std::make_unique<BVH>();
    }

#if defined(RIGID_USE_CUDA)
    bvh_->buildGPU(vertices_, triangles_);
#else
    bvh_->build(vertices_, triangles_);
#endif

    bvhDirty_ = false;
}

void Mesh::updateBounds() {
    localBounds_.reset();
    for (const auto& v : vertices_) {
        localBounds_.expand(v);
    }
    boundsDirty_ = false;
}

}  // namespace rigid
