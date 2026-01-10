/**
 * @file common.h
 * @brief Core type definitions and common utilities for the rigid body simulator.
 */
#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Third-party libraries
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tbb.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

// Build configuration
// #define RIGID_DEBUGGER
#define RIGID_USE_CUDA
// #undef RIGID_USE_CUDA

// CUDA types (must be included BEFORE namespace)
#if defined(RIGID_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace rigid {

// ============================================================================
// Fundamental type aliases
// ============================================================================

using Int     = int;
using Int64   = long int;
using UInt    = unsigned int;
using Float   = float;
using Double  = double;
using String  = std::string;

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using UniquePtr = std::unique_ptr<T>;

template <typename T>
using SharedPtr = std::shared_ptr<T>;

// ============================================================================
// Linear algebra types (Eigen-based)
// ============================================================================

using Vec3      = Eigen::Vector3f;
using Vec4      = Eigen::Vector4f;
using Mat3      = Eigen::Matrix3f;
using Mat4      = Eigen::Matrix4f;
using VecX      = Eigen::VectorXf;
using MatX      = Eigen::MatrixXf;
using Quat      = Eigen::Quaternionf;
using Triangle  = Eigen::Vector3i;
using SparseMat = Eigen::SparseMatrix<Float>;
using Triplet   = Eigen::Triplet<Float>;
using Triplets  = Vector<Triplet>;

// ============================================================================
// CUDA type aliases (redirect to global namespace)
// ============================================================================

#if defined(RIGID_USE_CUDA)

using Int3   = ::int3;
using Float3 = ::float3;
using Float4 = ::float4;

#endif  // RIGID_USE_CUDA

// ============================================================================
// Axis-Aligned Bounding Box
// ============================================================================

struct AABB {
    Vec3 min_pt;
    Vec3 max_pt;

    AABB() { reset(); }

    AABB(const Vec3& min_corner, const Vec3& max_corner)
        : min_pt(min_corner), max_pt(max_corner) {}

    void reset() {
        constexpr Float inf = std::numeric_limits<Float>::infinity();
        min_pt = Vec3(inf, inf, inf);
        max_pt = Vec3(-inf, -inf, -inf);
    }

    void expand(const Vec3& point) {
        min_pt = min_pt.cwiseMin(point);
        max_pt = max_pt.cwiseMax(point);
    }

    void merge(const AABB& other) {
        min_pt = min_pt.cwiseMin(other.min_pt);
        max_pt = max_pt.cwiseMax(other.max_pt);
    }

    [[nodiscard]] bool isValid() const {
        return (min_pt.array() <= max_pt.array()).all();
    }

    [[nodiscard]] Vec3 center() const {
        return (min_pt + max_pt) * 0.5f;
    }

    [[nodiscard]] Vec3 extents() const {
        return (max_pt - min_pt) * 0.5f;
    }

    [[nodiscard]] Vec3 size() const {
        return max_pt - min_pt;
    }

    [[nodiscard]] bool intersects(const AABB& other) const {
        return (min_pt.array() <= other.max_pt.array()).all() &&
               (max_pt.array() >= other.min_pt.array()).all();
    }

    [[nodiscard]] bool contains(const Vec3& point) const {
        return (point.array() >= min_pt.array()).all() &&
               (point.array() <= max_pt.array()).all();
    }
};

// ============================================================================
// Plane representation (normal + offset: n·x = d)
// ============================================================================

struct Plane {
    Vec3  normal = Vec3::Zero();
    Float offset = 0.0f;

    Plane() = default;
    Plane(const Vec3& n, Float d) : normal(n), offset(d) {}

    [[nodiscard]] Float signedDistance(const Vec3& point) const {
        return normal.dot(point) - offset;
    }
};

// ============================================================================
// Utility functions
// ============================================================================

/// Create a Vec3 with positive infinity components
inline Vec3 positiveInfinity() {
    constexpr Float inf = std::numeric_limits<Float>::infinity();
    return {inf, inf, inf};
}

/// Create a Vec3 with negative infinity components
inline Vec3 negativeInfinity() {
    constexpr Float inf = std::numeric_limits<Float>::infinity();
    return {-inf, -inf, -inf};
}

/// Clamp a value to be positive (at least epsilon)
inline Float clampPositive(Float value) {
    constexpr Float eps = std::numeric_limits<Float>::epsilon();
    return value < eps ? eps : value;
}

/// Compute the inverse of a symmetric positive-definite matrix
inline Mat3 invertSymmetric(const Mat3& M) {
    Eigen::LDLT<Mat3> ldlt(M);
    if (ldlt.info() == Eigen::Success) {
        return ldlt.solve(Mat3::Identity());
    }
    // Fallback to pseudo-inverse
    return M.completeOrthogonalDecomposition().pseudoInverse();
}

/// Compute global DOF index for a rigid body (6 DOF per body)
inline Int globalDofIndex(Int bodyIndex) {
    return bodyIndex * 6;
}

/// Check if a string starts with a given prefix
inline bool startsWith(const String& str, const char* prefix) {
    return str.compare(0, std::char_traits<char>::length(prefix), prefix) == 0;
}

}  // namespace rigid
