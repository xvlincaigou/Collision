/*
 * Fundamental Types and Core Utilities for Physics Simulation Engine
 */
#ifndef PHYS3D_FOUNDATION_TYPES_HPP
#define PHYS3D_FOUNDATION_TYPES_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

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

// #define PHYS3D_ENABLE_DEBUG
#define RIGID_USE_CUDA
// #undef RIGID_USE_CUDA

#if defined(RIGID_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace phys3d {

/* =========== Primitive Type Definitions =========== */

typedef int                 IntType;
typedef long int            Int64Type;
typedef unsigned int        UIntType;
typedef float               RealType;
typedef double              DoubleType;
typedef std::string         TextType;

template <typename T>
using DynArray = std::vector<T>;

template <typename T>
using SolePtr = std::unique_ptr<T>;

template <typename T>
using JointPtr = std::shared_ptr<T>;

/* =========== Linear Algebra Primitives =========== */

typedef Eigen::Vector3f                     Point3;
typedef Eigen::Vector4f                     Point4;
typedef Eigen::Matrix3f                     Matrix33;
typedef Eigen::Matrix4f                     Matrix44;
typedef Eigen::VectorXf                     VectorN;
typedef Eigen::MatrixXf                     MatrixMN;
typedef Eigen::Quaternionf                  Rotation4;
typedef Eigen::Vector3i                     Triplet3i;
typedef Eigen::SparseMatrix<RealType>       SparseGrid;
typedef Eigen::Triplet<RealType>            SparseEntry;
typedef DynArray<SparseEntry>               SparseEntryList;

/* =========== CUDA Compatibility Layer =========== */

#if defined(RIGID_USE_CUDA)
typedef ::int3      CudaInt3;
typedef ::float3    CudaFloat3;
typedef ::float4    CudaFloat4;
#endif

/* =========== Bounding Volume Definition =========== */

struct BoundingBox3D 
{
    Point3 corner_lo;
    Point3 corner_hi;

    BoundingBox3D() { invalidate(); }

    BoundingBox3D(const Point3& lo, const Point3& hi)
        : corner_lo(lo), corner_hi(hi) {}

    void invalidate() 
    {
        constexpr RealType kInf = std::numeric_limits<RealType>::infinity();
        corner_lo = Point3(kInf, kInf, kInf);
        corner_hi = Point3(-kInf, -kInf, -kInf);
    }

    void enclose(const Point3& pt) 
    {
        corner_lo = corner_lo.cwiseMin(pt);
        corner_hi = corner_hi.cwiseMax(pt);
    }

    void unite(const BoundingBox3D& rhs) 
    {
        corner_lo = corner_lo.cwiseMin(rhs.corner_lo);
        corner_hi = corner_hi.cwiseMax(rhs.corner_hi);
    }

    [[nodiscard]] bool valid() const 
    {
        return (corner_lo.array() <= corner_hi.array()).all();
    }

    [[nodiscard]] Point3 midpoint() const 
    {
        return (corner_lo + corner_hi) * static_cast<RealType>(0.5);
    }

    [[nodiscard]] Point3 halfSize() const 
    {
        return (corner_hi - corner_lo) * static_cast<RealType>(0.5);
    }

    [[nodiscard]] Point3 dimensions() const 
    {
        return corner_hi - corner_lo;
    }

    [[nodiscard]] bool overlaps(const BoundingBox3D& rhs) const 
    {
        bool xOk = (corner_lo.x() <= rhs.corner_hi.x()) && (corner_hi.x() >= rhs.corner_lo.x());
        bool yOk = (corner_lo.y() <= rhs.corner_hi.y()) && (corner_hi.y() >= rhs.corner_lo.y());
        bool zOk = (corner_lo.z() <= rhs.corner_hi.z()) && (corner_hi.z() >= rhs.corner_lo.z());
        return xOk && yOk && zOk;
    }

    [[nodiscard]] bool enclosesPoint(const Point3& pt) const 
    {
        bool inX = (pt.x() >= corner_lo.x()) && (pt.x() <= corner_hi.x());
        bool inY = (pt.y() >= corner_lo.y()) && (pt.y() <= corner_hi.y());
        bool inZ = (pt.z() >= corner_lo.z()) && (pt.z() <= corner_hi.z());
        return inX && inY && inZ;
    }
};

/* =========== Half-Space Definition =========== */

struct HalfSpace3D 
{
    Point3   direction = Point3::Zero();
    RealType distance  = static_cast<RealType>(0);

    HalfSpace3D() = default;
    HalfSpace3D(const Point3& dir, RealType dist) : direction(dir), distance(dist) {}

    [[nodiscard]] RealType evaluate(const Point3& pt) const 
    {
        return direction.dot(pt) - distance;
    }
};

/* =========== Utility Functions =========== */

inline Point3 makeInfinityVec() 
{
    constexpr RealType kInf = std::numeric_limits<RealType>::infinity();
    return Point3(kInf, kInf, kInf);
}

inline Point3 makeNegInfinityVec() 
{
    constexpr RealType kInf = std::numeric_limits<RealType>::infinity();
    return Point3(-kInf, -kInf, -kInf);
}

inline RealType ensurePositive(RealType val) 
{
    constexpr RealType kEps = std::numeric_limits<RealType>::epsilon();
    return (val < kEps) ? kEps : val;
}

inline Matrix33 safeInvert(const Matrix33& mat) 
{
    Eigen::LDLT<Matrix33> decomp(mat);
    if (decomp.info() == Eigen::Success) {
        return decomp.solve(Matrix33::Identity());
    }
    return mat.completeOrthogonalDecomposition().pseudoInverse();
}

inline IntType computeDofOffset(IntType entityIdx) 
{
    return entityIdx * 6;
}

inline bool hasPrefix(const TextType& str, const char* prefix) 
{
    return str.compare(0, std::char_traits<char>::length(prefix), prefix) == 0;
}

}  // namespace phys3d

/* =========== Backward Compatibility Layer =========== */
namespace rigid {
    using namespace phys3d;
    using Int = IntType;
    using Int64 = Int64Type;
    using UInt = UIntType;
    using Float = RealType;
    using Double = DoubleType;
    using String = TextType;
    template<typename T> using Vector = DynArray<T>;
    template<typename T> using UniquePtr = SolePtr<T>;
    template<typename T> using SharedPtr = JointPtr<T>;
    using Vec3 = Point3;
    using Vec4 = Point4;
    using Mat3 = Matrix33;
    using Mat4 = Matrix44;
    using VecX = VectorN;
    using MatX = MatrixMN;
    using Quat = Rotation4;
    using Triangle = Triplet3i;
    using SparseMat = SparseGrid;
    using Triplet = SparseEntry;
    using Triplets = SparseEntryList;
    using AABB = BoundingBox3D;
    using Plane = HalfSpace3D;
#if defined(RIGID_USE_CUDA)
    using Int3 = CudaInt3;
    using Float3 = CudaFloat3;
    using Float4 = CudaFloat4;
#endif
    inline Point3 positiveInfinity() { return makeInfinityVec(); }
    inline Point3 negativeInfinity() { return makeNegInfinityVec(); }
    inline RealType clampPositive(RealType v) { return ensurePositive(v); }
    inline Matrix33 invertSymmetric(const Matrix33& m) { return safeInvert(m); }
    inline IntType globalDofIndex(IntType idx) { return computeDofOffset(idx); }
    inline bool startsWith(const TextType& s, const char* p) { return hasPrefix(s, p); }
}

#endif // PHYS3D_FOUNDATION_TYPES_HPP
