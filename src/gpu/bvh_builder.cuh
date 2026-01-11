/*
 * CUDA-Accelerated Spatial Tree Construction
 */
#ifndef PHYS3D_DEVICE_TREE_BUILDER_CUH
#define PHYS3D_DEVICE_TREE_BUILDER_CUH

#include "builtin_clz_msvc.h"

#include <cuda_runtime.h>
#include <vector>

#include "core/common.h"

#ifndef __CUDACC__
#include <limits>
inline int __clz(unsigned int x) { return (x == 0) ? 32 : __builtin_clz(x); }
#endif

namespace phys3d {
namespace gpu {

/* ========== Device Data Structures ========== */

struct DeviceBounds3D 
{
    CudaFloat3 lo;
    CudaFloat3 hi;
};

struct DeviceTreeNode 
{
    DeviceBounds3D volume;
    IntType leftChild;
    IntType rightChild;
    IntType leafStart;
    IntType leafCount;

    __host__ __device__ bool terminal() const { return leafCount > 0; }
};

/* ========== Morton Code Utilities ========== */

__device__ __host__ inline UIntType spreadBits(UIntType val) 
{
    val = (val * 0x00010001u) & 0xFF0000FFu;
    val = (val * 0x00000101u) & 0x0F00F00Fu;
    val = (val * 0x00000011u) & 0xC30C30C3u;
    val = (val * 0x00000005u) & 0x49249249u;
    return val;
}

__device__ __host__ inline UIntType encodeMorton(float px, float py, float pz) 
{
    px = fminf(fmaxf(px * 1024.0f, 0.0f), 1023.0f);
    py = fminf(fmaxf(py * 1024.0f, 0.0f), 1023.0f);
    pz = fminf(fmaxf(pz * 1024.0f, 0.0f), 1023.0f);
    UIntType mx = spreadBits(static_cast<UIntType>(px));
    UIntType my = spreadBits(static_cast<UIntType>(py));
    UIntType mz = spreadBits(static_cast<UIntType>(pz));
    return (mx << 2) | (my << 1) | mz;
}

__device__ inline int commonPrefix(const UIntType* codes, int total, int i, int j) 
{
    if (j < 0 || j >= total) return -1;
    UIntType codeI = codes[i];
    UIntType codeJ = codes[j];
    if (codeI == codeJ) return 32 + __clz(i ^ j);
    return __clz(codeI ^ codeJ);
}

/* ========== Device Tree Builder Class ========== */

class DeviceTreeBuilder 
{
public:
    DeviceTreeBuilder();
    ~DeviceTreeBuilder();

    void execute(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces);
    void transferToHost(DynArray<DeviceTreeNode>& outNodes, DynArray<IntType>& outFaceOrder) const;
    void release();

    [[nodiscard]] DeviceTreeNode* nodeBuffer() const { return m_dNodes; }
    [[nodiscard]] IntType* faceOrderBuffer() const { return m_dFaceOrder; }
    [[nodiscard]] CudaFloat3* pointBuffer() const { return m_dPoints; }
    [[nodiscard]] CudaInt3* faceBuffer() const { return m_dFaces; }

    [[nodiscard]] IntType totalNodes() const { return m_totalNodes; }
    [[nodiscard]] IntType totalFaces() const { return m_faceCount; }
    [[nodiscard]] IntType totalPoints() const { return m_pointCount; }
    [[nodiscard]] IntType rootNode() const { return 0; }

private:
    void computePrimitiveBounds();
    void computeMortonCodes();
    void sortByMorton();
    void constructRadixTree();
    void propagateBounds();

    CudaFloat3* m_dPoints         = nullptr;
    CudaInt3* m_dFaces            = nullptr;
    CudaFloat3* m_dCenters        = nullptr;
    DeviceBounds3D* m_dPrimBounds = nullptr;
    UIntType* m_dMortonCodes      = nullptr;
    IntType* m_dSortedIdx         = nullptr;
    IntType* m_dFaceOrder         = nullptr;
    DeviceTreeNode* m_dNodes      = nullptr;
    IntType* m_dParentIdx         = nullptr;
    IntType* m_dAtomicFlags       = nullptr;

    IntType m_pointCount      = 0;
    IntType m_faceCount       = 0;
    IntType m_internalCount   = 0;
    IntType m_totalNodes      = 0;
};

}  // namespace gpu
}  // namespace phys3d

namespace rigid {
namespace gpu {
    using BVHBuilderGPU = phys3d::gpu::DeviceTreeBuilder;
    using BVHNodeDevice = phys3d::gpu::DeviceTreeNode;
    using AABBDevice = phys3d::gpu::DeviceBounds3D;
}
}

#endif // PHYS3D_DEVICE_TREE_BUILDER_CUH
