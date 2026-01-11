/*
 * Implementation: CUDA Spatial Tree Builder
 */
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <cstdio>

#include "bvh_builder.cuh"

namespace phys3d {
namespace gpu {

/* ========== Kernel Implementations ========== */

__global__ void kernelComputeCentersAndBounds(
    const CudaFloat3* pointData,
    const CudaInt3* faceData,
    CudaFloat3* centerData,
    DeviceBounds3D* primBoundsData,
    CudaFloat3* globalMin,
    CudaFloat3* globalMax,
    int faceTotal)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= faceTotal) return;

    CudaInt3 face = faceData[tid];
    CudaFloat3 pt0 = pointData[face.x];
    CudaFloat3 pt1 = pointData[face.y];
    CudaFloat3 pt2 = pointData[face.z];

    centerData[tid] = make_float3(
        (pt0.x + pt1.x + pt2.x) / 3.0f,
        (pt0.y + pt1.y + pt2.y) / 3.0f,
        (pt0.z + pt1.z + pt2.z) / 3.0f);

    DeviceBounds3D bounds;
    bounds.lo = make_float3(
        fminf(fminf(pt0.x, pt1.x), pt2.x),
        fminf(fminf(pt0.y, pt1.y), pt2.y),
        fminf(fminf(pt0.z, pt1.z), pt2.z));
    bounds.hi = make_float3(
        fmaxf(fmaxf(pt0.x, pt1.x), pt2.x),
        fmaxf(fmaxf(pt0.y, pt1.y), pt2.y),
        fmaxf(fmaxf(pt0.z, pt1.z), pt2.z));
    primBoundsData[tid] = bounds;

    atomicMin(reinterpret_cast<int*>(&globalMin->x), __float_as_int(bounds.lo.x));
    atomicMin(reinterpret_cast<int*>(&globalMin->y), __float_as_int(bounds.lo.y));
    atomicMin(reinterpret_cast<int*>(&globalMin->z), __float_as_int(bounds.lo.z));
    atomicMax(reinterpret_cast<int*>(&globalMax->x), __float_as_int(bounds.hi.x));
    atomicMax(reinterpret_cast<int*>(&globalMax->y), __float_as_int(bounds.hi.y));
    atomicMax(reinterpret_cast<int*>(&globalMax->z), __float_as_int(bounds.hi.z));
}

__global__ void kernelEncodeMorton(
    const CudaFloat3* centerData,
    UIntType* mortonData,
    CudaFloat3 rangeMin,
    CudaFloat3 rangeSpan,
    int faceTotal)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= faceTotal) return;

    CudaFloat3 c = centerData[tid];

    float normX = (rangeSpan.x > 0) ? (c.x - rangeMin.x) / rangeSpan.x : 0.5f;
    float normY = (rangeSpan.y > 0) ? (c.y - rangeMin.y) / rangeSpan.y : 0.5f;
    float normZ = (rangeSpan.z > 0) ? (c.z - rangeMin.z) / rangeSpan.z : 0.5f;

    mortonData[tid] = encodeMorton(normX, normY, normZ);
}

__global__ void kernelBuildRadixTree(
    const UIntType* sortedMorton,
    const int* sortedIdx,
    DeviceTreeNode* nodeData,
    int* parentData,
    int* faceOrderData,
    int faceTotal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= faceTotal - 1) return;

    int prefixLeft = commonPrefix(sortedMorton, faceTotal, i, i - 1);
    int prefixRight = commonPrefix(sortedMorton, faceTotal, i, i + 1);
    int dir = (prefixRight > prefixLeft) ? 1 : -1;

    int minPrefix = commonPrefix(sortedMorton, faceTotal, i, i - dir);
    int maxLen = 2;
    while (commonPrefix(sortedMorton, faceTotal, i, i + maxLen * dir) > minPrefix)
        maxLen *= 2;

    int len = 0;
    for (int step = maxLen / 2; step >= 1; step /= 2)
        if (commonPrefix(sortedMorton, faceTotal, i, i + (len + step) * dir) > minPrefix)
            len = len + step;
    int j = i + len * dir;

    int nodePrefix = commonPrefix(sortedMorton, faceTotal, i, j);
    int splitPos = 0;
    int divisor = 2;
    int t = (len + divisor - 1) / divisor;
    while (t >= 1) 
    {
        if (commonPrefix(sortedMorton, faceTotal, i, i + (splitPos + t) * dir) > nodePrefix)
            splitPos = splitPos + t;
        divisor *= 2;
        t = (len + divisor - 1) / divisor;
    }
    int gamma = i + splitPos * dir + min(dir, 0);

    int leftChild, rightChild;
    int rangeMin = min(i, j);
    int rangeMax = max(i, j);

    if (rangeMin == gamma) 
    {
        leftChild = faceTotal - 1 + gamma;
        nodeData[leftChild].leafStart = gamma;
        nodeData[leftChild].leafCount = 1;
        nodeData[leftChild].leftChild = -1;
        nodeData[leftChild].rightChild = -1;
        faceOrderData[gamma] = sortedIdx[gamma];
        parentData[leftChild] = i;
    } 
    else 
    {
        leftChild = gamma;
        parentData[gamma] = i;
    }

    if (rangeMax == gamma + 1) 
    {
        rightChild = faceTotal - 1 + gamma + 1;
        nodeData[rightChild].leafStart = gamma + 1;
        nodeData[rightChild].leafCount = 1;
        nodeData[rightChild].leftChild = -1;
        nodeData[rightChild].rightChild = -1;
        faceOrderData[gamma + 1] = sortedIdx[gamma + 1];
        parentData[rightChild] = i;
    } 
    else 
    {
        rightChild = gamma + 1;
        parentData[gamma + 1] = i;
    }

    nodeData[i].leftChild = leftChild;
    nodeData[i].rightChild = rightChild;
    nodeData[i].leafStart = -1;
    nodeData[i].leafCount = 0;
}

__global__ void kernelPropagateBounds(
    DeviceTreeNode* nodeData,
    const int* parentData,
    int* visitFlags,
    const DeviceBounds3D* primBoundsData,
    const int* faceOrderData,
    int faceTotal)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= faceTotal) return;

    int leafIdx = faceTotal - 1 + tid;
    int primIdx = faceOrderData[tid];
    nodeData[leafIdx].volume = primBoundsData[primIdx];

    int current = parentData[leafIdx];
    while (current >= 0) 
    {
        int prevCount = atomicAdd(&visitFlags[current], 1);
        if (prevCount == 0) return;

        int lc = nodeData[current].leftChild;
        int rc = nodeData[current].rightChild;
        DeviceBounds3D leftVol = nodeData[lc].volume;
        DeviceBounds3D rightVol = nodeData[rc].volume;

        nodeData[current].volume.lo = make_float3(
            fminf(leftVol.lo.x, rightVol.lo.x),
            fminf(leftVol.lo.y, rightVol.lo.y),
            fminf(leftVol.lo.z, rightVol.lo.z));
        nodeData[current].volume.hi = make_float3(
            fmaxf(leftVol.hi.x, rightVol.hi.x),
            fmaxf(leftVol.hi.y, rightVol.hi.y),
            fmaxf(leftVol.hi.z, rightVol.hi.z));

        current = parentData[current];
    }
}

/* ========== DeviceTreeBuilder Implementation ========== */

DeviceTreeBuilder::DeviceTreeBuilder() = default;

DeviceTreeBuilder::~DeviceTreeBuilder() 
{
    release();
}

void DeviceTreeBuilder::release() 
{
    if (m_dPoints) cudaFree(m_dPoints);
    if (m_dFaces) cudaFree(m_dFaces);
    if (m_dCenters) cudaFree(m_dCenters);
    if (m_dPrimBounds) cudaFree(m_dPrimBounds);
    if (m_dMortonCodes) cudaFree(m_dMortonCodes);
    if (m_dSortedIdx) cudaFree(m_dSortedIdx);
    if (m_dFaceOrder) cudaFree(m_dFaceOrder);
    if (m_dNodes) cudaFree(m_dNodes);
    if (m_dParentIdx) cudaFree(m_dParentIdx);
    if (m_dAtomicFlags) cudaFree(m_dAtomicFlags);

    m_dPoints = nullptr;
    m_dFaces = nullptr;
    m_dCenters = nullptr;
    m_dPrimBounds = nullptr;
    m_dMortonCodes = nullptr;
    m_dSortedIdx = nullptr;
    m_dFaceOrder = nullptr;
    m_dNodes = nullptr;
    m_dParentIdx = nullptr;
    m_dAtomicFlags = nullptr;
}

void DeviceTreeBuilder::execute(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces) 
{
    release();

    m_pointCount = static_cast<IntType>(pointCloud.size());
    m_faceCount = static_cast<IntType>(faces.size());
    m_internalCount = m_faceCount - 1;
    m_totalNodes = m_faceCount + m_internalCount;

    if (m_faceCount == 0) return;

    cudaMalloc(&m_dPoints, m_pointCount * sizeof(CudaFloat3));
    cudaMalloc(&m_dFaces, m_faceCount * sizeof(CudaInt3));
    cudaMalloc(&m_dCenters, m_faceCount * sizeof(CudaFloat3));
    cudaMalloc(&m_dPrimBounds, m_faceCount * sizeof(DeviceBounds3D));
    cudaMalloc(&m_dMortonCodes, m_faceCount * sizeof(UIntType));
    cudaMalloc(&m_dSortedIdx, m_faceCount * sizeof(int));
    cudaMalloc(&m_dFaceOrder, m_faceCount * sizeof(int));
    cudaMalloc(&m_dNodes, m_totalNodes * sizeof(DeviceTreeNode));
    cudaMalloc(&m_dParentIdx, m_totalNodes * sizeof(int));
    cudaMalloc(&m_dAtomicFlags, m_internalCount * sizeof(int));

    std::vector<CudaFloat3> hostPoints(m_pointCount);
    for (int i = 0; i < m_pointCount; ++i) 
    {
        hostPoints[i] = make_float3(pointCloud[i].x(), pointCloud[i].y(), pointCloud[i].z());
    }
    cudaMemcpy(m_dPoints, hostPoints.data(), m_pointCount * sizeof(CudaFloat3), cudaMemcpyHostToDevice);

    std::vector<CudaInt3> hostFaces(m_faceCount);
    for (int i = 0; i < m_faceCount; ++i) 
    {
        hostFaces[i] = make_int3(faces[i].x(), faces[i].y(), faces[i].z());
    }
    cudaMemcpy(m_dFaces, hostFaces.data(), m_faceCount * sizeof(CudaInt3), cudaMemcpyHostToDevice);

    cudaMemset(m_dAtomicFlags, 0, m_internalCount * sizeof(int));
    cudaMemset(m_dParentIdx, -1, m_totalNodes * sizeof(int));

    computePrimitiveBounds();
    computeMortonCodes();
    sortByMorton();
    constructRadixTree();
    propagateBounds();

    cudaDeviceSynchronize();
}

void DeviceTreeBuilder::computePrimitiveBounds() 
{
    CudaFloat3* dGlobalMin;
    CudaFloat3* dGlobalMax;
    cudaMalloc(&dGlobalMin, sizeof(CudaFloat3));
    cudaMalloc(&dGlobalMax, sizeof(CudaFloat3));

    CudaFloat3 initMin = make_float3(1e30f, 1e30f, 1e30f);
    CudaFloat3 initMax = make_float3(-1e30f, -1e30f, -1e30f);
    cudaMemcpy(dGlobalMin, &initMin, sizeof(CudaFloat3), cudaMemcpyHostToDevice);
    cudaMemcpy(dGlobalMax, &initMax, sizeof(CudaFloat3), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (m_faceCount + threadsPerBlock - 1) / threadsPerBlock;

    kernelComputeCentersAndBounds<<<numBlocks, threadsPerBlock>>>(
        m_dPoints, m_dFaces, m_dCenters, m_dPrimBounds,
        dGlobalMin, dGlobalMax, m_faceCount);

    cudaFree(dGlobalMin);
    cudaFree(dGlobalMax);
}

void DeviceTreeBuilder::computeMortonCodes() 
{
    std::vector<CudaFloat3> hostCenters(m_faceCount);
    cudaMemcpy(hostCenters.data(), m_dCenters, m_faceCount * sizeof(CudaFloat3), cudaMemcpyDeviceToHost);

    CudaFloat3 rangeMin = make_float3(1e30f, 1e30f, 1e30f);
    CudaFloat3 rangeMax = make_float3(-1e30f, -1e30f, -1e30f);
    for (const auto& c : hostCenters) 
    {
        rangeMin.x = fminf(rangeMin.x, c.x);
        rangeMin.y = fminf(rangeMin.y, c.y);
        rangeMin.z = fminf(rangeMin.z, c.z);
        rangeMax.x = fmaxf(rangeMax.x, c.x);
        rangeMax.y = fmaxf(rangeMax.y, c.y);
        rangeMax.z = fmaxf(rangeMax.z, c.z);
    }
    CudaFloat3 rangeSpan = make_float3(
        rangeMax.x - rangeMin.x,
        rangeMax.y - rangeMin.y,
        rangeMax.z - rangeMin.z);

    int threadsPerBlock = 256;
    int numBlocks = (m_faceCount + threadsPerBlock - 1) / threadsPerBlock;

    kernelEncodeMorton<<<numBlocks, threadsPerBlock>>>(
        m_dCenters, m_dMortonCodes, rangeMin, rangeSpan, m_faceCount);
}

void DeviceTreeBuilder::sortByMorton() 
{
    thrust::device_ptr<UIntType> mortonPtr(m_dMortonCodes);
    thrust::device_ptr<int> idxPtr(m_dSortedIdx);
    thrust::sequence(idxPtr, idxPtr + m_faceCount);
    thrust::sort_by_key(mortonPtr, mortonPtr + m_faceCount, idxPtr);
}

void DeviceTreeBuilder::constructRadixTree() 
{
    int threadsPerBlock = 256;
    int numBlocks = (m_internalCount + threadsPerBlock - 1) / threadsPerBlock;

    kernelBuildRadixTree<<<numBlocks, threadsPerBlock>>>(
        m_dMortonCodes, m_dSortedIdx, m_dNodes, m_dParentIdx,
        m_dFaceOrder, m_faceCount);
}

void DeviceTreeBuilder::propagateBounds() 
{
    int threadsPerBlock = 256;
    int numBlocks = (m_faceCount + threadsPerBlock - 1) / threadsPerBlock;

    kernelPropagateBounds<<<numBlocks, threadsPerBlock>>>(
        m_dNodes, m_dParentIdx, m_dAtomicFlags,
        m_dPrimBounds, m_dFaceOrder, m_faceCount);
}

void DeviceTreeBuilder::transferToHost(DynArray<DeviceTreeNode>& outNodes, DynArray<IntType>& outFaceOrder) const 
{
    outNodes.resize(m_totalNodes);
    outFaceOrder.resize(m_faceCount);
    cudaMemcpy(outNodes.data(), m_dNodes, m_totalNodes * sizeof(DeviceTreeNode), cudaMemcpyDeviceToHost);
    cudaMemcpy(outFaceOrder.data(), m_dFaceOrder, m_faceCount * sizeof(IntType), cudaMemcpyDeviceToHost);
}

}  // namespace gpu
}  // namespace phys3d
