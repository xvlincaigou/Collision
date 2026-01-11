/*
 * Implementation: CUDA Spatial Tree Builder
 * Restructured with fused kernels and modified memory access patterns
 */
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <cstdio>

#include "bvh_builder.cuh"

namespace phys3d {
namespace gpu {

/* ========== Device Helper Functions ========== */

// Fused min/max for float3
__device__ __forceinline__ void updateMinMax3(
    CudaFloat3& minVec, CudaFloat3& maxVec,
    float x, float y, float z)
{
    minVec.x = fminf(minVec.x, x);
    minVec.y = fminf(minVec.y, y);
    minVec.z = fminf(minVec.z, z);
    maxVec.x = fmaxf(maxVec.x, x);
    maxVec.y = fmaxf(maxVec.y, y);
    maxVec.z = fmaxf(maxVec.z, z);
}

// Merge two bounding boxes
__device__ __forceinline__ DeviceBounds3D mergeBounds(
    const DeviceBounds3D& a, const DeviceBounds3D& b)
{
    DeviceBounds3D result;
    result.lo.x = fminf(a.lo.x, b.lo.x);
    result.lo.y = fminf(a.lo.y, b.lo.y);
    result.lo.z = fminf(a.lo.z, b.lo.z);
    result.hi.x = fmaxf(a.hi.x, b.hi.x);
    result.hi.y = fmaxf(a.hi.y, b.hi.y);
    result.hi.z = fmaxf(a.hi.z, b.hi.z);
    return result;
}

/* ========== Phase 1: Compute Primitive Data (Fused Kernel) ========== */

__global__ void kernelComputePrimitiveData(
    const CudaFloat3* __restrict__ pointData,
    const CudaInt3* __restrict__ faceData,
    CudaFloat3* __restrict__ centerData,
    DeviceBounds3D* __restrict__ primBoundsData,
    CudaFloat3* __restrict__ globalMin,
    CudaFloat3* __restrict__ globalMax,
    int faceTotal)
{
    // Shared memory for block-level reduction
    __shared__ CudaFloat3 sharedMin[32];
    __shared__ CudaFloat3 sharedMax[32];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int laneId = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    
    // Initialize local accumulators
    CudaFloat3 localMin = make_float3(1e30f, 1e30f, 1e30f);
    CudaFloat3 localMax = make_float3(-1e30f, -1e30f, -1e30f);
    
    if (tid < faceTotal)
    {
        // Load face indices
        const CudaInt3 face = faceData[tid];
        
        // Load vertex positions
        const CudaFloat3 v0 = pointData[face.x];
        const CudaFloat3 v1 = pointData[face.y];
        const CudaFloat3 v2 = pointData[face.z];
        
        // Compute centroid with explicit operations
        const float cx = (v0.x + v1.x + v2.x) * 0.333333333f;
        const float cy = (v0.y + v1.y + v2.y) * 0.333333333f;
        const float cz = (v0.z + v1.z + v2.z) * 0.333333333f;
        centerData[tid] = make_float3(cx, cy, cz);
        
        // Compute AABB using chained comparisons
        DeviceBounds3D bounds;
        bounds.lo.x = fminf(v0.x, fminf(v1.x, v2.x));
        bounds.lo.y = fminf(v0.y, fminf(v1.y, v2.y));
        bounds.lo.z = fminf(v0.z, fminf(v1.z, v2.z));
        bounds.hi.x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
        bounds.hi.y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
        bounds.hi.z = fmaxf(v0.z, fmaxf(v1.z, v2.z));
        primBoundsData[tid] = bounds;
        
        localMin = bounds.lo;
        localMax = bounds.hi;
    }
    
    // Warp-level reduction for min/max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        localMin.x = fminf(localMin.x, __shfl_down_sync(0xffffffff, localMin.x, offset));
        localMin.y = fminf(localMin.y, __shfl_down_sync(0xffffffff, localMin.y, offset));
        localMin.z = fminf(localMin.z, __shfl_down_sync(0xffffffff, localMin.z, offset));
        localMax.x = fmaxf(localMax.x, __shfl_down_sync(0xffffffff, localMax.x, offset));
        localMax.y = fmaxf(localMax.y, __shfl_down_sync(0xffffffff, localMax.y, offset));
        localMax.z = fmaxf(localMax.z, __shfl_down_sync(0xffffffff, localMax.z, offset));
    }
    
    // First lane of each warp writes to shared memory
    if (laneId == 0)
    {
        sharedMin[warpId] = localMin;
        sharedMax[warpId] = localMax;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (threadIdx.x < 32)
    {
        const int numWarps = (blockDim.x + 31) >> 5;
        localMin = (threadIdx.x < numWarps) ? sharedMin[threadIdx.x] : make_float3(1e30f, 1e30f, 1e30f);
        localMax = (threadIdx.x < numWarps) ? sharedMax[threadIdx.x] : make_float3(-1e30f, -1e30f, -1e30f);
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            localMin.x = fminf(localMin.x, __shfl_down_sync(0xffffffff, localMin.x, offset));
            localMin.y = fminf(localMin.y, __shfl_down_sync(0xffffffff, localMin.y, offset));
            localMin.z = fminf(localMin.z, __shfl_down_sync(0xffffffff, localMin.z, offset));
            localMax.x = fmaxf(localMax.x, __shfl_down_sync(0xffffffff, localMax.x, offset));
            localMax.y = fmaxf(localMax.y, __shfl_down_sync(0xffffffff, localMax.y, offset));
            localMax.z = fmaxf(localMax.z, __shfl_down_sync(0xffffffff, localMax.z, offset));
        }
        
        // Atomic update to global bounds
        if (threadIdx.x == 0)
        {
            atomicMin(reinterpret_cast<int*>(&globalMin->x), __float_as_int(localMin.x));
            atomicMin(reinterpret_cast<int*>(&globalMin->y), __float_as_int(localMin.y));
            atomicMin(reinterpret_cast<int*>(&globalMin->z), __float_as_int(localMin.z));
            atomicMax(reinterpret_cast<int*>(&globalMax->x), __float_as_int(localMax.x));
            atomicMax(reinterpret_cast<int*>(&globalMax->y), __float_as_int(localMax.y));
            atomicMax(reinterpret_cast<int*>(&globalMax->z), __float_as_int(localMax.z));
        }
    }
}

/* ========== Phase 2: Morton Code Encoding ========== */

__global__ void kernelEncodeMortonCodes(
    const CudaFloat3* __restrict__ centerData,
    UIntType* __restrict__ mortonData,
    CudaFloat3 rangeMin,
    CudaFloat3 rangeInvSpan,
    int faceTotal)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= faceTotal) return;

    const CudaFloat3 c = centerData[tid];
    
    // Normalize to [0,1] using precomputed inverse span
    const float normX = (c.x - rangeMin.x) * rangeInvSpan.x;
    const float normY = (c.y - rangeMin.y) * rangeInvSpan.y;
    const float normZ = (c.z - rangeMin.z) * rangeInvSpan.z;

    mortonData[tid] = encodeMorton(normX, normY, normZ);
}

/* ========== Phase 3: Radix Tree Construction ========== */

__global__ void kernelConstructRadixTree(
    const UIntType* __restrict__ sortedMorton,
    const int* __restrict__ sortedIdx,
    DeviceTreeNode* __restrict__ nodeData,
    int* __restrict__ parentData,
    int* __restrict__ faceOrderData,
    int faceTotal)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= faceTotal - 1) return;

    // Determine split direction using prefix comparison
    const int prefixWithPrev = commonPrefix(sortedMorton, faceTotal, i, i - 1);
    const int prefixWithNext = commonPrefix(sortedMorton, faceTotal, i, i + 1);
    const int direction = (prefixWithNext > prefixWithPrev) ? 1 : -1;
    
    // Find range using exponential search
    const int minPrefix = commonPrefix(sortedMorton, faceTotal, i, i - direction);
    
    int rangeBound = 2;
    while (commonPrefix(sortedMorton, faceTotal, i, i + rangeBound * direction) > minPrefix)
    {
        rangeBound <<= 1;
    }
    
    // Binary search for exact range end
    int rangeLen = 0;
    int step = rangeBound >> 1;
    do {
        const int testPos = i + (rangeLen + step) * direction;
        if (commonPrefix(sortedMorton, faceTotal, i, testPos) > minPrefix)
            rangeLen += step;
        step >>= 1;
    } while (step >= 1);
    
    const int j = i + rangeLen * direction;
    
    // Find split position using binary search
    const int nodePrefix = commonPrefix(sortedMorton, faceTotal, i, j);
    int split = 0;
    int divisor = 2;
    int testLen = (rangeLen + divisor - 1) / divisor;
    
    do {
        const int testPos = i + (split + testLen) * direction;
        if (commonPrefix(sortedMorton, faceTotal, i, testPos) > nodePrefix)
            split += testLen;
        divisor <<= 1;
        testLen = (rangeLen + divisor - 1) / divisor;
    } while (testLen >= 1);
    
    const int gamma = i + split * direction + min(direction, 0);
    
    // Compute range bounds
    const int rangeMin = min(i, j);
    const int rangeMax = max(i, j);
    
    // Set up left child
    int leftChild;
    if (rangeMin == gamma)
    {
        leftChild = faceTotal - 1 + gamma;
        DeviceTreeNode& leafNode = nodeData[leftChild];
        leafNode.leafStart = gamma;
        leafNode.leafCount = 1;
        leafNode.leftChild = -1;
        leafNode.rightChild = -1;
        faceOrderData[gamma] = sortedIdx[gamma];
        parentData[leftChild] = i;
    }
    else
    {
        leftChild = gamma;
        parentData[gamma] = i;
    }
    
    // Set up right child
    int rightChild;
    if (rangeMax == gamma + 1)
    {
        rightChild = faceTotal - 1 + gamma + 1;
        DeviceTreeNode& leafNode = nodeData[rightChild];
        leafNode.leafStart = gamma + 1;
        leafNode.leafCount = 1;
        leafNode.leftChild = -1;
        leafNode.rightChild = -1;
        faceOrderData[gamma + 1] = sortedIdx[gamma + 1];
        parentData[rightChild] = i;
    }
    else
    {
        rightChild = gamma + 1;
        parentData[gamma + 1] = i;
    }
    
    // Set up internal node
    DeviceTreeNode& internalNode = nodeData[i];
    internalNode.leftChild = leftChild;
    internalNode.rightChild = rightChild;
    internalNode.leafStart = -1;
    internalNode.leafCount = 0;
}

/* ========== Phase 4: Bottom-Up Bound Propagation ========== */

__global__ void kernelPropagateBoundsBottomUp(
    DeviceTreeNode* __restrict__ nodeData,
    const int* __restrict__ parentData,
    int* __restrict__ visitFlags,
    const DeviceBounds3D* __restrict__ primBoundsData,
    const int* __restrict__ faceOrderData,
    int faceTotal)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= faceTotal) return;

    // Initialize leaf node bounds
    const int leafIdx = faceTotal - 1 + tid;
    const int primIdx = faceOrderData[tid];
    nodeData[leafIdx].volume = primBoundsData[primIdx];

    // Traverse up the tree
    int currentNode = parentData[leafIdx];
    
    while (currentNode >= 0)
    {
        // Atomic increment to track arrivals
        const int arrivalCount = atomicAdd(&visitFlags[currentNode], 1);
        
        // First thread to arrive exits, second computes bounds
        if (arrivalCount == 0) 
            return;
        
        // Load child bounds
        const int leftChild = nodeData[currentNode].leftChild;
        const int rightChild = nodeData[currentNode].rightChild;
        const DeviceBounds3D leftBounds = nodeData[leftChild].volume;
        const DeviceBounds3D rightBounds = nodeData[rightChild].volume;
        
        // Merge and store
        nodeData[currentNode].volume = mergeBounds(leftBounds, rightBounds);
        
        // Move to parent
        currentNode = parentData[currentNode];
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
    // Use local pointers to avoid multiple member accesses
    void* ptrs[] = {
        m_dPoints, m_dFaces, m_dCenters, m_dPrimBounds,
        m_dMortonCodes, m_dSortedIdx, m_dFaceOrder,
        m_dNodes, m_dParentIdx, m_dAtomicFlags
    };
    
    int idx = 0;
    while (idx < 10)
    {
        if (ptrs[idx] != nullptr)
            cudaFree(ptrs[idx]);
        ++idx;
    }

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

    // Allocate device memory
    allocateDeviceMemory();
    
    // Upload input data
    uploadInputData(pointCloud, faces);
    
    // Execute build phases
    computePrimitiveBounds();
    computeMortonCodes();
    sortByMorton();
    constructRadixTree();
    propagateBounds();

    cudaDeviceSynchronize();
}

void DeviceTreeBuilder::allocateDeviceMemory()
{
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
    
    cudaMemset(m_dAtomicFlags, 0, m_internalCount * sizeof(int));
    cudaMemset(m_dParentIdx, -1, m_totalNodes * sizeof(int));
}

void DeviceTreeBuilder::uploadInputData(const DynArray<Point3>& pointCloud, const DynArray<Triplet3i>& faces)
{
    // Convert and upload points
    std::vector<CudaFloat3> hostPoints(m_pointCount);
    int i = m_pointCount;
    while (i > 0)
    {
        --i;
        const Point3& pt = pointCloud[i];
        hostPoints[i] = make_float3(pt.x(), pt.y(), pt.z());
    }
    cudaMemcpy(m_dPoints, hostPoints.data(), m_pointCount * sizeof(CudaFloat3), cudaMemcpyHostToDevice);

    // Convert and upload faces
    std::vector<CudaInt3> hostFaces(m_faceCount);
    i = m_faceCount;
    while (i > 0)
    {
        --i;
        const Triplet3i& face = faces[i];
        hostFaces[i] = make_int3(face.x(), face.y(), face.z());
    }
    cudaMemcpy(m_dFaces, hostFaces.data(), m_faceCount * sizeof(CudaInt3), cudaMemcpyHostToDevice);
}

void DeviceTreeBuilder::computePrimitiveBounds() 
{
    CudaFloat3* dGlobalMin;
    CudaFloat3* dGlobalMax;
    cudaMalloc(&dGlobalMin, sizeof(CudaFloat3));
    cudaMalloc(&dGlobalMax, sizeof(CudaFloat3));

    const CudaFloat3 initMin = make_float3(1e30f, 1e30f, 1e30f);
    const CudaFloat3 initMax = make_float3(-1e30f, -1e30f, -1e30f);
    cudaMemcpy(dGlobalMin, &initMin, sizeof(CudaFloat3), cudaMemcpyHostToDevice);
    cudaMemcpy(dGlobalMax, &initMax, sizeof(CudaFloat3), cudaMemcpyHostToDevice);

    constexpr int kBlockSize = 256;
    const int numBlocks = (m_faceCount + kBlockSize - 1) / kBlockSize;

    kernelComputePrimitiveData<<<numBlocks, kBlockSize>>>(
        m_dPoints, m_dFaces, m_dCenters, m_dPrimBounds,
        dGlobalMin, dGlobalMax, m_faceCount);

    cudaFree(dGlobalMin);
    cudaFree(dGlobalMax);
}

void DeviceTreeBuilder::computeMortonCodes() 
{
    // Download centers for range computation
    std::vector<CudaFloat3> hostCenters(m_faceCount);
    cudaMemcpy(hostCenters.data(), m_dCenters, m_faceCount * sizeof(CudaFloat3), cudaMemcpyDeviceToHost);

    // Compute range on host
    CudaFloat3 rangeMin = make_float3(1e30f, 1e30f, 1e30f);
    CudaFloat3 rangeMax = make_float3(-1e30f, -1e30f, -1e30f);
    
    int i = static_cast<int>(hostCenters.size());
    while (i > 0)
    {
        --i;
        const CudaFloat3& c = hostCenters[i];
        rangeMin.x = fminf(rangeMin.x, c.x);
        rangeMin.y = fminf(rangeMin.y, c.y);
        rangeMin.z = fminf(rangeMin.z, c.z);
        rangeMax.x = fmaxf(rangeMax.x, c.x);
        rangeMax.y = fmaxf(rangeMax.y, c.y);
        rangeMax.z = fmaxf(rangeMax.z, c.z);
    }
    
    // Compute inverse span for efficient normalization
    const CudaFloat3 span = make_float3(
        rangeMax.x - rangeMin.x,
        rangeMax.y - rangeMin.y,
        rangeMax.z - rangeMin.z);
    
    const CudaFloat3 invSpan = make_float3(
        (span.x > 0) ? 1.0f / span.x : 0.5f,
        (span.y > 0) ? 1.0f / span.y : 0.5f,
        (span.z > 0) ? 1.0f / span.z : 0.5f);

    constexpr int kBlockSize = 256;
    const int numBlocks = (m_faceCount + kBlockSize - 1) / kBlockSize;

    kernelEncodeMortonCodes<<<numBlocks, kBlockSize>>>(
        m_dCenters, m_dMortonCodes, rangeMin, invSpan, m_faceCount);
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
    constexpr int kBlockSize = 256;
    const int numBlocks = (m_internalCount + kBlockSize - 1) / kBlockSize;

    kernelConstructRadixTree<<<numBlocks, kBlockSize>>>(
        m_dMortonCodes, m_dSortedIdx, m_dNodes, m_dParentIdx,
        m_dFaceOrder, m_faceCount);
}

void DeviceTreeBuilder::propagateBounds() 
{
    constexpr int kBlockSize = 256;
    const int numBlocks = (m_faceCount + kBlockSize - 1) / kBlockSize;

    kernelPropagateBoundsBottomUp<<<numBlocks, kBlockSize>>>(
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
