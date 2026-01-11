/*
 * Implementation: TriangleSurface class
 */
#include "surface.h"
#include "accel/spatial_index.h"

#include <fstream>
#include <sstream>

namespace phys3d {

TriangleSurface::TriangleSurface(TextType identifier) 
    : m_identifier(std::move(identifier)) 
{}

TriangleSurface::~TriangleSurface() = default;

TriangleSurface::TriangleSurface(TriangleSurface&&) noexcept = default;
TriangleSurface& TriangleSurface::operator=(TriangleSurface&&) noexcept = default;

bool TriangleSurface::parseWavefrontFile(const TextType& filepath, bool constructTree) 
{
    std::ifstream inputStream(filepath);
    if (!inputStream.is_open()) 
    {
        return false;
    }

    purge();

    TextType lineBuffer;
    while (std::getline(inputStream, lineBuffer)) 
    {
        std::istringstream tokenizer(lineBuffer);
        TextType prefix;
        
        if (!(tokenizer >> prefix)) 
        {
            continue;
        }

        if (prefix == "v") 
        {
            RealType px, py, pz;
            tokenizer >> px >> py >> pz;
            m_pointCloud.emplace_back(px, py, pz);
        } 
        else if (prefix == "vn") 
        {
            RealType nx, ny, nz;
            tokenizer >> nx >> ny >> nz;
            m_faceNormals.emplace_back(nx, ny, nz);
        } 
        else if (prefix == "f") 
        {
            DynArray<IntType> vertIdxList;
            TextType segment;

            while (tokenizer >> segment) 
            {
                std::istringstream segStream(segment);
                TextType indexPart;
                std::getline(segStream, indexPart, '/');

                if (!indexPart.empty()) 
                {
                    vertIdxList.push_back(std::stoi(indexPart) - 1);
                }
            }

            if (vertIdxList.size() >= 3) 
            {
                for (size_t k = 1; k + 1 < vertIdxList.size(); ++k) 
                {
                    m_faceIndices.emplace_back(vertIdxList[0], vertIdxList[k], vertIdxList[k + 1]);
                }
            }
        }
    }

    recomputeExtent();
    m_treeStale = true;

    if (constructTree) 
    {
        reconstructAccelerator();
    }

    return true;
}

void TriangleSurface::purge() 
{
    m_pointCloud.clear();
    m_faceIndices.clear();
    m_faceNormals.clear();
    m_localExtent.invalidate();
    m_extentStale = true;
    m_spatialTree.reset();
    m_treeStale = true;
}

bool TriangleSurface::hasAccelerator() const 
{
    return m_spatialTree && m_spatialTree->constructed();
}

const SpatialTree& TriangleSurface::accelerator() const 
{
    static const SpatialTree kEmptyTree;
    return m_spatialTree ? *m_spatialTree : kEmptyTree;
}

void TriangleSurface::reconstructAccelerator() 
{
    if (m_extentStale) 
    {
        recomputeExtent();
    }

    if (!m_spatialTree) 
    {
        m_spatialTree = std::make_unique<SpatialTree>();
    }

#if defined(RIGID_USE_CUDA)
    m_spatialTree->constructOnDevice(m_pointCloud, m_faceIndices);
#else
    m_spatialTree->construct(m_pointCloud, m_faceIndices);
#endif

    m_treeStale = false;
}

void TriangleSurface::recomputeExtent() 
{
    m_localExtent.invalidate();
    for (const auto& pt : m_pointCloud) 
    {
        m_localExtent.enclose(pt);
    }
    m_extentStale = false;
}

}  // namespace phys3d
