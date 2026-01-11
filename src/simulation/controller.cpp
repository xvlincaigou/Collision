/*
 * Implementation: SimulationController class
 */
#include "controller.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace phys3d {

SimulationController::SimulationController() = default;

namespace {

struct GeometryIndexer 
{
    DynArray<size_t> pointCounts;
    DynArray<size_t> pointOffsets;
    DynArray<bool> validEntries;
};

GeometryIndexer buildGeometryIndexer(World& world) 
{
    const IntType entityCount = world.entityCount();
    GeometryIndexer indexer;
    indexer.pointCounts.assign(entityCount, 0);
    indexer.pointOffsets.assign(entityCount, 0);
    indexer.validEntries.assign(entityCount, false);

    size_t runningOffset = 1;
    for (IntType i = 0; i < entityCount; ++i) 
    {
        DynamicEntity* entity = world.entity(i);
        if (!entity || !entity->hasSurface()) 
        {
            indexer.pointOffsets[i] = runningOffset;
            continue;
        }

        const TriangleSurface& surface = entity->surface();
        indexer.validEntries[i] = true;
        indexer.pointCounts[i] = surface.pointCount();
        indexer.pointOffsets[i] = runningOffset;
        runningOffset += indexer.pointCounts[i];
    }

    return indexer;
}

TextType generateOBJBlock(World& world, IntType entityIdx, const GeometryIndexer& indexer) 
{
    if (!indexer.validEntries[entityIdx]) 
    {
        return {};
    }

    DynamicEntity* entity = world.entity(entityIdx);
    if (!entity || !entity->hasSurface()) 
    {
        return {};
    }

    const TriangleSurface& surface = entity->surface();
    const EntityState& state = entity->kinematic();
    const auto& pointCloud = surface.pointCloud();
    const auto& faceIndices = surface.faceIndices();
    const size_t offset = indexer.pointOffsets[entityIdx];

    std::ostringstream buffer;
    buffer.setf(std::ios::fmtflags(0), std::ios::floatfield);
    buffer.precision(9);

    buffer << "o " << surface.identifier() << "_" << entityIdx << "\n";

    for (const auto& localPt : pointCloud) 
    {
        Point3 worldPt = state.transformToWorld(localPt);
        buffer << "v " << worldPt.x() << " "
               << worldPt.y() << " "
               << worldPt.z() << "\n";
    }

    for (const auto& face : faceIndices) 
    {
        buffer << "f " << (face[0] + offset) << " "
               << (face[1] + offset) << " "
               << (face[2] + offset) << "\n";
    }

    return buffer.str();
}

void writeOBJOutput(std::ofstream& file, const DynArray<TextType>& blocks,
                    const DynArray<bool>& validFlags) 
{
    const IntType blockCount = static_cast<IntType>(blocks.size());
    for (IntType i = 0; i < blockCount; ++i) 
    {
        if (validFlags[i]) 
        {
            file << blocks[i];
        }
    }
}

bool serializeWorldToOBJ(World& world, const TextType& outputPath) 
{
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) 
    {
        std::cerr << "Failed to open file for writing: " << outputPath << std::endl;
        return false;
    }

    const IntType entityCount = world.entityCount();
    GeometryIndexer indexer = buildGeometryIndexer(world);
    DynArray<TextType> blocks(entityCount);

    tbb::parallel_for(tbb::blocked_range<IntType>(0, entityCount),
        [&](const tbb::blocked_range<IntType>& range) {
            for (IntType i = range.begin(); i != range.end(); ++i) 
            {
                blocks[i] = generateOBJBlock(world, i, indexer);
            }
        });

    writeOBJOutput(outFile, blocks, indexer.validEntries);

    return true;
}

}  // namespace

void SimulationController::initialize() 
{
    m_integrator.prepare(m_world);
    resetFrameCounter();
}

void SimulationController::reset() 
{
    m_world.destroyAllEntities();
    resetFrameCounter();
}

void SimulationController::tick() 
{
    m_integrator.advance(m_world);
    ++m_frameNumber;
}

DynamicEntity& SimulationController::createEntity(const TextType& identifier, 
                                                   const TextType& geometryPath,
                                                   RealType mass) 
{
    auto descriptor = EntityDescriptor::Standard(identifier, geometryPath, mass);
    return createEntity(descriptor);
}

DynamicEntity& SimulationController::createEntity(const TextType& identifier, 
                                                   const TextType& geometryPath,
                                                   RealType mass, RealType scale, 
                                                   RealType bounciness, RealType friction) 
{
    EntityDescriptor descriptor = EntityDescriptor::Standard(identifier, geometryPath, mass);
    descriptor.scale = scale;
    descriptor.bounciness = bounciness;
    descriptor.friction = friction;
    return createEntity(descriptor);
}

DynamicEntity& SimulationController::createEntity(const EntityDescriptor& descriptor) 
{
    DynamicEntity& entity = instantiateEntity(descriptor.identifier, descriptor.geometryPath);
    configureMaterial(entity, descriptor.mass, descriptor.bounciness, descriptor.friction);
    configureScale(entity, descriptor.scale);
    return entity;
}

DynamicEntity& SimulationController::instantiateEntity(const TextType& identifier, 
                                                        const TextType& geometryPath) 
{
    DynamicEntity& entity = m_world.spawnEntity(identifier);
    auto surfacePtr = SurfaceResourcePool::global().fetch(geometryPath, true);
    entity.assignSurface(surfacePtr);
    return entity;
}

void SimulationController::configureMaterial(DynamicEntity& entity, RealType mass,
                                              RealType bounciness, RealType friction) 
{
    MaterialProperties material;
    material.assignMass(mass);
    material.bounciness = bounciness;
    material.surfaceFriction = friction;
    material.prepare();
    entity.assignMaterial(material);
}

void SimulationController::configureScale(DynamicEntity& entity, RealType scale) 
{
    EntityState state = entity.kinematic();
    state.scaleFactor = scale;
    entity.assignKinematic(state);
}

void SimulationController::resetFrameCounter() 
{
    m_frameNumber = 0;
}

void SimulationController::setBoundaryLimits(const Point3& lowerBound, const Point3& upperBound) 
{
    m_world.configureBounds(lowerBound, upperBound);
}

void SimulationController::exportFrame(const TextType& outputPath) 
{
    serializeWorldToOBJ(m_world, outputPath);
}

}  // namespace phys3d
