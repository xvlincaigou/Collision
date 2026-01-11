/*
 * Physics Simulation Entry Point
 */
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "simulation/simulator.h"

namespace {

struct ExecutionParams 
{
    std::string geometryFile = "assets/diamond.obj";
    std::string outputFolder = "output";
    int entityCount = 20;
    int frameLimit = 200;
    float spawnMargin = 3.0f;
    float minHeight = 2.0f;
    bool verboseOutput = true;
    bool writeFrames = true;
    bool performanceMode = false;

    phys3d::Point3 domainMin{-10.0f, -10.0f, 0.0f};
    phys3d::Point3 domainMax{10.0f, 10.0f, 10.0f};

    float massLo = 0.5f;
    float massHi = 3.0f;
    float scaleLo = 0.5f;
    float scaleHi = 2.0f;
    float bouncinessLo = 0.2f;
    float bouncinessHi = 0.9f;
    float frictionLo = 0.3f;
    float frictionHi = 0.8f;
    float velocityLo = -3.0f;
    float velocityHi = 3.0f;
};

void displayHelp(const char* executableName) 
{
    std::cout << "Usage: " << executableName << " [options]\n"
              << "  -m, --mesh <path>       Path to mesh file (default: assets/diamond.obj)\n"
              << "  -n, --num-objects <n>   Number of objects (default: 20)\n"
              << "  -f, --frames <n>        Number of frames (default: 200)\n"
              << "  -o, --output <dir>      Output directory (default: output)\n"
              << "  --mass <min> <max>      Mass range (default: 0.5 3.0)\n"
              << "  --scale <min> <max>     Scale/radius range (default: 0.5 2.0)\n"
              << "  --restitution <min> <max>  Bounciness range (default: 0.2 0.9)\n"
              << "  --friction <min> <max>  Friction range (default: 0.3 0.8)\n"
              << "  --velocity <min> <max>  Initial velocity range (default: -3.0 3.0)\n"
              << "  --no-export             Skip frame export (for benchmarking)\n"
              << "  --benchmark             Benchmark mode: minimal output, timing only\n"
              << "  -q, --quiet             Quiet mode: suppress per-frame output\n"
              << "  -h, --help              Show this help message\n"
              << std::endl;
}

bool processArguments(int argc, char* argv[], ExecutionParams& params) 
{
    for (int idx = 1; idx < argc; ++idx) 
    {
        std::string token = argv[idx];

        if (token == "-h" || token == "--help") 
        {
            displayHelp(argv[0]);
            return false;
        }
        else if ((token == "-m" || token == "--mesh") && idx + 1 < argc) 
        {
            params.geometryFile = argv[++idx];
        }
        else if ((token == "-n" || token == "--num-objects") && idx + 1 < argc) 
        {
            params.entityCount = std::stoi(argv[++idx]);
        }
        else if ((token == "-f" || token == "--frames") && idx + 1 < argc) 
        {
            params.frameLimit = std::stoi(argv[++idx]);
        }
        else if ((token == "-o" || token == "--output") && idx + 1 < argc) 
        {
            params.outputFolder = argv[++idx];
        }
        else if (token == "--mass" && idx + 2 < argc) 
        {
            params.massLo = std::stof(argv[++idx]);
            params.massHi = std::stof(argv[++idx]);
        }
        else if (token == "--scale" && idx + 2 < argc) 
        {
            params.scaleLo = std::stof(argv[++idx]);
            params.scaleHi = std::stof(argv[++idx]);
        }
        else if (token == "--restitution" && idx + 2 < argc) 
        {
            params.bouncinessLo = std::stof(argv[++idx]);
            params.bouncinessHi = std::stof(argv[++idx]);
        }
        else if (token == "--friction" && idx + 2 < argc) 
        {
            params.frictionLo = std::stof(argv[++idx]);
            params.frictionHi = std::stof(argv[++idx]);
        }
        else if (token == "--velocity" && idx + 2 < argc) 
        {
            params.velocityLo = std::stof(argv[++idx]);
            params.velocityHi = std::stof(argv[++idx]);
        }
        else if (token == "--no-export") 
        {
            params.writeFrames = false;
        }
        else if (token == "--benchmark") 
        {
            params.performanceMode = true;
            params.verboseOutput = false;
            params.writeFrames = false;
        }
        else if (token == "-q" || token == "--quiet") 
        {
            params.verboseOutput = false;
        }
        else 
        {
            std::cerr << "[ERROR] Unknown argument: " << token << std::endl;
            displayHelp(argv[0]);
            return false;
        }
    }
    return true;
}

class PseudoRandomSource 
{
public:
    static PseudoRandomSource& shared() 
    {
        static PseudoRandomSource instance;
        return instance;
    }

    void setSeed(unsigned int seed) 
    {
        m_engine.seed(seed);
    }

    float randomInRange(float lo, float hi) 
    {
        std::uniform_real_distribution<float> distribution(lo, hi);
        return distribution(m_engine);
    }

    phys3d::Rotation4 randomRotation() 
    {
        constexpr float kTwoPi = 6.28318530717958647692f;

        float r1 = randomInRange(0.0f, 1.0f);
        float r2 = randomInRange(0.0f, 1.0f);
        float r3 = randomInRange(0.0f, 1.0f);

        float sqrtOneMinusR1 = std::sqrt(1.0f - r1);
        float sqrtR1 = std::sqrt(r1);

        return phys3d::Rotation4(
                   sqrtOneMinusR1 * std::sin(kTwoPi * r2),
                   sqrtOneMinusR1 * std::cos(kTwoPi * r2),
                   sqrtR1 * std::sin(kTwoPi * r3),
                   sqrtR1 * std::cos(kTwoPi * r3))
            .normalized();
    }

private:
    PseudoRandomSource() : m_engine(42) {}
    std::mt19937 m_engine;
};

bool prepareOutputFolder(const std::string& folderPath) 
{
    namespace fs = std::filesystem;
    std::error_code errorCode;
    fs::create_directories(folderPath, errorCode);
    if (errorCode) 
    {
        std::cerr << "[ERROR] Could not create output directory: " << folderPath
                  << " (" << errorCode.message() << ")" << std::endl;
        return false;
    }
    return true;
}

std::string generateFramePath(const std::string& folder, int frameNum) 
{
    std::ostringstream pathBuilder;
    pathBuilder << folder << "/frame_" << std::setw(4) << std::setfill('0') << frameNum << ".obj";
    return pathBuilder.str();
}

using TimeClock = std::chrono::high_resolution_clock;
using TimeDelta = std::chrono::duration<double, std::milli>;

struct PerformanceMetrics 
{
    double setupDurationMs = 0.0;
    double simulationDurationMs = 0.0;
    double totalDurationMs = 0.0;
    double averageFrameMs = 0.0;
    int totalFrames = 0;
    int totalEntities = 0;
    std::string geometryFile;
};

void displayMetrics(const PerformanceMetrics& metrics) 
{
    std::cout << "\n========== TIMING RESULTS ==========\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Mesh:              " << metrics.geometryFile << "\n";
    std::cout << "Objects:           " << metrics.totalEntities << "\n";
    std::cout << "Frames:            " << metrics.totalFrames << "\n";
    std::cout << "------------------------------------\n";
    std::cout << "Setup time:        " << metrics.setupDurationMs << " ms\n";
    std::cout << "Simulation time:   " << metrics.simulationDurationMs << " ms\n";
    std::cout << "Total time:        " << metrics.totalDurationMs << " ms\n";
    std::cout << "Avg frame time:    " << metrics.averageFrameMs << " ms\n";
    std::cout << "FPS:               " << (1000.0 / metrics.averageFrameMs) << "\n";
    std::cout << "====================================\n";
}

void outputCSV(const PerformanceMetrics& metrics) 
{
    std::cout << metrics.geometryFile << ","
              << metrics.totalEntities << ","
              << metrics.totalFrames << ","
              << std::fixed << std::setprecision(3)
              << metrics.setupDurationMs << ","
              << metrics.simulationDurationMs << ","
              << metrics.totalDurationMs << ","
              << metrics.averageFrameMs << ","
              << (1000.0 / metrics.averageFrameMs) << std::endl;
}

bool configureSimulation(phys3d::SimulationController& controller, const ExecutionParams& params) 
{
    controller.initialize();
    controller.setBoundaryLimits(params.domainMin, params.domainMax);
    controller.setIterationLimit(20);
    controller.setConvergenceThreshold(1e-3f);

    auto& rng = PseudoRandomSource::shared();
    rng.setSeed(42);

    for (int entityIdx = 0; entityIdx < params.entityCount; ++entityIdx) 
    {
        float mass = rng.randomInRange(params.massLo, params.massHi);
        float scale = rng.randomInRange(params.scaleLo, params.scaleHi);
        float bounciness = rng.randomInRange(params.bouncinessLo, params.bouncinessHi);
        float friction = rng.randomInRange(params.frictionLo, params.frictionHi);

        std::string entityId = "entity_" + std::to_string(entityIdx);
        phys3d::DynamicEntity& entity = controller.createEntity(entityId, params.geometryFile,
                                                                 mass, scale, bounciness, friction);

        if (!entity.hasSurface()) 
        {
            std::cerr << "[ERROR] Failed to load mesh for entity " << entityIdx
                      << ". Check file path: " << params.geometryFile << std::endl;
            return false;
        }

        float effectiveMargin = params.spawnMargin * scale;

        float posX = rng.randomInRange(params.domainMin.x() + effectiveMargin,
                                        params.domainMax.x() - effectiveMargin);
        float posY = rng.randomInRange(params.domainMin.y() + effectiveMargin,
                                        params.domainMax.y() - effectiveMargin);
        float posZ = rng.randomInRange(params.minHeight + effectiveMargin,
                                        params.domainMax.z() - effectiveMargin);

        float velX = rng.randomInRange(params.velocityLo, params.velocityHi);
        float velY = rng.randomInRange(params.velocityLo, params.velocityHi);
        float velZ = rng.randomInRange(params.velocityLo, params.velocityHi);

        phys3d::EntityState state = entity.kinematic();
        state.translation = phys3d::Point3(posX, posY, posZ);
        state.orientation = rng.randomRotation();
        state.velocity = phys3d::Point3(velX, velY, velZ);

        entity.assignKinematic(state);

        if (params.verboseOutput && !params.performanceMode) 
        {
            std::cout << "[INFO] Entity " << entityIdx << ": scale=" << scale
                      << ", mass=" << mass << ", restitution=" << bounciness
                      << ", friction=" << friction << std::endl;
        }
    }

    return true;
}

double executeSimulation(phys3d::SimulationController& controller, const ExecutionParams& params) 
{
    if (params.verboseOutput) 
    {
        std::cout << "[INFO] Starting simulation for " << params.frameLimit
                  << " frames..." << std::endl;
    }

    auto startMoment = TimeClock::now();

    for (int frameIdx = 0; frameIdx < params.frameLimit; ++frameIdx) 
    {
        if (params.writeFrames) 
        {
            std::string framePath = generateFramePath(params.outputFolder, frameIdx);
            controller.exportFrame(framePath);
        }

        controller.tick();

        if (params.verboseOutput) 
        {
            std::cout << "[INFO] Frame " << (frameIdx + 1) << "/" << params.frameLimit
                      << " completed." << std::endl;
        }
    }

    auto endMoment = TimeClock::now();
    TimeDelta elapsed = endMoment - startMoment;

    if (params.verboseOutput) 
    {
        std::cout << "[INFO] Simulation finished." << std::endl;
        if (params.writeFrames) 
        {
            std::cout << "[INFO] Output saved to " << params.outputFolder << "/" << std::endl;
        }
    }

    return elapsed.count();
}

}  // namespace

int main(int argc, char* argv[]) 
{
    ExecutionParams params;

    if (!processArguments(argc, argv, params)) 
    {
        return 1;
    }

    if (params.writeFrames && !prepareOutputFolder(params.outputFolder)) 
    {
        return 1;
    }

    if (!params.performanceMode) 
    {
        std::cout << "[INFO] Initializing rigid body simulation..." << std::endl;
        std::cout << "[INFO] Mesh: " << params.geometryFile << std::endl;
        std::cout << "[INFO] Objects: " << params.entityCount << std::endl;
        std::cout << "[INFO] Frames: " << params.frameLimit << std::endl;
    }

    phys3d::SimulationController controller;

    auto setupStartMoment = TimeClock::now();
    if (!configureSimulation(controller, params)) 
    {
        return 1;
    }
    auto setupEndMoment = TimeClock::now();
    TimeDelta setupDuration = setupEndMoment - setupStartMoment;

    double simDurationMs = executeSimulation(controller, params);

    PerformanceMetrics metrics;
    metrics.geometryFile = params.geometryFile;
    metrics.totalEntities = params.entityCount;
    metrics.totalFrames = params.frameLimit;
    metrics.setupDurationMs = setupDuration.count();
    metrics.simulationDurationMs = simDurationMs;
    metrics.totalDurationMs = metrics.setupDurationMs + metrics.simulationDurationMs;
    metrics.averageFrameMs = simDurationMs / params.frameLimit;

    if (params.performanceMode) 
    {
        outputCSV(metrics);
    }
    else 
    {
        displayMetrics(metrics);
    }

    return 0;
}
