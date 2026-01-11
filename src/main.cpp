/*
 * Physics Simulation Entry Point
 * Restructured with state machine pattern and modular initialization
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
#include <functional>
#include <array>

#include "simulation/simulator.h"

namespace {

/* ========== Configuration Data Types ========== */

struct PropertyRanges
{
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

struct DomainSettings
{
    phys3d::Point3 boundsMin{-10.0f, -10.0f, 0.0f};
    phys3d::Point3 boundsMax{10.0f, 10.0f, 10.0f};
    float spawnMargin = 3.0f;
    float minSpawnHeight = 2.0f;
};

struct ExecutionSettings
{
    std::string geometryFile = "assets/diamond.obj";
    std::string outputFolder = "output";
    int entityCount = 20;
    int frameLimit = 200;
    bool verboseOutput = true;
    bool writeFrames = true;
    bool performanceMode = false;
    
    PropertyRanges properties;
    DomainSettings domain;
};

struct TimingData
{
    double setupMs = 0.0;
    double simulationMs = 0.0;
    
    double totalMs() const { return setupMs + simulationMs; }
    double avgFrameMs(int frames) const { return simulationMs / frames; }
    double fps(int frames) const { return 1000.0 / avgFrameMs(frames); }
};

/* ========== Random Number Generation ========== */

class UniformGenerator
{
public:
    static UniformGenerator& instance()
    {
        static UniformGenerator gen;
        return gen;
    }

    void resetSeed(unsigned int seed) { m_rng.seed(seed); }

    float uniform(float lo, float hi)
    {
        return std::uniform_real_distribution<float>(lo, hi)(m_rng);
    }

    phys3d::Rotation4 uniformRotation()
    {
        constexpr float kPi2 = 6.28318530717958647692f;
        
        const float u1 = uniform(0.0f, 1.0f);
        const float u2 = uniform(0.0f, 1.0f);
        const float u3 = uniform(0.0f, 1.0f);
        
        const float s1 = std::sqrt(1.0f - u1);
        const float s2 = std::sqrt(u1);
        
        return phys3d::Rotation4(
            s1 * std::sin(kPi2 * u2),
            s1 * std::cos(kPi2 * u2),
            s2 * std::sin(kPi2 * u3),
            s2 * std::cos(kPi2 * u3)).normalized();
    }

private:
    UniformGenerator() : m_rng(42) {}
    std::mt19937 m_rng;
};

/* ========== Command Line Parsing ========== */

class ArgumentProcessor
{
public:
    explicit ArgumentProcessor(int argc, char* argv[])
        : m_argc(argc), m_argv(argv), m_idx(1), m_valid(true) {}

    bool process(ExecutionSettings& settings)
    {
        while (m_idx < m_argc && m_valid)
        {
            const std::string token = m_argv[m_idx];
            
            if (token == "-h" || token == "--help")
            {
                printUsage();
                return false;
            }
            
            m_valid = dispatchArgument(token, settings);
            ++m_idx;
        }
        
        return m_valid;
    }

private:
    void printUsage()
    {
        std::cout << "Usage: " << m_argv[0] << " [options]\n"
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

    bool dispatchArgument(const std::string& token, ExecutionSettings& settings)
    {
        // Map of argument handlers
        using Handler = std::function<bool(ExecutionSettings&)>;
        
        if (token == "-m" || token == "--mesh")
            return parseString(settings.geometryFile);
        if (token == "-n" || token == "--num-objects")
            return parseInt(settings.entityCount);
        if (token == "-f" || token == "--frames")
            return parseInt(settings.frameLimit);
        if (token == "-o" || token == "--output")
            return parseString(settings.outputFolder);
        if (token == "--mass")
            return parseFloatPair(settings.properties.massLo, settings.properties.massHi);
        if (token == "--scale")
            return parseFloatPair(settings.properties.scaleLo, settings.properties.scaleHi);
        if (token == "--restitution")
            return parseFloatPair(settings.properties.bouncinessLo, settings.properties.bouncinessHi);
        if (token == "--friction")
            return parseFloatPair(settings.properties.frictionLo, settings.properties.frictionHi);
        if (token == "--velocity")
            return parseFloatPair(settings.properties.velocityLo, settings.properties.velocityHi);
        if (token == "--no-export")
        {
            settings.writeFrames = false;
            return true;
        }
        if (token == "--benchmark")
        {
            settings.performanceMode = true;
            settings.verboseOutput = false;
            settings.writeFrames = false;
            return true;
        }
        if (token == "-q" || token == "--quiet")
        {
            settings.verboseOutput = false;
            return true;
        }
        
        std::cerr << "[ERROR] Unknown argument: " << token << std::endl;
        return false;
    }

    bool parseString(std::string& target)
    {
        if (m_idx + 1 >= m_argc) return false;
        target = m_argv[++m_idx];
        return true;
    }

    bool parseInt(int& target)
    {
        if (m_idx + 1 >= m_argc) return false;
        target = std::stoi(m_argv[++m_idx]);
        return true;
    }

    bool parseFloatPair(float& lo, float& hi)
    {
        if (m_idx + 2 >= m_argc) return false;
        lo = std::stof(m_argv[++m_idx]);
        hi = std::stof(m_argv[++m_idx]);
        return true;
    }

    int m_argc;
    char** m_argv;
    int m_idx;
    bool m_valid;
};

bool ensureDirectoryExists(const std::string& path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path, ec);
    
    if (ec)
    {
        std::cerr << "[ERROR] Could not create directory: " << path
                  << " (" << ec.message() << ")" << std::endl;
        return false;
    }
    return true;
}

std::string buildFramePath(const std::string& folder, int frameNum)
{
    std::ostringstream oss;
    oss << folder << "/frame_" << std::setw(4) << std::setfill('0') << frameNum << ".obj";
    return oss.str();
}

/* ========== Output Formatting ========== */

class ResultReporter
{
public:
    static void displayTimings(const ExecutionSettings& settings, const TimingData& timing)
    {
        std::cout << "\n========== TIMING RESULTS ==========\n"
                  << std::fixed << std::setprecision(3)
                  << "Mesh:              " << settings.geometryFile << "\n"
                  << "Objects:           " << settings.entityCount << "\n"
                  << "Frames:            " << settings.frameLimit << "\n"
                  << "------------------------------------\n"
                  << "Setup time:        " << timing.setupMs << " ms\n"
                  << "Simulation time:   " << timing.simulationMs << " ms\n"
                  << "Total time:        " << timing.totalMs() << " ms\n"
                  << "Avg frame time:    " << timing.avgFrameMs(settings.frameLimit) << " ms\n"
                  << "FPS:               " << timing.fps(settings.frameLimit) << "\n"
                  << "====================================\n";
    }

    static void outputCSVLine(const ExecutionSettings& settings, const TimingData& timing)
    {
        std::cout << settings.geometryFile << ","
                  << settings.entityCount << ","
                  << settings.frameLimit << ","
                  << std::fixed << std::setprecision(3)
                  << timing.setupMs << ","
                  << timing.simulationMs << ","
                  << timing.totalMs() << ","
                  << timing.avgFrameMs(settings.frameLimit) << ","
                  << timing.fps(settings.frameLimit) << std::endl;
    }
};

/* ========== Entity Spawner ========== */

class EntitySpawner
{
public:
    explicit EntitySpawner(const ExecutionSettings& settings)
        : m_settings(settings) {}

    bool spawnEntity(phys3d::SimulationController& controller, int entityIdx)
    {
        auto& rng = UniformGenerator::instance();
        const auto& props = m_settings.properties;
        const auto& domain = m_settings.domain;

        // Generate random properties
        const float mass = rng.uniform(props.massLo, props.massHi);
        const float scale = rng.uniform(props.scaleLo, props.scaleHi);
        const float bounciness = rng.uniform(props.bouncinessLo, props.bouncinessHi);
        const float friction = rng.uniform(props.frictionLo, props.frictionHi);

        // Create entity
        const std::string entityId = "entity_" + std::to_string(entityIdx);
        phys3d::DynamicEntity& entity = controller.createEntity(
            entityId, m_settings.geometryFile, mass, scale, bounciness, friction);

        if (!entity.hasSurface())
        {
            std::cerr << "[ERROR] Failed to load mesh for entity " << entityIdx
                      << ". Check file path: " << m_settings.geometryFile << std::endl;
            return false;
        }

        // Generate spawn position
        const float margin = domain.spawnMargin * scale;
        const float posX = rng.uniform(domain.boundsMin.x() + margin, domain.boundsMax.x() - margin);
        const float posY = rng.uniform(domain.boundsMin.y() + margin, domain.boundsMax.y() - margin);
        const float posZ = rng.uniform(domain.minSpawnHeight + margin, domain.boundsMax.z() - margin);

        // Generate initial velocity
        const float velX = rng.uniform(props.velocityLo, props.velocityHi);
        const float velY = rng.uniform(props.velocityLo, props.velocityHi);
        const float velZ = rng.uniform(props.velocityLo, props.velocityHi);

        // Configure kinematic state
        phys3d::EntityState state = entity.kinematic();
        state.translation = phys3d::Point3(posX, posY, posZ);
        state.orientation = rng.uniformRotation();
        state.velocity = phys3d::Point3(velX, velY, velZ);
        entity.assignKinematic(state);

        // Log if verbose
        if (m_settings.verboseOutput && !m_settings.performanceMode)
        {
            std::cout << "[INFO] Entity " << entityIdx << ": scale=" << scale
                      << ", mass=" << mass << ", restitution=" << bounciness
                      << ", friction=" << friction << std::endl;
        }

        return true;
    }

private:
    const ExecutionSettings& m_settings;
};

/* ========== Simulation Runner ========== */

class SimulationRunner
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;

public:
    explicit SimulationRunner(const ExecutionSettings& settings)
        : m_settings(settings) {}

    bool initialize(phys3d::SimulationController& controller)
    {
        controller.initialize();
        controller.setBoundaryLimits(m_settings.domain.boundsMin, m_settings.domain.boundsMax);
        controller.setIterationLimit(20);
        controller.setConvergenceThreshold(1e-3f);

        UniformGenerator::instance().resetSeed(42);

        EntitySpawner spawner(m_settings);
        
        int idx = 0;
        while (idx < m_settings.entityCount)
        {
            if (!spawner.spawnEntity(controller, idx))
                return false;
            ++idx;
        }

        return true;
    }

    double execute(phys3d::SimulationController& controller)
    {
        if (m_settings.verboseOutput)
        {
            std::cout << "[INFO] Starting simulation for " << m_settings.frameLimit
                      << " frames..." << std::endl;
        }

        const auto startTime = Clock::now();

        int frameIdx = 0;
        while (frameIdx < m_settings.frameLimit)
        {
            if (m_settings.writeFrames)
            {
                const std::string framePath = buildFramePath(m_settings.outputFolder, frameIdx);
                controller.exportFrame(framePath);
            }

            controller.tick();

            if (m_settings.verboseOutput)
            {
                std::cout << "[INFO] Frame " << (frameIdx + 1) << "/" << m_settings.frameLimit
                          << " completed." << std::endl;
            }

            ++frameIdx;
        }

        const auto endTime = Clock::now();
        const Duration elapsed = endTime - startTime;

        if (m_settings.verboseOutput)
        {
            std::cout << "[INFO] Simulation finished." << std::endl;
            if (m_settings.writeFrames)
                std::cout << "[INFO] Output saved to " << m_settings.outputFolder << "/" << std::endl;
        }

        return elapsed.count();
    }

private:
    const ExecutionSettings& m_settings;
};

/* ========== Application Entry Point ========== */

int runApplication(int argc, char* argv[])
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;

    // Parse command line
    ExecutionSettings settings;
    ArgumentProcessor parser(argc, argv);
    
    if (!parser.process(settings))
        return 1;

    // Prepare output directory
    if (settings.writeFrames && !ensureDirectoryExists(settings.outputFolder))
        return 1;

    // Display startup info
    if (!settings.performanceMode)
    {
        std::cout << "[INFO] Initializing rigid body simulation..." << std::endl
                  << "[INFO] Mesh: " << settings.geometryFile << std::endl
                  << "[INFO] Objects: " << settings.entityCount << std::endl
                  << "[INFO] Frames: " << settings.frameLimit << std::endl;
    }

    // Create and configure simulation
    phys3d::SimulationController controller;
    SimulationRunner runner(settings);

    const auto setupStart = Clock::now();
    if (!runner.initialize(controller))
        return 1;
    const auto setupEnd = Clock::now();

    TimingData timing;
    timing.setupMs = Duration(setupEnd - setupStart).count();
    timing.simulationMs = runner.execute(controller);

    // Report results
    if (settings.performanceMode)
        ResultReporter::outputCSVLine(settings, timing);
    else
        ResultReporter::displayTimings(settings, timing);

    return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
    return runApplication(argc, argv);
}
