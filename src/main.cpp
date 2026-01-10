/**
 * @file main.cpp
 * @brief Entry point for the rigid body simulation with CLI support.
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

// ============================================================================
// Simulation Configuration
// ============================================================================

struct SimConfig {
    std::string meshPath    = "assets/diamond.obj";
    std::string outputDir   = "output";
    int numObjects          = 20;
    int totalFrames         = 200;
    float spawnPadding      = 3.0f;
    float minSpawnHeight    = 2.0f;
    bool verbose            = true;
    bool exportFrames       = true;
    bool benchmarkMode      = false;

    rigid::Vec3 sceneMin{-10.0f, -10.0f, 0.0f};
    rigid::Vec3 sceneMax{10.0f, 10.0f, 10.0f};

    // Randomization ranges for object properties
    float massMin           = 0.5f;
    float massMax           = 3.0f;
    float scaleMin          = 0.5f;
    float scaleMax          = 2.0f;
    float restitutionMin    = 0.2f;
    float restitutionMax    = 0.9f;
    float frictionMin       = 0.3f;
    float frictionMax       = 0.8f;
    float velocityMin       = -3.0f;
    float velocityMax       = 3.0f;
};

// ============================================================================
// Command Line Parsing
// ============================================================================

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  -m, --mesh <path>       Path to mesh file (default: assets/diamond.obj)\n"
              << "  -n, --num-objects <n>   Number of objects (default: 20)\n"
              << "  -f, --frames <n>        Number of frames (default: 200)\n"
              << "  -o, --output <dir>      Output directory (default: output)\n"
              << "\nPhysics randomization ranges:\n"
              << "  --mass <min> <max>      Mass range (default: 0.5 3.0)\n"
              << "  --scale <min> <max>     Scale/radius range (default: 0.5 2.0)\n"
              << "  --restitution <min> <max>  Bounciness range (default: 0.2 0.9)\n"
              << "  --friction <min> <max>  Friction range (default: 0.3 0.8)\n"
              << "  --velocity <min> <max>  Initial velocity range (default: -3.0 3.0)\n"
              << "\nOther options:\n"
              << "  --no-export             Skip frame export (for benchmarking)\n"
              << "  --benchmark             Benchmark mode: minimal output, timing only\n"
              << "  -q, --quiet             Quiet mode: suppress per-frame output\n"
              << "  -h, --help              Show this help message\n"
              << std::endl;
}

bool parseArgs(int argc, char* argv[], SimConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        } else if ((arg == "-m" || arg == "--mesh") && i + 1 < argc) {
            config.meshPath = argv[++i];
        } else if ((arg == "-n" || arg == "--num-objects") && i + 1 < argc) {
            config.numObjects = std::stoi(argv[++i]);
        } else if ((arg == "-f" || arg == "--frames") && i + 1 < argc) {
            config.totalFrames = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.outputDir = argv[++i];
        } else if (arg == "--mass" && i + 2 < argc) {
            config.massMin = std::stof(argv[++i]);
            config.massMax = std::stof(argv[++i]);
        } else if (arg == "--scale" && i + 2 < argc) {
            config.scaleMin = std::stof(argv[++i]);
            config.scaleMax = std::stof(argv[++i]);
        } else if (arg == "--restitution" && i + 2 < argc) {
            config.restitutionMin = std::stof(argv[++i]);
            config.restitutionMax = std::stof(argv[++i]);
        } else if (arg == "--friction" && i + 2 < argc) {
            config.frictionMin = std::stof(argv[++i]);
            config.frictionMax = std::stof(argv[++i]);
        } else if (arg == "--velocity" && i + 2 < argc) {
            config.velocityMin = std::stof(argv[++i]);
            config.velocityMax = std::stof(argv[++i]);
        } else if (arg == "--no-export") {
            config.exportFrames = false;
        } else if (arg == "--benchmark") {
            config.benchmarkMode = true;
            config.verbose = false;
            config.exportFrames = false;
        } else if (arg == "-q" || arg == "--quiet") {
            config.verbose = false;
        } else {
            std::cerr << "[ERROR] Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// ============================================================================
// Random Number Generation
// ============================================================================

class RandomGenerator {
public:
    static RandomGenerator& instance() {
        static RandomGenerator gen;
        return gen;
    }

    void seed(unsigned int s) {
        engine_.seed(s);
    }

    float uniform(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(engine_);
    }

    rigid::Quat randomOrientation() {
        constexpr float kPi = 3.14159265358979323846f;

        float u1 = uniform(0.0f, 1.0f);
        float u2 = uniform(0.0f, 1.0f);
        float u3 = uniform(0.0f, 1.0f);

        float sqrt1u1 = std::sqrt(1.0f - u1);
        float sqrtu1 = std::sqrt(u1);

        return rigid::Quat(
            sqrt1u1 * std::sin(2.0f * kPi * u2),
            sqrt1u1 * std::cos(2.0f * kPi * u2),
            sqrtu1 * std::sin(2.0f * kPi * u3),
            sqrtu1 * std::cos(2.0f * kPi * u3)
        ).normalized();
    }

private:
    RandomGenerator() : engine_(42) {}  // Fixed seed for reproducibility
    std::mt19937 engine_;
};

// ============================================================================
// Utility Functions
// ============================================================================

bool ensureOutputDirectory(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::cerr << "[ERROR] Could not create output directory: " << dir
                  << " (" << ec.message() << ")" << std::endl;
        return false;
    }
    return true;
}

std::string formatFrameFilename(const std::string& dir, int frame) {
    std::ostringstream ss;
    ss << dir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    return ss.str();
}

// ============================================================================
// Timing Utilities
// ============================================================================

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct TimingResult {
    double setupTimeMs = 0.0;
    double simulationTimeMs = 0.0;
    double totalTimeMs = 0.0;
    double avgFrameTimeMs = 0.0;
    int numFrames = 0;
    int numObjects = 0;
    std::string meshPath;
};

void printTimingResult(const TimingResult& result) {
    std::cout << "\n========== TIMING RESULTS ==========\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Mesh:              " << result.meshPath << "\n";
    std::cout << "Objects:           " << result.numObjects << "\n";
    std::cout << "Frames:            " << result.numFrames << "\n";
    std::cout << "------------------------------------\n";
    std::cout << "Setup time:        " << result.setupTimeMs << " ms\n";
    std::cout << "Simulation time:   " << result.simulationTimeMs << " ms\n";
    std::cout << "Total time:        " << result.totalTimeMs << " ms\n";
    std::cout << "Avg frame time:    " << result.avgFrameTimeMs << " ms\n";
    std::cout << "FPS:               " << (1000.0 / result.avgFrameTimeMs) << "\n";
    std::cout << "====================================\n";
}

void printBenchmarkCSV(const TimingResult& result) {
    // CSV format: mesh,objects,frames,setup_ms,sim_ms,total_ms,avg_frame_ms,fps
    std::cout << result.meshPath << ","
              << result.numObjects << ","
              << result.numFrames << ","
              << std::fixed << std::setprecision(3)
              << result.setupTimeMs << ","
              << result.simulationTimeMs << ","
              << result.totalTimeMs << ","
              << result.avgFrameTimeMs << ","
              << (1000.0 / result.avgFrameTimeMs) << std::endl;
}

// ============================================================================
// Simulation Setup
// ============================================================================

bool setupSimulation(rigid::Simulator& sim, const SimConfig& config) {
    sim.initialize();
    sim.setEnvironmentBounds(config.sceneMin, config.sceneMax);
    sim.setMaxIterations(20);
    sim.setTolerance(1e-3f);

    auto& rng = RandomGenerator::instance();
    rng.seed(42);  // Reset seed for reproducibility

    for (int i = 0; i < config.numObjects; ++i) {
        // Generate random physical properties for this body
        float mass = rng.uniform(config.massMin, config.massMax);
        float scale = rng.uniform(config.scaleMin, config.scaleMax);
        float restitution = rng.uniform(config.restitutionMin, config.restitutionMax);
        float friction = rng.uniform(config.frictionMin, config.frictionMax);

        std::string name = "body_" + std::to_string(i);
        rigid::RigidBody& body = sim.addBody(name, config.meshPath,
                                              mass, scale, restitution, friction);

        if (!body.hasMesh()) {
            std::cerr << "[ERROR] Failed to load mesh for body " << i
                      << ". Check file path: " << config.meshPath << std::endl;
            return false;
        }

        // Adjust spawn padding based on scale to prevent initial overlaps
        float effectivePadding = config.spawnPadding * scale;

        // Random initial position
        float x = rng.uniform(config.sceneMin.x() + effectivePadding,
                              config.sceneMax.x() - effectivePadding);
        float y = rng.uniform(config.sceneMin.y() + effectivePadding,
                              config.sceneMax.y() - effectivePadding);
        float z = rng.uniform(config.minSpawnHeight + effectivePadding,
                              config.sceneMax.z() - effectivePadding);

        // Random initial velocity
        float vx = rng.uniform(config.velocityMin, config.velocityMax);
        float vy = rng.uniform(config.velocityMin, config.velocityMax);
        float vz = rng.uniform(config.velocityMin, config.velocityMax);

        rigid::BodyState state = body.state();  // Get current state (has scale)
        state.position = rigid::Vec3(x, y, z);
        state.orientation = rng.randomOrientation();
        state.linearVel = rigid::Vec3(vx, vy, vz);

        body.setState(state);

        if (config.verbose && !config.benchmarkMode) {
            std::cout << "[INFO] Body " << i << ": scale=" << scale
                      << ", mass=" << mass << ", restitution=" << restitution
                      << ", friction=" << friction << std::endl;
        }
    }

    return true;
}

// ============================================================================
// Main Simulation Loop
// ============================================================================

double runSimulation(rigid::Simulator& sim, const SimConfig& config) {
    if (config.verbose) {
        std::cout << "[INFO] Starting simulation for " << config.totalFrames
                  << " frames..." << std::endl;
    }

    auto startTime = Clock::now();

    for (int frame = 0; frame < config.totalFrames; ++frame) {
        if (config.exportFrames) {
            std::string filename = formatFrameFilename(config.outputDir, frame);
            sim.exportFrame(filename);
        }

        sim.step();

        if (config.verbose) {
            std::cout << "[INFO] Frame " << (frame + 1) << "/" << config.totalFrames
                      << " completed." << std::endl;
        }
    }

    auto endTime = Clock::now();
    Duration elapsed = endTime - startTime;

    if (config.verbose) {
        std::cout << "[INFO] Simulation finished." << std::endl;
        if (config.exportFrames) {
            std::cout << "[INFO] Output saved to " << config.outputDir << "/" << std::endl;
        }
    }

    return elapsed.count();
}

}  // namespace

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    SimConfig config;

    if (!parseArgs(argc, argv, config)) {
        return 1;
    }

    if (config.exportFrames && !ensureOutputDirectory(config.outputDir)) {
        return 1;
    }

    if (!config.benchmarkMode) {
        std::cout << "[INFO] Initializing rigid body simulation..." << std::endl;
        std::cout << "[INFO] Mesh: " << config.meshPath << std::endl;
        std::cout << "[INFO] Objects: " << config.numObjects << std::endl;
        std::cout << "[INFO] Frames: " << config.totalFrames << std::endl;
    }

    rigid::Simulator sim;

    // Measure setup time
    auto setupStart = Clock::now();
    if (!setupSimulation(sim, config)) {
        return 1;
    }
    auto setupEnd = Clock::now();
    Duration setupTime = setupEnd - setupStart;

    // Run simulation and measure time
    double simTimeMs = runSimulation(sim, config);

    // Collect timing results
    TimingResult result;
    result.meshPath = config.meshPath;
    result.numObjects = config.numObjects;
    result.numFrames = config.totalFrames;
    result.setupTimeMs = setupTime.count();
    result.simulationTimeMs = simTimeMs;
    result.totalTimeMs = result.setupTimeMs + result.simulationTimeMs;
    result.avgFrameTimeMs = simTimeMs / config.totalFrames;

    // Output results
    if (config.benchmarkMode) {
        printBenchmarkCSV(result);
    } else {
        printTimingResult(result);
    }

    return 0;
}
