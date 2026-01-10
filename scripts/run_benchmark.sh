#!/usr/bin/env bash
#
# run_benchmark.sh - Build and run benchmarks for the Rigid Body Simulator
#
# Usage:
#   ./run_benchmark.sh                  # Run default benchmark
#   ./run_benchmark.sh --quick          # Quick test
#   ./run_benchmark.sh --stress         # Stress test (heavy load)
#   ./run_benchmark.sh --full           # Full benchmark suite
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

BUILD_DIR="${ROOT_DIR}/build"
EXECUTABLE="${BUILD_DIR}/CollisionSimulator"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}       ${GREEN}Rigid Body Simulator - Benchmark Suite${NC}                    ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Benchmark modes:"
    echo "  --quick       Quick test (5-20 objects, 50 frames)"
    echo "  --default     Default benchmark (5-100 objects, 50-500 frames)"
    echo "  --stress      Stress test (50-200 objects, 200-1000 frames)"
    echo "  --full        Full suite (all configurations)"
    echo "  --extreme     Extreme stress test (200+ objects, 1000+ frames)"
    echo "  --custom      Custom (interactive prompts)"
    echo ""
    echo "Options:"
    echo "  --no-build    Skip the build step"
    echo "  --runs N      Number of runs per config for averaging (default: 1)"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Default benchmark"
    echo "  $0 --stress --runs 3  # Stress test, 3 runs averaged"
    echo "  $0 --extreme          # Maximum stress test"
}

print_mesh_info() {
    echo -e "${CYAN}[MESH INFO]${NC} Available meshes:"
    for obj in "${ROOT_DIR}"/assets/*.obj; do
        if [[ -f "$obj" ]]; then
            name=$(basename "$obj")
            size=$(du -h "$obj" | cut -f1)
            verts=$(grep -c "^v " "$obj" || echo 0)
            faces=$(grep -c "^f " "$obj" || echo 0)
            echo -e "  ${name}: ${verts} vertices, ${faces} faces, ${size}"
        fi
    done
    echo ""
}

build_project() {
    echo -e "${YELLOW}[BUILD]${NC} Configuring and building..."
    
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    cmake --build "${BUILD_DIR}" --parallel "$(nproc --ignore=1 2>/dev/null || nproc)" 2>&1 | \
        grep -E "^\[|Built target|error:" || true
    
    if [[ ! -x "${EXECUTABLE}" ]]; then
        echo -e "${RED}[ERROR]${NC} Build failed! Executable not found."
        exit 1
    fi
    
    echo -e "${GREEN}[BUILD]${NC} Success!"
}

run_quick_benchmark() {
    echo -e "${YELLOW}[BENCH]${NC} Running ${CYAN}QUICK${NC} benchmark..."
    python3 "${SCRIPT_DIR}/benchmark.py" \
        -n 5 -n 10 -n 20 \
        -f 50 \
        --runs "${RUNS}" \
        -o "benchmark_quick_$(date +%Y%m%d_%H%M%S).csv"
}

run_default_benchmark() {
    echo -e "${YELLOW}[BENCH]${NC} Running ${CYAN}DEFAULT${NC} benchmark..."
    python3 "${SCRIPT_DIR}/benchmark.py" \
        --runs "${RUNS}" \
        -o "benchmark_default_$(date +%Y%m%d_%H%M%S).csv"
}

run_stress_benchmark() {
    echo -e "${YELLOW}[BENCH]${NC} Running ${CYAN}STRESS${NC} benchmark..."
    python3 "${SCRIPT_DIR}/benchmark.py" \
        --stress \
        --runs "${RUNS}" \
        -o "benchmark_stress_$(date +%Y%m%d_%H%M%S).csv"
}

run_full_benchmark() {
    echo -e "${YELLOW}[BENCH]${NC} Running ${CYAN}FULL${NC} benchmark suite..."
    python3 "${SCRIPT_DIR}/benchmark.py" \
        -m assets/ball.obj -m assets/diamond.obj \
        -n 10 -n 20 -n 50 -n 100 -n 150 \
        -f 50 -f 100 -f 200 -f 500 \
        --runs "${RUNS}" \
        -o "benchmark_full_$(date +%Y%m%d_%H%M%S).csv"
}

run_extreme_benchmark() {
    echo -e "${YELLOW}[BENCH]${NC} Running ${RED}EXTREME${NC} stress test..."
    echo -e "${RED}[WARN]${NC} This may take a VERY long time!"
    python3 "${SCRIPT_DIR}/benchmark.py" \
        -n 100 -n 150 -n 200 -n 300 \
        -f 500 -f 1000 \
        --runs "${RUNS}" \
        --timeout 3600 \
        -o "benchmark_extreme_$(date +%Y%m%d_%H%M%S).csv"
}

run_custom_benchmark() {
    echo -e "${YELLOW}[CUSTOM]${NC} Custom benchmark configuration"
    
    read -p "Enter object counts (space-separated, e.g., 50 100 200): " -a OBJECTS
    read -p "Enter frame counts (space-separated, e.g., 100 500 1000): " -a FRAMES
    read -p "Enter number of runs per config (default: 1): " CUSTOM_RUNS
    CUSTOM_RUNS=${CUSTOM_RUNS:-1}
    
    # Build arguments
    OBJ_ARGS=""
    for n in "${OBJECTS[@]}"; do
        OBJ_ARGS="${OBJ_ARGS} -n ${n}"
    done
    
    FRAME_ARGS=""
    for f in "${FRAMES[@]}"; do
        FRAME_ARGS="${FRAME_ARGS} -f ${f}"
    done
    
    echo -e "${YELLOW}[BENCH]${NC} Running custom benchmark..."
    python3 "${SCRIPT_DIR}/benchmark.py" \
        ${OBJ_ARGS} \
        ${FRAME_ARGS} \
        --runs "${CUSTOM_RUNS}" \
        -o "benchmark_custom_$(date +%Y%m%d_%H%M%S).csv"
}

# Parse arguments
MODE="default"
NO_BUILD=false
RUNS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --default)
            MODE="default"
            shift
            ;;
        --stress)
            MODE="stress"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --extreme)
            MODE="extreme"
            shift
            ;;
        --custom)
            MODE="custom"
            shift
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
print_header
echo ""

# Show mesh info
print_mesh_info

# Build if needed
if [[ "${NO_BUILD}" == false ]]; then
    build_project
else
    echo -e "${YELLOW}[INFO]${NC} Skipping build step"
fi

echo ""

# Run the appropriate benchmark
case ${MODE} in
    quick)
        run_quick_benchmark
        ;;
    stress)
        run_stress_benchmark
        ;;
    full)
        run_full_benchmark
        ;;
    extreme)
        run_extreme_benchmark
        ;;
    custom)
        run_custom_benchmark
        ;;
    default)
        run_default_benchmark
        ;;
esac

echo ""
echo -e "${GREEN}[DONE]${NC} Benchmark completed!"
