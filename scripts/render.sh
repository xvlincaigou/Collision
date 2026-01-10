#!/usr/bin/env bash
#
# render.sh - Render OBJ sequence to video using PyVista
#
# Usage:
#   ./render.sh                           # Render with default settings
#   ./render.sh --input simulation/output # Custom input directory
#   ./render.sh --interactive             # Interactive preview
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

# Default values
INPUT_DIR="simulation/output"
OUTPUT_DIR="render"
FPS=30
WIDTH=1920
HEIGHT=1080
INTERACTIVE=false
CREATE_VIDEO=true

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Render OBJ sequence to video/images using PyVista"
    echo ""
    echo "Options:"
    echo "  -i, --input DIR      Input directory with OBJ files (default: simulation/output)"
    echo "  -o, --output DIR     Output directory (default: render)"
    echo "  --interactive        Open interactive viewer"
    echo "  --fps N              Video FPS (default: 30)"
    echo "  --width N            Output width (default: 1920)"
    echo "  --height N           Output height (default: 1080)"
    echo "  --no-video           Don't create video, only images"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Render simulation/output/"
    echo "  $0 -i output/ -o my_render/           # Custom directories"
    echo "  $0 --interactive                      # Interactive preview"
}

check_pyvista() {
    python3 -c "import pyvista" 2>/dev/null
}

install_pyvista() {
    echo -e "${YELLOW}[INSTALL]${NC} Installing PyVista and dependencies..."
    pip3 install pyvista imageio imageio-ffmpeg tqdm numpy --quiet
    echo -e "${GREEN}[INSTALL]${NC} Done!"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --no-video)
            CREATE_VIDEO=false
            shift
            ;;
        -h|--help)
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

# Check input directory
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo -e "${RED}[ERROR]${NC} Input directory not found: ${INPUT_DIR}"
    echo "  Run the simulation first to generate OBJ files."
    exit 1
fi

# Count OBJ files
OBJ_COUNT=$(ls -1 "${INPUT_DIR}"/frame_*.obj 2>/dev/null | wc -l)
if [[ ${OBJ_COUNT} -eq 0 ]]; then
    echo -e "${RED}[ERROR]${NC} No OBJ files found in: ${INPUT_DIR}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}              ${GREEN}OBJ Sequence Renderer${NC}                             ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Input:      ${INPUT_DIR} (${OBJ_COUNT} frames)"
echo -e "Output:     ${OUTPUT_DIR}"
echo -e "Resolution: ${WIDTH}x${HEIGHT}"
echo -e "FPS:        ${FPS}"
echo ""

# Check PyVista
if ! check_pyvista; then
    echo -e "${YELLOW}[INFO]${NC} PyVista not found."
    read -p "Install PyVista and dependencies? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        install_pyvista
    else
        echo -e "${RED}[ERROR]${NC} PyVista required for rendering."
        exit 1
    fi
fi

if ${INTERACTIVE}; then
    echo -e "${YELLOW}[RENDER]${NC} Opening interactive viewer..."
    python3 "${SCRIPT_DIR}/render_pyvista.py" \
        --input "${INPUT_DIR}" \
        --interactive
else
    echo -e "${YELLOW}[RENDER]${NC} Rendering with PyVista..."
    
    VIDEO_FLAG=""
    if ${CREATE_VIDEO}; then
        VIDEO_FLAG="--video"
    fi
    
    python3 "${SCRIPT_DIR}/render_pyvista.py" \
        --input "${INPUT_DIR}" \
        --output "${OUTPUT_DIR}" \
        --width ${WIDTH} \
        --height ${HEIGHT} \
        --fps ${FPS} \
        ${VIDEO_FLAG}
fi

echo ""
echo -e "${GREEN}[DONE]${NC} Rendering complete!"

if ${CREATE_VIDEO} && [[ -f "${OUTPUT_DIR}/simulation.mp4" ]]; then
    echo -e "Video: ${OUTPUT_DIR}/simulation.mp4"
fi

if [[ -d "${OUTPUT_DIR}/frames" ]]; then
    FRAME_COUNT=$(ls -1 "${OUTPUT_DIR}"/frames/*.png 2>/dev/null | wc -l)
    echo -e "Frames: ${OUTPUT_DIR}/frames/ (${FRAME_COUNT} images)"
fi
