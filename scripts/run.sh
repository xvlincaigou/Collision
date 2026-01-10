set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${ROOT_DIR}/build"
SIM_DIR="${ROOT_DIR}/simulation"
ASSETS_DIR="${ROOT_DIR}/assets"
BINARY_NAME="CollisionSimulator"
BIN_PATH="${BUILD_DIR}/${BINARY_NAME}"

cmake_configure() {
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
}

cmake_build() {
    cmake --build "${BUILD_DIR}" --parallel "$(nproc --ignore=1 2>/dev/null || nproc)"
}

echo "[run.sh] Configuring..."
cmake_configure
echo "[run.sh] Building..."
cmake_build

rm -rf "${SIM_DIR}"
mkdir -p "${SIM_DIR}/output"

cp "${BIN_PATH}" "${SIM_DIR}/${BINARY_NAME}"
cp -r "${ASSETS_DIR}" "${SIM_DIR}/assets"

pushd "${SIM_DIR}" >/dev/null
./${BINARY_NAME}
popd >/dev/null
