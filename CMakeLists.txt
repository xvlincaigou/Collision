cmake_minimum_required(VERSION 3.17)

project(CollisionSimulator LANGUAGES CXX CUDA)

# ============================================================================
# Build Settings
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ============================================================================
# Dependencies
# ============================================================================

find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found. Fetching Eigen 3.4.0...")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_MakeAvailable(eigen)
endif()

# ============================================================================
# Source Files
# ============================================================================

set(CORE_SOURCES
    src/core/mesh.cpp
    src/core/mesh_cache.cpp
)

set(ACCEL_SOURCES
    src/accel/bvh.cpp
)

set(PHYSICS_SOURCES
    src/physics/rigid_body.cpp
    src/physics/collision_detector.cpp
    src/physics/integrator.cpp
)

set(SCENE_SOURCES
    src/scene/environment.cpp
    src/scene/scene.cpp
)

set(SIMULATION_SOURCES
    src/simulation/simulator.cpp
)

set(CUDA_SOURCES
    src/gpu/bvh_builder.cu
    src/gpu/broadphase.cu
    src/gpu/collision_detector.cu
)

set(ALL_SOURCES
    src/main.cpp
    ${CORE_SOURCES}
    ${ACCEL_SOURCES}
    ${PHYSICS_SOURCES}
    ${SCENE_SOURCES}
    ${SIMULATION_SOURCES}
    ${CUDA_SOURCES}
)

# ============================================================================
# Executable Target
# ============================================================================

add_executable(${PROJECT_NAME} ${ALL_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    NO_SYSTEM_FROM_IMPORTED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    TBB::tbb
    Eigen3::Eigen
)

# ============================================================================
# Compiler Options
# ============================================================================

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20014>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20013>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20015>
)

# Warnings for C++
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

# ============================================================================
# Post-Build: Copy Assets
# ============================================================================

if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets
            ${CMAKE_CURRENT_BINARY_DIR}/assets
        COMMENT "Copying assets to build directory..."
    )
endif()

# ============================================================================
# Install Target (Optional)
# ============================================================================

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(DIRECTORY assets/
    DESTINATION share/${PROJECT_NAME}/assets
    OPTIONAL
)
