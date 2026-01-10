# Rigid Body Simulator

A high-performance rigid body collision simulation engine with CUDA acceleration.

## Features

- **GPU-Accelerated Collision Detection**: BVH construction and traversal on CUDA
- **Implicit Euler Integration**: Stable time-stepping with penalty-based contact response
- **Parallel Processing**: TBB-based parallelization for CPU operations
- **Mesh Support**: OBJ file loading with automatic caching

## Dependencies

- **CMake** 3.17+
- **CUDA Toolkit** (tested with CUDA 11+)
- **TBB** (`libtbb-dev`)
- **Eigen3** (`libeigen3-dev`) - auto-fetched if not found
- **C++17** compatible compiler

### Installing TBB (Ubuntu/Debian)

```bash
apt-get update && apt-get install -y libtbb-dev
```

## Building and Running

```bash
bash run.sh
```

This will:
1. Configure the project with CMake
2. Build in Release mode
3. Run the simulation
4. Output frames to `simulation/output/`

## Project Structure

```
src/
├── core/                    # Core utilities and data structures
│   ├── common.h            # Type definitions, AABB, Plane
│   ├── mesh.h/cpp          # Triangle mesh with BVH support
│   └── mesh_cache.h/cpp    # Thread-safe mesh caching
│
├── accel/                   # Acceleration structures
│   └── bvh.h/cpp           # Bounding Volume Hierarchy
│
├── physics/                 # Physics simulation
│   ├── body_properties.h   # Mass, inertia, material properties
│   ├── rigid_body.h/cpp    # Rigid body state and dynamics
│   ├── contact.h           # Contact information
│   ├── collision_detector.h/cpp  # Collision detection system
│   ├── force_builder.h     # Force/Jacobian assembly
│   └── integrator.h/cpp    # Time integration
│
├── scene/                   # Scene management
│   ├── environment.h/cpp   # Simulation boundaries
│   └── scene.h/cpp         # Scene container
│
├── simulation/              # High-level simulation
│   └── simulator.h/cpp     # Main simulation controller
│
├── gpu/                     # CUDA implementations
│   ├── bvh_builder.cuh/cu  # GPU BVH construction (LBVH)
│   ├── broadphase.cuh/cu   # GPU broadphase detection
│   └── collision_detector.cuh/cu  # GPU collision kernels
│
└── main.cpp                 # Entry point
```

## Architecture

### Namespace: `rigid`

All code is organized under the `rigid` namespace, with GPU-specific code in `rigid::gpu`.

### Key Classes

| Class | Description |
|-------|-------------|
| `Simulator` | High-level simulation controller |
| `Scene` | Container for rigid bodies and environment |
| `RigidBody` | Body with mesh, properties, and state |
| `Mesh` | Triangle mesh with optional BVH |
| `BVH` | Bounding Volume Hierarchy for acceleration |
| `CollisionDetector` | Detects body-body and body-environment collisions |
| `Integrator` | Implicit Euler time integration |

### GPU Acceleration

The simulator uses CUDA for:
- **LBVH Construction**: Linear BVH built using Morton codes
- **Broadphase Detection**: Sweep-and-prune on GPU
- **Narrowphase Collision**: BVH traversal for body-body contacts

## Configuration

Edit constants in `main.cpp` to customize:
- Number of objects
- Total frames
- Scene bounds
- Object mass
- Spawn positions

## Output

Frames are exported as OBJ files in `simulation/output/`:
```
frame_0000.obj
frame_0001.obj
...
```

## License

MIT License
