# 基于 CUDA 的刚体碰撞检测与仿真系统

## 1. 程序运行环境

### 1.1 硬件要求
- **GPU**: NVIDIA RTX 3060
- **CPU**: 12th Gen Intel(R) Core(TM) i7-12700H
- **内存**: 16GB

### 1.2 软件依赖

| 依赖项 | 版本要求 |
|--------|----------|
| CMake | ≥ 3.17 |
| CUDA Toolkit | 12.3/12.5 |
| TBB (libtbb-dev) | - |
| Eigen3 | ≥ 3.3 |
| C++ 编译器 | g++ 9.4.0 |

### 1.3 安装依赖（WSL Ubuntu 20.04.2）

```bash
# 安装 TBB
sudo apt-get update && sudo apt-get install -y libtbb-dev

# 安装 Eigen3（可选，CMake 会自动下载）
sudo apt-get install -y libeigen3-dev

# CUDA Toolkit
```

---

## 2. 程序模块结构与逻辑关系

### 2.1 目录结构

```
Collision/
├── CMakeLists.txt              
├── README.md               
├── assets/                     # 网格资源
├── scripts/                    # 工具脚本
├── src/                        # 源代码根目录
│   ├── main.cpp                # 程序入口
│   ├── core/                   # 核心基础模块
│   ├── accel/                  # 加速结构模块
│   ├── physics/                # 物理仿真模块
│   ├── scene/                  # 场景管理模块
│   ├── simulation/             # 仿真控制模块
│   └── gpu/                    # CUDA GPU 模块
├── simulation/                 # 运行输出目录
└── render/                     # 渲染输出目录
```

### 2.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.cpp                                │
│                   (ArgumentProcessor/SimulationRunner)           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 simulation/controller                            │
│               (仿真控制/帧循环/OBJ导出)                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌─────────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
│    scene/world      │ │    physics/     │ │  core/               │
│ (场景/刚体/边界管理) │ │     solver      │ │    resource_pool     │
└──────────┬──────────┘ └──────┬──────────┘ └──────────────────────┘
           │                   │
           │       ┌───────────┴───────────┐
           │       ▼                       ▼
           │ ┌─────────────────┐  ┌─────────────────────────┐
           │ │ physics/        │  │ physics/                │
           │ │   dynamics      │  │   proximity             │
           │ │ (力的组装)       │  │ (碰撞检测调度)           │
           │ └─────────────────┘  └────────────┬────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────┐            ┌───────────────────────────────┐
│ physics/            │            │         gpu/ 模块              │
│   entity            │            │  ┌─────────────────────────┐  │
│ (组件式刚体状态)     │            │  │  gpu/tree_kernel        │  │
└──────────┬──────────┘            │  │  (Morton码/warp归约构建) │  │
           │                       │  └─────────────────────────┘  │
           ▼                       │  ┌─────────────────────────┐  │
┌─────────────────────┐            │  │  gpu/contact_kernel     │  │
│  core/surface       │◄───────────│  │  (精细碰撞检测)          │  │
│   (三角网格)         │            │  └─────────────────────────┘  │
└──────────┬──────────┘            │  ┌─────────────────────────┐  │
           │                       │  │  gpu/sweep_kernel       │  │
           ▼                       │  │  (Sweep & Prune)        │  │
┌─────────────────────┐            │  └─────────────────────────┘  │
│  accel/spatial_index│◄───────────└───────────────────────────────┘
│  (迭代式BVH构建)     │
└─────────────────────┘
```

### 2.3 模块职责说明

| 模块 | 文件 | 主要职责 |
|------|------|----------|
| `core/` | types.h, surface.h/.cpp, resource_pool.h/.cpp | 基础类型定义、三角网格与资源池 |
| `accel/` | spatial_index.h/.cpp | 迭代式BVH加速结构 |
| `physics/` | material.h, entity.h/.cpp, intersection.h, proximity.h/.cpp, dynamics.h, solver.h/.cpp | 组件式刚体、碰撞检测、力计算、时间积分 |
| `scene/` | boundary.h/.cpp, world.h/.cpp | 场景管理、环境边界 |
| `simulation/` | controller.h/.cpp | 顶层仿真控制、帧导出 |
| `gpu/` | tree_kernel.cu/.cuh, contact_kernel.cu/.cuh, sweep_kernel.cu/.cuh | CUDA加速（warp级优化） |

---

## 3. 功能演示方法

### 3.1 一键构建与运行

```bash
cd /path/to/Collision
bash scripts/run.sh
```

脚本将自动完成：
1. CMake 配置（Release 模式）
2. 并行编译
3. 创建 `simulation/` 目录并复制资源
4. 运行仿真程序

### 3.2 命令行参数

```bash
./CollisionSimulator [options]

# 基本参数
  -m, --mesh <path>       网格文件路径 (默认: assets/diamond.obj)
  -n, --num-objects <n>   物体数量 (默认: 20)
  -f, --frames <n>        仿真帧数 (默认: 200)
  -o, --output <dir>      输出目录 (默认: output)

# 物理属性随机化范围
  --mass <min> <max>          质量范围 (默认: 0.5 3.0)
  --scale <min> <max>         缩放/半径范围 (默认: 0.5 2.0)
  --restitution <min> <max>   弹性系数范围 (默认: 0.2 0.9)
  --friction <min> <max>      摩擦系数范围 (默认: 0.3 0.8)
  --velocity <min> <max>      初速度范围 (默认: -3.0 3.0)

# 其他选项
  --no-export             跳过帧导出 (用于性能测试)
  --benchmark             基准测试模式
  -q, --quiet             安静模式
  -h, --help              显示帮助
```

#### 使用示例

```bash
./build/CollisionSimulator -n 30 -f 500 \
    --scale 0.4 2.5 \
    --mass 0.3 4.0 \
    --velocity -5.0 5.0
```

#### 运行时输出

程序会显示每个物体的随机属性：

```
[INFO] Entity 0: scale=1.69, mass=1.44, restitution=0.87, friction=0.39
[INFO] Entity 1: scale=1.80, mass=1.65, restitution=0.43, friction=0.60
[INFO] Entity 2: scale=0.50, mass=1.03, restitution=0.33, friction=0.80
...
```

### 3.3 输出结果

仿真结果以 OBJ 帧序列形式输出（默认目录为 `output/`）：

```
output/
├── frame_0000.obj
├── frame_0001.obj
├── frame_0002.obj
└── ...
```

每帧 OBJ 文件包含所有刚体的世界坐标顶点。

### 3.4 动画渲染

```bash
# 安装依赖
pip install pyvista imageio imageio-ffmpeg tqdm numpy

# 一键渲染
./scripts/render.sh

# 或使用脚本
python3 scripts/render_pyvista.py -i output -o render --video

# 交互式预览
./scripts/render.sh --interactive
```

#### 渲染参数

```bash
./scripts/render.sh [选项]
  -i, --input DIR      输入目录 (默认: simulation/output)
  -o, --output DIR     输出目录 (默认: render)
  --interactive        交互式预览
  --fps N              视频帧率 (默认: 30)
  --width N            输出宽度 (默认: 1920)
  --height N           输出高度 (默认: 1080)
```

#### 输出文件

```
render/
├── simulation.mp4      # 视频文件
└── frames/             # 逐帧 PNG 图片
    ├── frame_0000.png
    ├── frame_0001.png
    └── ...
```

---

### Windows平台编译
要求：
- VS 2022
- vckpg
- cmake

```pwsh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
```

对于编译好的 exe 文件，可以使用与unix平台上相同的参数去使用。

## 4. 程序运行主要流程

```
┌──────────────────┐
│   程序启动        │  ArgumentProcessor 解析命令行参数
└────────┬─────────┘
         ▼
┌──────────────────┐
│  SimulationRunner│  SimulationController::initialize()
│  初始化仿真器     │
│  - 配置积分器    │
│  - 创建边界      │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  EntitySpawner   │  controller::createEntity()
│  生成刚体对象     │
│  - 加载网格      │  resource_pool::fetch()
│  - 构建 GPU BVH  │  spatial_index::constructOnDevice()
│  - 随机物理属性  │  mass, scale, restitution, friction
│  - 设置初始状态  │  translation, orientation, velocity
└────────┬─────────┘
         ▼
┌──────────────────────────────────────────┐
│              仿真主循环                    │
│  for frame in [0, kTotalFrames):         │
│  ┌─────────────────────────────────────┐ │
│  │  1. 导出当前帧                       │ │
│  │     exportFrame() → OBJ 文件        │ │
│  ├─────────────────────────────────────┤ │
│  │  2. 更新世界包围盒                   │ │
│  │     entity::worldExtent()           │ │
│  ├─────────────────────────────────────┤ │
│  │  3. 碰撞检测                        │ │
│  │  ┌───────────────────────────────┐  │ │
│  │  │ 3.1 刚体-环境碰撞 (GPU)        │  │ │
│  │  │     迭代式BVH遍历 + 平面检测   │  │ │
│  │  ├───────────────────────────────┤  │ │
│  │  │ 3.2 粗检测 (GPU Broadphase)   │  │ │
│  │  │     Sweep & Prune 算法        │  │ │
│  │  ├───────────────────────────────┤  │ │
│  │  │ 3.3 精细检测 (GPU Narrowphase)│  │ │
│  │  │     顶点-三角形距离查询        │  │ │
│  │  └───────────────────────────────┘  │ │
│  ├─────────────────────────────────────┤ │
│  │  4. 力的组装                        │ │
│  │     ForceAssembler                  │ │
│  │     - 重力                          │ │
│  │     - 惩罚碰撞力                    │ │
│  ├─────────────────────────────────────┤ │
│  │  5. 隐式欧拉积分                    │ │
│  │     (M - h²K) Δv = h·F             │ │
│  │     共轭梯度法求解                   │ │
│  ├─────────────────────────────────────┤ │
│  │  6. 状态更新                        │ │
│  │     位置、速度、姿态                 │ │
│  └─────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         ▼
┌──────────────────┐
│    仿真结束       │
└──────────────────┘
```

## 5. 算法设计与实现

### 5.1 BVH 构建（迭代式 + GPU 加速）

采用 **LBVH (Linear BVH)** 算法，基于 Morton 码的并行构建，支持迭代式CPU构建和GPU加速构建两种模式：

#### 5.1.1 Morton 码计算（warp级优化）

```cpp
// tree_kernel.cu::kernelComputePrimitiveData
// 使用warp级归约优化全局边界计算
__shared__ CudaFloat3 sharedMin[32];
__shared__ CudaFloat3 sharedMax[32];

// Warp级归约
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    localMin.x = fminf(localMin.x, __shfl_down_sync(0xffffffff, localMin.x, offset));
    // ...
}
```

#### 5.1.2 迭代式树构建（CPU版本）

```cpp
// spatial_index.cpp::construct() - 使用显式栈替代递归
using WorkItem = std::tuple<IntType, IntType, IntType, bool, IntType>;
std::stack<WorkItem> workStack;

while (!workStack.empty()) {
    auto [rangeStart, rangeEnd, parentIdx, isLeft, depth] = workStack.top();
    workStack.pop();
    
    // 构建节点并按需推入子任务
    if (!shouldTerminate) {
        workStack.push(rightChildTask);
        workStack.push(leftChildTask);
    }
}
```

#### 5.1.3 自底向上边界传播

```cpp
__global__ void kernelPropagateBoundsBottomUp(...) {
    // 从叶节点开始
    int leafIdx = faceTotal - 1 + tid;
    nodeData[leafIdx].volume = primBoundsData[primIdx];
    
    // 原子计数器确保等待两个子节点
    int current = parentData[leafIdx];
    while (current >= 0) {
        int old = atomicAdd(&visitFlags[current], 1);
        if (old == 0) return;  // 第一个到达的线程退出
        
        // 第二个线程合并包围盒
        nodeData[current].volume = mergeBounds(leftBounds, rightBounds);
        current = parentData[current];
    }
}
```

### 5.2 粗检测（Broadphase）

采用 **Sweep and Prune** 算法的 GPU 实现：

```
算法步骤:
1. 为每个 AABB 创建端点 (min, max) × 3 轴
2. 按 X 轴排序端点
3. 扫描检测重叠:
   - 遇到 min 端点: 与当前活跃集合检测 YZ 重叠
   - 遇到 max 端点: 从活跃集合移除
```

### 5.3 精细检测（Narrowphase）

#### 5.3.1 刚体-环境碰撞

使用迭代式BVH遍历进行顶点-平面距离检测：

```cpp
// proximity.cpp
void traverseTreeAgainstPlane(...) {
    std::array<IntType, kMaxStackDepth> nodeStack;
    IntType stackTop = 0;
    nodeStack[stackTop++] = tree.rootIdx();
    
    while (stackTop > 0) {
        const IntType nodeIdx = nodeStack[--stackTop];
        const TreeNode& node = tree.nodeAt(nodeIdx);
        
        if (!nodeIntersectsPlane(node.volume, localPlane))
            continue;
        
        if (node.isTerminal()) {
            // 处理叶节点
        } else {
            // 推入子节点
            if (node.childRight != -1) nodeStack[stackTop++] = node.childRight;
            if (node.childLeft != -1) nodeStack[stackTop++] = node.childLeft;
        }
    }
}
```

#### 5.3.2 刚体-刚体碰撞

采用顶点-三角形最近点查询，使用重心坐标法：

```cpp
__device__ CudaFloat3 computeTriangleClosestPoint(
    const CudaFloat3& query,
    const CudaFloat3& v0, const CudaFloat3& v1, const CudaFloat3& v2)
{
    // 边向量
    const CudaFloat3 e01 = v1 - v0;
    const CudaFloat3 e02 = v2 - v0;
    
    // 依次检查各区域：顶点、边、内部
    // 返回最近点
}
```

### 5.4 刚体组件式设计

采用组件化设计模式，分离关注点：

```cpp
struct CacheControl {
    bool extentValid = false;
    bool inertiaValid = false;
    void invalidateAll();
};

struct ForceAccumulator {
    Point3 force = Point3::Zero();
    Point3 torque = Point3::Zero();
    void addForceAtPoint(const Point3& f, const Point3& worldPt, const Point3& bodyCenter);
};

class DynamicEntity {
    CacheControl m_cache;           // 缓存有效性跟踪
    ForceAccumulator m_forces;      // 力/力矩累加器
    MaterialProperties m_material;  // 物理属性
    EntityState m_kinematic;        // 运动学状态
};
```

### 5.5 碰撞响应

采用 **惩罚力法 (Penalty Method)** 结合隐式欧拉积分：

#### 5.5.1 惩罚力计算

```cpp
// solver.cpp
void applyForceContribution(VectorN& forceVec, IntType entityIdx,
                            const Point3& entityPos, const Point3& contactPos,
                            const Point3& force, RealType sign)
{
    const Point3 scaledForce(force.x() * sign, force.y() * sign, force.z() * sign);
    DofAccessor::addLinear(forceVec, entityIdx, scaledForce);
    
    const Point3 torque = computeContactTorque(contactPos, entityPos, scaledForce);
    DofAccessor::addAngular(forceVec, entityIdx, torque);
}
```

#### 5.5.2 隐式积分

```
线性系统: (M - h²K) Δv = h·F

其中:
- M: 质量/惯性矩阵 (对角块)
- K: 刚度矩阵 (∂F/∂x)
- h: 时间步长 (0.01s)
- F: 外力 (重力 + 碰撞力)

求解方法: 共轭梯度法 (CG)
```

---

## 6. 性能测试与分析

### 6.1 测试环境

见 1.1 硬件要求。

### 6.2 网格信息

| 网格文件 | 顶点数 | 面数 | 文件大小 |
|----------|--------|------|----------|
| ball.obj | 2,050 | 4,096 | 271 KB |
| diamond.obj | 65 | 65 | 5 KB |

### 6.3 性能测试工具

项目提供了完整的性能测试脚本：

```bash
# 快速测试 (5-20 对象, 50 帧)
./scripts/run_benchmark.sh --quick

# 默认测试 (5-100 对象, 50-500 帧)  
./scripts/run_benchmark.sh

# 压力测试 (50-200 对象, 200-1000 帧)
./scripts/run_benchmark.sh --stress

# 极限测试 (100-300 对象, 500-1000 帧)
./scripts/run_benchmark.sh --extreme

# 完整测试 (多网格, 多配置)
./scripts/run_benchmark.sh --full
```

测试输出包含详细的网格和场景信息：

```
================================================================================
MESH INFORMATION
================================================================================
Mesh                        Vertices      Faces    Size (KB)
--------------------------------------------------------------------------------
ball.obj                        2050       4096        271.4
diamond.obj                       65         65          5.4
================================================================================

[1/5] ball_n50_f100 (50 objs × 4096 faces = 204800 triangles)
  → Total: 1234.56ms, Avg frame: 12.35ms, FPS: 81.00
```

CSV 输出字段：

| 字段 | 说明 |
|------|------|
| `mesh_verts` | 单个网格顶点数 |
| `mesh_faces` | 单个网格面数 |
| `mesh_size_kb` | OBJ 文件大小 |
| `total_verts` | 场景总顶点数 |
| `total_faces` | 场景总三角形数 |
| `setup_ms` | 初始化时间 |
| `sim_ms` | 仿真时间 |
| `avg_frame_ms` | 平均帧时间 |
| `fps` | 帧率 |

### 6.4 测试配置

| 模式 | 对象数 | 帧数 | 场景三角形数 |
|------|--------|------|--------------|
| quick | 5-20 | 50 | ~80K |
| default | 5-100 | 50-500 | ~400K |
| stress | 50-200 | 200-1000 | ~800K |
| extreme | 100-300 | 500-1000 | ~1.2M |

### 6.5 测试结果
参见仓库当中的csv文件

## 7. 参考文献

NVIDIA GPU Gems 3 - Collision Detection: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda

---

## 附录: 构建与运行命令参考

```bash
# 一键构建运行
./scripts/run.sh

# 手动构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# 运行仿真
./build/CollisionSimulator \
    --scale 0.5 2.0 \
    --mass 0.5 3.0 \
    --restitution 0.3 0.9 \
    --velocity -3.0 3.0

# 渲染动画
./scripts/render.sh

# 性能测试
./scripts/run_benchmark.sh              # 默认测试
./scripts/run_benchmark.sh --stress     # 压力测试
./scripts/run_benchmark.sh --extreme    # 极限测试

# Python 脚本直接调用
python3 scripts/benchmark.py --build -n 50 -n 100 -f 200 --runs 3
```
