# 基于 CUDA 的碰撞检测算法

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
├── CMakeLists.txt              # 构建配置
├── README.md               
├── assets/                     # 网格资源
│   ├── ball.obj           
│   └── diamond.obj        
├── scripts/                    # 工具脚本
│   ├── run.sh                  # 一键构建运行脚本
│   ├── render.sh               # 渲染脚本
│   ├── render_pyvista.py       # PyVista 渲染实现
│   ├── benchmark.py            # Python 性能测试
│   └── run_benchmark.sh        # 性能测试脚本
├── src/                        # 源代码根目录
│   ├── main.cpp                # 程序入口（含CLI解析）
│   │
│   ├── core/                   # 核心基础模块
│   │   ├── common.h            # 基础类型定义 (Vec3, Mat3, AABB, Plane...)
│   │   ├── mesh.h/.cpp         # 三角网格类
│   │   └── mesh_cache.h/.cpp   # 网格缓存管理
│   │
│   ├── accel/                  # 加速结构模块
│   │   └── bvh.h/.cpp          # BVH 层次包围盒树
│   │
│   ├── physics/                # 物理仿真模块
│   │   ├── body_properties.h   # 刚体物理属性（质量/弹性/摩擦）
│   │   ├── rigid_body.h/.cpp   # 刚体类（含缩放支持）
│   │   ├── contact.h           # 接触点数据结构
│   │   ├── collision_detector.h/.cpp  # 碰撞检测器
│   │   ├── force_builder.h     # 力的组装器
│   │   └── integrator.h/.cpp   # 时间积分器
│   │
│   ├── scene/                  # 场景管理模块
│   │   ├── environment.h/.cpp  # 环境边界
│   │   └── scene.h/.cpp        # 场景容器
│   │
│   ├── simulation/             # 仿真控制模块
│   │   └── simulator.h/.cpp    # 顶层仿真器
│   │
│   └── gpu/                    # CUDA GPU 模块
│       ├── bvh_builder.cu/.cuh         # GPU BVH 构建
│       ├── collision_detector.cu/.cuh  # GPU 碰撞检测
│       └── broadphase.cu/.cuh          # GPU 粗检测
│
├── simulation/                 # 运行输出目录
│   └── output/                 # 帧序列输出
└── render/                     # 渲染输出目录
    ├── simulation.mp4          # 渲染视频
    └── frames/                 # 渲染帧图片
```

### 2.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.cpp                                │
│                      (程序入口/场景初始化)                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    simulation/Simulator                          │
│               (仿真控制/帧循环/OBJ导出)                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌─────────────────────┐ ┌─────────────┐ ┌──────────────────┐
│    scene/Scene      │ │  physics/   │ │  core/MeshCache  │
│ (场景/刚体/环境管理) │ │  Integrator │ │   (网格缓存)      │
└──────────┬──────────┘ └──────┬──────┘ └──────────────────┘
           │                   │
           │       ┌───────────┴───────────┐
           │       ▼                       ▼
           │ ┌─────────────────┐  ┌─────────────────────────┐
           │ │ physics/        │  │ physics/                │
           │ │ ForceBuilder    │  │ CollisionDetector       │
           │ │ (力的组装)       │  │ (碰撞检测调度)           │
           │ └─────────────────┘  └────────────┬────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────┐            ┌───────────────────────────────┐
│  physics/RigidBody  │            │         gpu/ 模块              │
│    (刚体状态)        │            │  ┌─────────────────────────┐  │
└──────────┬──────────┘            │  │  gpu/BVHBuilderGPU      │  │
           │                       │  │  (Morton码/基数树构建)   │  │
           ▼                       │  └─────────────────────────┘  │
┌─────────────────────┐            │  ┌─────────────────────────┐  │
│    core/Mesh        │            │  │  gpu/CollisionDetector  │  │
│   (三角网格)         │◄───────────│  │       GPU              │  │
└──────────┬──────────┘            │  │  (精细碰撞检测)          │  │
           │                       │  └─────────────────────────┘  │
           ▼                       │  ┌─────────────────────────┐  │
┌─────────────────────┐            │  │  gpu/BroadphaseGPU      │  │
│    accel/BVH        │            │  │  (Sweep & Prune)        │  │
│  (层次包围盒树)      │◄───────────│  └─────────────────────────┘  │
└─────────────────────┘            └───────────────────────────────┘
```

### 2.3 模块职责说明

| 模块 | 主要职责 |
|------|----------|
| `core/` | 基础类型定义、数学工具、网格数据结构 |
| `accel/` | 加速数据结构（BVH） |
| `physics/` | 刚体物理、碰撞检测、力计算、时间积分 |
| `scene/` | 场景管理、环境边界 |
| `simulation/` | 顶层仿真控制、帧导出 |
| `gpu/` | CUDA 加速实现 |

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
[INFO] Body 0: scale=1.69, mass=1.44, restitution=0.87, friction=0.39
[INFO] Body 1: scale=1.80, mass=1.65, restitution=0.43, friction=0.60
[INFO] Body 2: scale=0.50, mass=1.03, restitution=0.33, friction=0.80
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
│   程序启动        │  解析命令行参数
└────────┬─────────┘
         ▼
┌──────────────────┐
│  初始化仿真器     │  Simulator::initialize()
│  - 设置积分器    │
│  - 创建场景边界  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  生成刚体对象     │  Simulator::addBody()
│  - 加载网格      │  MeshCache::acquire()
│  - 构建 GPU BVH  │  BVH::buildGPU()
│  - 随机物理属性  │  mass, scale, restitution, friction
│  - 设置初始状态  │  position, orientation, velocity
└────────┬─────────┘
         ▼
┌──────────────────────────────────────────┐
│              仿真主循环                    │
│  for frame in [0, kTotalFrames):         │
│  ┌─────────────────────────────────────┐ │
│  │  1. 导出当前帧                       │ │
│  │     export_frame() → OBJ 文件       │ │
│  ├─────────────────────────────────────┤ │
│  │  2. 更新世界包围盒                   │ │
│  │     RigidBody::world_bounds()       │ │
│  ├─────────────────────────────────────┤ │
│  │  3. 碰撞检测                        │ │
│  │  ┌───────────────────────────────┐  │ │
│  │  │ 3.1 刚体-环境碰撞 (GPU)        │  │ │
│  │  │     BVH 遍历 + 平面检测        │  │ │
│  │  ├───────────────────────────────┤  │ │
│  │  │ 3.2 粗检测 (GPU Broadphase)   │  │ │
│  │  │     Sweep & Prune 算法        │  │ │
│  │  ├───────────────────────────────┤  │ │
│  │  │ 3.3 精细检测 (GPU Narrowphase)│  │ │
│  │  │     顶点-三角形距离查询        │  │ │
│  │  └───────────────────────────────┘  │ │
│  ├─────────────────────────────────────┤ │
│  │  4. 力的组装                        │ │
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

### 5.1 BVH 构建（GPU 加速）

采用 **LBVH (Linear BVH)** 算法，基于 Morton 码的并行构建：

#### 5.1.1 Morton 码计算

```cpp
// gpu_bvh.cu
__device__ unsigned int compute_morton_3d(float x, float y, float z) {
    // 将 [0,1] 归一化坐标扩展为 10-bit 整数
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    
    // 位交织: xxxxx... yyyyy... zzzzz... → xyzxyzxyz...
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    unsigned int zz = expand_bits((unsigned int)z);
    
    return (xx << 2) | (yy << 1) | zz;
}
```

#### 5.1.2 基数树构建

使用并行基数树算法：

```
输入: 排序后的 Morton 码数组
输出: 完整的 BVH 树

1. 每个内部节点 i 处理 [0, n-2]
2. 计算方向 d = sign(δ(i, i+1) - δ(i, i-1))
3. 二分查找范围边界 j
4. 二分查找分割点 γ
5. 设置左右子节点连接
```

#### 5.1.3 包围盒计算（自底向上）

```cpp
__global__ void compute_node_bounds_kernel(...) {
    // 从叶节点开始
    int leaf_idx = n - 1 + idx;
    nodes[leaf_idx].bounds = prim_bounds[prim_indices[idx]];
    
    // 原子计数器确保等待两个子节点
    int current = parent_indices[leaf_idx];
    while (current >= 0) {
        int old = atomicAdd(&atomic_counters[current], 1);
        if (old == 0) return;  // 第一个到达的线程退出
        
        // 第二个线程合并包围盒
        nodes[current].bounds = merge(
            nodes[nodes[current].left].bounds,
            nodes[nodes[current].right].bounds
        );
        current = parent_indices[current];
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

```cpp
// gpu_broadphase.cu
__global__ void sweep_prune_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    AABBEndpoint ep = sorted_endpoints[idx];
    
    if (ep.is_max) return;  // 只处理 min 端点
    
    int body_a = ep.body_id;
    
    // 向后扫描直到遇到 body_a 的 max 端点
    for (int j = idx + 1; j < num_endpoints; ++j) {
        AABBEndpoint other = sorted_endpoints[j];
        
        if (other.body_id == body_a && other.is_max) break;
        if (other.is_max) continue;
        
        int body_b = other.body_id;
        
        // 检测三轴重叠
        if (overlap_3d(body_a, body_b)) {
            output_pair(body_a, body_b);
        }
    }
}
```

### 5.3 精细检测（Narrowphase）

#### 5.3.1 刚体-环境碰撞

对每个平面边界，使用 BVH 加速的顶点-平面距离检测：

```cpp
__global__ void detect_BVH_plane_collision_kernel(...) {
    // 每个线程处理一个叶节点
    int leaf_id = blockIdx.x * blockDim.x + threadIdx.x;
    int node_idx = num_triangles - 1 + leaf_id;
    
    // AABB-平面快速剔除
    if (!aabb_intersects_plane(node.bounds, plane)) return;
    
    // 检查三角形的三个顶点
    for (int k = 0; k < 3; ++k) {
        float dist = dot(vertex, plane.normal) - plane.offset;
        if (dist < 0) {
            // 记录接触点
            output_contact(vertex, plane.normal, -dist);
        }
    }
}
```

#### 5.3.2 刚体-刚体碰撞

采用顶点-三角形最近点查询：

```cpp
__global__ void detect_body_body_collision_kernel(...) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 将 B 的顶点变换到 A 的局部空间
    float3 v_world = rotate(v_local_b, orientation_b) + pos_b;
    float3 p_local_a = rotate_inv(v_world - pos_a, orientation_a);
    
    // BVH 栈式遍历
    int stack[64];
    while (stack_ptr > 0) {
        BVHNode node = nodes_a[stack[--stack_ptr]];
        
        // AABB 距离剔除
        if (dist_sq_to_aabb(p_local_a, node.bounds) > threshold) continue;
        
        if (node.is_leaf()) {
            // 计算到三角形的最近点
            float3 closest = closest_point_on_triangle(p_local_a, p0, p1, p2);
            update_min_distance(closest);
        } else {
            stack[stack_ptr++] = node.left;
            stack[stack_ptr++] = node.right;
        }
    }
}
```

### 5.4 物体属性随机化

每个刚体支持独立的物理属性，可在指定范围内随机生成：

| 属性 | 字段 | 范围 | 物理意义 |
|------|------|------|----------|
| 缩放 | `BodyState::scale` | 0.5 ~ 2.0 | 影响几何尺寸、包围盒、惯性张量 |
| 质量 | `BodyProperties::mass` | 0.5 ~ 3.0 kg | 影响惯性和碰撞响应 |
| 弹性 | `BodyProperties::restitution` | 0.2 ~ 0.9 | 0=完全非弹性, 1=完全弹性 |
| 摩擦 | `BodyProperties::friction` | 0.3 ~ 0.8 | 切向阻力系数 |
| 速度 | `BodyState::linearVel` | ±3.0 m/s | 初始线速度 |

### 5.5 碰撞响应

采用 **惩罚力法 (Penalty Method)** 结合隐式欧拉积分：

#### 5.5.1 惩罚力计算

```cpp
// integrator.cpp
void ForceAssembler::add_collision_force(...) {
    for (const auto& contact : contacts) {
        // 二次惩罚力: F = k * d² * n
        Vec3 penalty = k * contact.depth * contact.depth * contact.normal;
        
        // 作用于穿透体 B（推出方向）
        force[body_b] += penalty;
        torque[body_b] += r_b.cross(penalty);
        
        // 反作用于障碍体 A
        force[body_a] -= penalty;
        torque[body_a] += r_a.cross(-penalty);
    }
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
```bash
(Collision) (base) root@xvlin:~/Collision# bash scripts/run_benchmark.sh --full
╔════════════════════════════════════════════════════════════════╗
║       Rigid Body Simulator - Benchmark Suite                    ║
╚════════════════════════════════════════════════════════════════╝

[MESH INFO] Available meshes:
  ball.obj: 2050 vertices, 4096 faces, 272K
  diamond.obj: 65 vertices, 65 faces, 8.0K

[BUILD] Configuring and building...
[100%] Built target CollisionSimulator
[BUILD] Success!

[BENCH] Running FULL benchmark suite...
[INFO] Project root: /root/Collision
[INFO] Using executable: /root/Collision/build/CollisionSimulator

================================================================================
MESH INFORMATION
================================================================================
Mesh                        Vertices      Faces    Size (KB)
--------------------------------------------------------------------------------
ball.obj                        2050       4096        271.4
diamond.obj                       65         65          5.4
================================================================================

[INFO] Running 40 benchmark configurations...
[INFO] Runs per configuration: 1
[INFO] Timeout per run: 1800s

[1/40] ball_n10_f50 (10 objs × 4096 faces = 40960 triangles)
  → Total: 169.25ms, Avg frame: 3.116ms, FPS: 320.96

[2/40] ball_n10_f100 (10 objs × 4096 faces = 40960 triangles)
  → Total: 284.47ms, Avg frame: 2.726ms, FPS: 366.90

[3/40] ball_n10_f200 (10 objs × 4096 faces = 40960 triangles)
  → Total: 551.79ms, Avg frame: 2.693ms, FPS: 371.36

[4/40] ball_n10_f500 (10 objs × 4096 faces = 40960 triangles)
  → Total: 1158.03ms, Avg frame: 2.291ms, FPS: 436.51

[5/40] ball_n20_f50 (20 objs × 4096 faces = 81920 triangles)
  → Total: 352.88ms, Avg frame: 6.833ms, FPS: 146.35

[6/40] ball_n20_f100 (20 objs × 4096 faces = 81920 triangles)
  → Total: 634.58ms, Avg frame: 6.222ms, FPS: 160.71

[7/40] ball_n20_f200 (20 objs × 4096 faces = 81920 triangles)
  → Total: 1134.52ms, Avg frame: 5.613ms, FPS: 178.15

[8/40] ball_n20_f500 (20 objs × 4096 faces = 81920 triangles)
  → Total: 2551.23ms, Avg frame: 5.081ms, FPS: 196.81

[9/40] ball_n50_f50 (50 objs × 4096 faces = 204800 triangles)
  → Total: 1299.87ms, Avg frame: 25.772ms, FPS: 38.80

[10/40] ball_n50_f100 (50 objs × 4096 faces = 204800 triangles)
  → Total: 1970.82ms, Avg frame: 19.606ms, FPS: 51.01

[11/40] ball_n50_f200 (50 objs × 4096 faces = 204800 triangles)
  → Total: 3559.75ms, Avg frame: 17.740ms, FPS: 56.37

[12/40] ball_n50_f500 (50 objs × 4096 faces = 204800 triangles)
  → Total: 7537.14ms, Avg frame: 15.054ms, FPS: 66.43

[13/40] ball_n100_f50 (100 objs × 4096 faces = 409600 triangles)
  → Total: 4262.86ms, Avg frame: 85.007ms, FPS: 11.76

[14/40] ball_n100_f100 (100 objs × 4096 faces = 409600 triangles)
  → Total: 6932.32ms, Avg frame: 69.219ms, FPS: 14.45

[15/40] ball_n100_f200 (100 objs × 4096 faces = 409600 triangles)
  → Total: 12073.15ms, Avg frame: 60.314ms, FPS: 16.58

[16/40] ball_n100_f500 (100 objs × 4096 faces = 409600 triangles)
  → Total: 26910.89ms, Avg frame: 53.800ms, FPS: 18.59

[17/40] ball_n150_f50 (150 objs × 4096 faces = 614400 triangles)
  → Total: 9883.80ms, Avg frame: 197.464ms, FPS: 5.06

[18/40] ball_n150_f100 (150 objs × 4096 faces = 614400 triangles)
  → Total: 14925.76ms, Avg frame: 149.139ms, FPS: 6.71

[19/40] ball_n150_f200 (150 objs × 4096 faces = 614400 triangles)
  → Total: 24188.53ms, Avg frame: 120.892ms, FPS: 8.27

[20/40] ball_n150_f500 (150 objs × 4096 faces = 614400 triangles)
  → Total: 52295.47ms, Avg frame: 104.571ms, FPS: 9.56

[21/40] diamond_n10_f50 (10 objs × 65 faces = 650 triangles)
  → Total: 118.46ms, Avg frame: 2.304ms, FPS: 434.06

[22/40] diamond_n10_f100 (10 objs × 65 faces = 650 triangles)
  → Total: 166.79ms, Avg frame: 1.640ms, FPS: 609.71

[23/40] diamond_n10_f200 (10 objs × 65 faces = 650 triangles)
  → Total: 355.56ms, Avg frame: 1.762ms, FPS: 567.53

[24/40] diamond_n10_f500 (10 objs × 65 faces = 650 triangles)
  → Total: 901.79ms, Avg frame: 1.793ms, FPS: 557.79

[25/40] diamond_n20_f50 (20 objs × 65 faces = 1300 triangles)
  → Total: 166.97ms, Avg frame: 3.277ms, FPS: 305.18

[26/40] diamond_n20_f100 (20 objs × 65 faces = 1300 triangles)
  → Total: 375.13ms, Avg frame: 3.718ms, FPS: 268.94

[27/40] diamond_n20_f200 (20 objs × 65 faces = 1300 triangles)
  → Total: 753.92ms, Avg frame: 3.748ms, FPS: 266.79

[28/40] diamond_n20_f500 (20 objs × 65 faces = 1300 triangles)
  → Total: 1713.88ms, Avg frame: 3.408ms, FPS: 293.45

[29/40] diamond_n50_f50 (50 objs × 65 faces = 3250 triangles)
  → Total: 436.40ms, Avg frame: 8.657ms, FPS: 115.51

[30/40] diamond_n50_f100 (50 objs × 65 faces = 3250 triangles)
  → Total: 880.52ms, Avg frame: 8.777ms, FPS: 113.93

[31/40] diamond_n50_f200 (50 objs × 65 faces = 3250 triangles)
  → Total: 1854.52ms, Avg frame: 9.243ms, FPS: 108.19

[32/40] diamond_n50_f500 (50 objs × 65 faces = 3250 triangles)
  → Total: 4608.89ms, Avg frame: 9.212ms, FPS: 108.55

[33/40] diamond_n100_f50 (100 objs × 65 faces = 6500 triangles)
  → Total: 1148.87ms, Avg frame: 22.918ms, FPS: 43.63

[34/40] diamond_n100_f100 (100 objs × 65 faces = 6500 triangles)
  → Total: 2086.84ms, Avg frame: 20.839ms, FPS: 47.99

[35/40] diamond_n100_f200 (100 objs × 65 faces = 6500 triangles)
  → Total: 3910.19ms, Avg frame: 19.532ms, FPS: 51.20

[36/40] diamond_n100_f500 (100 objs × 65 faces = 6500 triangles)
  → Total: 10189.05ms, Avg frame: 20.372ms, FPS: 49.09

[37/40] diamond_n150_f50 (150 objs × 65 faces = 9750 triangles)
  → Total: 1866.08ms, Avg frame: 37.262ms, FPS: 26.84

[38/40] diamond_n150_f100 (150 objs × 65 faces = 9750 triangles)
  → Total: 3415.41ms, Avg frame: 34.122ms, FPS: 29.31

[39/40] diamond_n150_f200 (150 objs × 65 faces = 9750 triangles)
  → Total: 6363.81ms, Avg frame: 31.790ms, FPS: 31.46

[40/40] diamond_n150_f500 (150 objs × 65 faces = 9750 triangles)
  → Total: 17339.65ms, Avg frame: 34.672ms, FPS: 28.84

========================================================================================================================
BENCHMARK RESULTS
========================================================================================================================
Name                           Objs       Tris  Frames   Total(ms)    Avg(ms)      FPS   Status
------------------------------------------------------------------------------------------------------------------------
ball_n10_f50                     10      40960      50      169.25      3.116   320.96        ✓
ball_n10_f100                    10      40960     100      284.47      2.726   366.90        ✓
ball_n10_f200                    10      40960     200      551.79      2.693   371.36        ✓
ball_n10_f500                    10      40960     500     1158.03      2.291   436.51        ✓
ball_n20_f50                     20      81920      50      352.88      6.833   146.35        ✓
ball_n20_f100                    20      81920     100      634.58      6.222   160.71        ✓
ball_n20_f200                    20      81920     200     1134.52      5.613   178.15        ✓
ball_n20_f500                    20      81920     500     2551.23      5.081   196.81        ✓
ball_n50_f50                     50     204800      50     1299.87     25.772    38.80        ✓
ball_n50_f100                    50     204800     100     1970.82     19.606    51.01        ✓
ball_n50_f200                    50     204800     200     3559.75     17.740    56.37        ✓
ball_n50_f500                    50     204800     500     7537.14     15.054    66.43        ✓
ball_n100_f50                   100     409600      50     4262.86     85.007    11.76        ✓
ball_n100_f100                  100     409600     100     6932.32     69.219    14.45        ✓
ball_n100_f200                  100     409600     200    12073.15     60.314    16.58        ✓
ball_n100_f500                  100     409600     500    26910.89     53.800    18.59        ✓
ball_n150_f50                   150     614400      50     9883.80    197.464     5.06        ✓
ball_n150_f100                  150     614400     100    14925.76    149.139     6.71        ✓
ball_n150_f200                  150     614400     200    24188.53    120.892     8.27        ✓
ball_n150_f500                  150     614400     500    52295.47    104.571     9.56        ✓
diamond_n10_f50                  10        650      50      118.46      2.304   434.06        ✓
diamond_n10_f100                 10        650     100      166.79      1.640   609.71        ✓
diamond_n10_f200                 10        650     200      355.56      1.762   567.53        ✓
diamond_n10_f500                 10        650     500      901.79      1.793   557.79        ✓
diamond_n20_f50                  20       1300      50      166.97      3.277   305.18        ✓
diamond_n20_f100                 20       1300     100      375.13      3.718   268.94        ✓
diamond_n20_f200                 20       1300     200      753.92      3.748   266.79        ✓
diamond_n20_f500                 20       1300     500     1713.88      3.408   293.45        ✓
diamond_n50_f50                  50       3250      50      436.40      8.657   115.51        ✓
diamond_n50_f100                 50       3250     100      880.52      8.777   113.93        ✓
diamond_n50_f200                 50       3250     200     1854.52      9.243   108.19        ✓
diamond_n50_f500                 50       3250     500     4608.89      9.212   108.55        ✓
diamond_n100_f50                100       6500      50     1148.87     22.918    43.63        ✓
diamond_n100_f100               100       6500     100     2086.84     20.839    47.99        ✓
diamond_n100_f200               100       6500     200     3910.19     19.532    51.20        ✓
diamond_n100_f500               100       6500     500    10189.05     20.372    49.09        ✓
diamond_n150_f50                150       9750      50     1866.08     37.262    26.84        ✓
diamond_n150_f100               150       9750     100     3415.41     34.122    29.31        ✓
diamond_n150_f200               150       9750     200     6363.81     31.790    31.46        ✓
diamond_n150_f500               150       9750     500    17339.65     34.672    28.84        ✓
========================================================================================================================

Statistics (40/40 successful):
  FPS range:          5.06 - 609.71
  Average FPS:        163.48
  Average frame time: 30.805 ms

[INFO] Results saved to: benchmark_full_20260110_223117.csv

[DONE] Benchmark completed!
```

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

