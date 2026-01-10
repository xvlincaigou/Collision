#!/usr/bin/env python3
"""
Benchmark script for the Rigid Body Simulator.

This script runs the simulator with various configurations and records
the execution times to a CSV file, including mesh information.
"""

import subprocess
import sys
import os
import csv
import re
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MeshInfo:
    """Information about an OBJ mesh file."""
    path: str
    file_size_bytes: int
    vertex_count: int
    face_count: int
    
    @property
    def file_size_kb(self) -> float:
        return self.file_size_bytes / 1024
    
    def __str__(self) -> str:
        return f"{Path(self.path).name} ({self.vertex_count} verts, {self.face_count} faces, {self.file_size_kb:.1f}KB)"


def parse_obj_file(path: Path) -> MeshInfo:
    """Parse OBJ file to get vertex and face counts."""
    vertex_count = 0
    face_count = 0
    file_size = path.stat().st_size if path.exists() else 0
    
    if path.exists():
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        vertex_count += 1
                    elif line.startswith('f '):
                        face_count += 1
        except Exception as e:
            print(f"[WARN] Could not parse {path}: {e}")
    
    return MeshInfo(
        path=str(path),
        file_size_bytes=file_size,
        vertex_count=vertex_count,
        face_count=face_count
    )


@dataclass
class BenchmarkConfig:
    """Single benchmark configuration."""
    mesh: str
    num_objects: int
    num_frames: int
    mesh_info: Optional[MeshInfo] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            mesh_name = Path(self.mesh).stem
            self.name = f"{mesh_name}_n{self.num_objects}_f{self.num_frames}"
    
    @property
    def total_triangles(self) -> int:
        """Total triangles in scene = faces * objects."""
        if self.mesh_info:
            return self.mesh_info.face_count * self.num_objects
        return 0
    
    @property
    def total_vertices(self) -> int:
        """Total vertices in scene = vertices * objects."""
        if self.mesh_info:
            return self.mesh_info.vertex_count * self.num_objects
        return 0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    config: BenchmarkConfig
    setup_time_ms: float
    sim_time_ms: float
    total_time_ms: float
    avg_frame_time_ms: float
    fps: float
    success: bool
    error_msg: str = ""


def find_executable() -> Optional[Path]:
    """Find the simulator executable."""
    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent
    
    possible_paths = [
        root_dir / "build" / "CollisionSimulator",
        root_dir / "simulation" / "CollisionSimulator",
        root_dir / "CollisionSimulator",
    ]
    
    for path in possible_paths:
        if path.exists() and os.access(path, os.X_OK):
            return path
    
    return None


def run_single_benchmark(
    executable: Path,
    config: BenchmarkConfig,
    root_dir: Path,
    timeout: int = 600
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    cmd = [
        str(executable),
        "--benchmark",
        "--mesh", config.mesh,
        "--num-objects", str(config.num_objects),
        "--frames", str(config.num_frames),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root_dir  # Run from project root so asset paths work
        )
        
        if result.returncode != 0:
            return BenchmarkResult(
                config=config,
                setup_time_ms=0,
                sim_time_ms=0,
                total_time_ms=0,
                avg_frame_time_ms=0,
                fps=0,
                success=False,
                error_msg=result.stderr.strip() or f"Exit code: {result.returncode}"
            )
        
        # Parse CSV output: mesh,objects,frames,setup_ms,sim_ms,total_ms,avg_frame_ms,fps
        line = result.stdout.strip()
        parts = line.split(",")
        
        if len(parts) < 8:
            return BenchmarkResult(
                config=config,
                setup_time_ms=0,
                sim_time_ms=0,
                total_time_ms=0,
                avg_frame_time_ms=0,
                fps=0,
                success=False,
                error_msg=f"Invalid output: {line[:100]}"
            )
        
        return BenchmarkResult(
            config=config,
            setup_time_ms=float(parts[3]),
            sim_time_ms=float(parts[4]),
            total_time_ms=float(parts[5]),
            avg_frame_time_ms=float(parts[6]),
            fps=float(parts[7]),
            success=True
        )
        
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            config=config,
            setup_time_ms=0,
            sim_time_ms=0,
            total_time_ms=0,
            avg_frame_time_ms=0,
            fps=0,
            success=False,
            error_msg=f"Timeout after {timeout}s"
        )
    except Exception as e:
        return BenchmarkResult(
            config=config,
            setup_time_ms=0,
            sim_time_ms=0,
            total_time_ms=0,
            avg_frame_time_ms=0,
            fps=0,
            success=False,
            error_msg=str(e)
        )


def generate_default_configs(root_dir: Path) -> List[BenchmarkConfig]:
    """Generate default benchmark configurations."""
    configs = []
    
    # Parse mesh info
    meshes = {
        "assets/ball.obj": parse_obj_file(root_dir / "assets/ball.obj"),
        "assets/diamond.obj": parse_obj_file(root_dir / "assets/diamond.obj"),
    }
    
    # Default: ball.obj with varying objects
    mesh = "assets/ball.obj"
    mesh_info = meshes.get(mesh)
    for n in [5, 10, 20, 50, 100]:
        configs.append(BenchmarkConfig(mesh=mesh, num_objects=n, num_frames=100, mesh_info=mesh_info))
    
    # Vary frames with fixed objects
    for f in [50, 100, 200, 500]:
        configs.append(BenchmarkConfig(mesh=mesh, num_objects=20, num_frames=f, mesh_info=mesh_info))
    
    return configs


def generate_stress_configs(root_dir: Path) -> List[BenchmarkConfig]:
    """Generate stress test configurations (heavier load)."""
    configs = []
    
    # Parse mesh info
    meshes = {
        "assets/ball.obj": parse_obj_file(root_dir / "assets/ball.obj"),
        "assets/diamond.obj": parse_obj_file(root_dir / "assets/diamond.obj"),
    }
    
    # Heavy ball.obj tests
    mesh = "assets/ball.obj"
    mesh_info = meshes.get(mesh)
    
    # Stress test: many objects
    for n in [50, 100, 150, 200]:
        configs.append(BenchmarkConfig(mesh=mesh, num_objects=n, num_frames=100, mesh_info=mesh_info))
    
    # Stress test: many frames
    for f in [200, 500, 1000]:
        configs.append(BenchmarkConfig(mesh=mesh, num_objects=50, num_frames=f, mesh_info=mesh_info))
    
    # Combined stress
    configs.append(BenchmarkConfig(mesh=mesh, num_objects=100, num_frames=500, mesh_info=mesh_info))
    configs.append(BenchmarkConfig(mesh=mesh, num_objects=200, num_frames=200, mesh_info=mesh_info))
    
    return configs


def save_results_csv(results: List[BenchmarkResult], output_path: Path, mesh_infos: dict):
    """Save benchmark results to CSV file with mesh info."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header with mesh info
        writer.writerow([
            'name', 'mesh', 'mesh_verts', 'mesh_faces', 'mesh_size_kb',
            'objects', 'total_verts', 'total_faces', 'frames',
            'setup_ms', 'sim_ms', 'total_ms', 'avg_frame_ms', 'fps',
            'success', 'error'
        ])
        
        # Data rows
        for r in results:
            cfg = r.config
            mi = cfg.mesh_info
            writer.writerow([
                cfg.name,
                cfg.mesh,
                mi.vertex_count if mi else 0,
                mi.face_count if mi else 0,
                f"{mi.file_size_kb:.1f}" if mi else 0,
                cfg.num_objects,
                cfg.total_vertices,
                cfg.total_triangles,
                cfg.num_frames,
                f"{r.setup_time_ms:.3f}",
                f"{r.sim_time_ms:.3f}",
                f"{r.total_time_ms:.3f}",
                f"{r.avg_frame_time_ms:.3f}",
                f"{r.fps:.2f}",
                r.success,
                r.error_msg
            ])


def print_mesh_info(mesh_infos: dict):
    """Print information about mesh files."""
    print("\n" + "=" * 80)
    print("MESH INFORMATION")
    print("=" * 80)
    print(f"{'Mesh':<25} {'Vertices':>10} {'Faces':>10} {'Size (KB)':>12}")
    print("-" * 80)
    for path, info in mesh_infos.items():
        print(f"{Path(path).name:<25} {info.vertex_count:>10} {info.face_count:>10} {info.file_size_kb:>12.1f}")
    print("=" * 80)


def print_summary_table(results: List[BenchmarkResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    
    # Header
    print(f"{'Name':<28} {'Objs':>6} {'Tris':>10} {'Frames':>7} "
          f"{'Total(ms)':>11} {'Avg(ms)':>10} {'FPS':>8} {'Status':>8}")
    print("-" * 120)
    
    # Results
    for r in results:
        status = "✓" if r.success else "✗"
        total_tris = r.config.total_triangles
        if r.success:
            print(f"{r.config.name:<28} {r.config.num_objects:>6} {total_tris:>10} "
                  f"{r.config.num_frames:>7} {r.total_time_ms:>11.2f} "
                  f"{r.avg_frame_time_ms:>10.3f} {r.fps:>8.2f} {status:>8}")
        else:
            print(f"{r.config.name:<28} {r.config.num_objects:>6} {total_tris:>10} "
                  f"{r.config.num_frames:>7} {'N/A':>11} {'N/A':>10} {'N/A':>8} {status:>8}")
            print(f"    Error: {r.error_msg}")
    
    print("=" * 120)
    
    # Statistics
    successful = [r for r in results if r.success]
    if successful:
        avg_fps = sum(r.fps for r in successful) / len(successful)
        avg_frame_time = sum(r.avg_frame_time_ms for r in successful) / len(successful)
        min_fps = min(r.fps for r in successful)
        max_fps = max(r.fps for r in successful)
        
        print(f"\nStatistics ({len(successful)}/{len(results)} successful):")
        print(f"  FPS range:          {min_fps:.2f} - {max_fps:.2f}")
        print(f"  Average FPS:        {avg_fps:.2f}")
        print(f"  Average frame time: {avg_frame_time:.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Rigid Body Simulator with various configurations."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path'
    )
    parser.add_argument(
        '-m', '--mesh',
        type=str,
        action='append',
        help='Mesh file(s) to test'
    )
    parser.add_argument(
        '-n', '--num-objects',
        type=int,
        action='append',
        help='Number of objects to test'
    )
    parser.add_argument(
        '-f', '--frames',
        type=int,
        action='append',
        help='Number of frames to test'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,  # 30 minutes for stress tests
        help='Timeout per benchmark in seconds (default: 1800)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of runs per configuration (default: 1)'
    )
    parser.add_argument(
        '--build',
        action='store_true',
        help='Build the project before running benchmarks'
    )
    parser.add_argument(
        '--stress',
        action='store_true',
        help='Run stress test configurations (heavier load)'
    )
    
    args = parser.parse_args()
    
    # Get project root directory
    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent
    os.chdir(root_dir)
    
    print(f"[INFO] Project root: {root_dir}")
    
    # Build if requested
    if args.build:
        print("[BUILD] Building project...")
        
        # Configure
        result = subprocess.run(
            ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
            capture_output=True,
            text=True,
            cwd=root_dir
        )
        if result.returncode != 0:
            print("[BUILD] Configure FAILED!")
            print(result.stderr)
            sys.exit(1)
        
        # Build
        result = subprocess.run(
            ["cmake", "--build", "build", "--parallel"],
            capture_output=True,
            text=True,
            cwd=root_dir
        )
        if result.returncode != 0:
            print("[BUILD] Build FAILED!")
            print(result.stderr)
            sys.exit(1)
        print("[BUILD] Success!")
    
    # Find executable
    executable = find_executable()
    if executable is None:
        print("[ERROR] Could not find CollisionSimulator executable.")
        print("  Run with --build flag or build manually first.")
        sys.exit(1)
    
    print(f"[INFO] Using executable: {executable}")
    
    # Parse available meshes
    mesh_infos = {}
    assets_dir = root_dir / "assets"
    if assets_dir.exists():
        for obj_file in assets_dir.glob("*.obj"):
            rel_path = f"assets/{obj_file.name}"
            mesh_infos[rel_path] = parse_obj_file(obj_file)
    
    print_mesh_info(mesh_infos)
    
    # Generate configurations
    if args.mesh or args.num_objects or args.frames:
        # Custom configurations from command line
        meshes = args.mesh or ["assets/ball.obj"]
        objects = args.num_objects or [10]
        frames = args.frames or [100]
        
        configs = []
        for mesh in meshes:
            mesh_info = mesh_infos.get(mesh) or parse_obj_file(root_dir / mesh)
            for n in objects:
                for f in frames:
                    configs.append(BenchmarkConfig(
                        mesh=mesh, num_objects=n, num_frames=f, mesh_info=mesh_info
                    ))
    elif args.stress:
        configs = generate_stress_configs(root_dir)
    else:
        configs = generate_default_configs(root_dir)
    
    print(f"\n[INFO] Running {len(configs)} benchmark configurations...")
    print(f"[INFO] Runs per configuration: {args.runs}")
    print(f"[INFO] Timeout per run: {args.timeout}s")
    
    # Run benchmarks
    all_results = []
    for i, config in enumerate(configs, 1):
        total_tris = config.total_triangles
        print(f"\n[{i}/{len(configs)}] {config.name} "
              f"({config.num_objects} objs × {config.mesh_info.face_count if config.mesh_info else '?'} faces = {total_tris} triangles)")
        
        run_results = []
        for run in range(args.runs):
            if args.runs > 1:
                print(f"  Run {run + 1}/{args.runs}...", end=" ", flush=True)
            
            result = run_single_benchmark(executable, config, root_dir, args.timeout)
            run_results.append(result)
            
            if args.runs > 1:
                if result.success:
                    print(f"{result.total_time_ms:.2f}ms ({result.fps:.1f} FPS)")
                else:
                    print(f"FAILED: {result.error_msg}")
        
        # Average results if multiple runs
        if args.runs > 1:
            successful_runs = [r for r in run_results if r.success]
            if successful_runs:
                avg_result = BenchmarkResult(
                    config=config,
                    setup_time_ms=sum(r.setup_time_ms for r in successful_runs) / len(successful_runs),
                    sim_time_ms=sum(r.sim_time_ms for r in successful_runs) / len(successful_runs),
                    total_time_ms=sum(r.total_time_ms for r in successful_runs) / len(successful_runs),
                    avg_frame_time_ms=sum(r.avg_frame_time_ms for r in successful_runs) / len(successful_runs),
                    fps=sum(r.fps for r in successful_runs) / len(successful_runs),
                    success=True
                )
                all_results.append(avg_result)
            else:
                all_results.append(run_results[0])
        else:
            all_results.append(run_results[0])
        
        if all_results[-1].success:
            r = all_results[-1]
            print(f"  → Total: {r.total_time_ms:.2f}ms, "
                  f"Avg frame: {r.avg_frame_time_ms:.3f}ms, "
                  f"FPS: {r.fps:.2f}")
        else:
            print(f"  → FAILED: {all_results[-1].error_msg}")
    
    # Print summary
    print_summary_table(all_results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "stress" if args.stress else "default"
        output_path = Path(f"benchmark_{mode}_{timestamp}.csv")
    
    save_results_csv(all_results, output_path, mesh_infos)
    print(f"\n[INFO] Results saved to: {output_path}")
    
    # Return non-zero if any benchmark failed
    if any(not r.success for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
