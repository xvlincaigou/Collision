#!/usr/bin/env python3
"""
Lightweight OBJ Sequence Renderer using PyVista

This script renders OBJ sequences to images/video without needing Blender.
It's faster but produces simpler visualizations.

Usage:
    python render_pyvista.py --input output/ --output render/
    python render_pyvista.py --input output/ --output render/simulation.mp4 --video

Requirements:
    pip install pyvista imageio imageio-ffmpeg tqdm numpy
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    import pyvista as pv
    from pyvista import themes
except ImportError:
    print("Please install pyvista: pip install pyvista")
    sys.exit(1)

try:
    import imageio
except ImportError:
    print("Please install imageio: pip install imageio imageio-ffmpeg")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_WINDOW_SIZE = (1920, 1080)
DEFAULT_CAMERA_POSITION = [(30, -30, 25), (0, 0, 5), (0, 0, 1)]
DEFAULT_BACKGROUND = "#1a1a2e"
DEFAULT_OBJECT_COLOR = "#e94560"


# ============================================================================
# OBJ Loading
# ============================================================================

def load_obj_simple(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple OBJ loader that returns vertices and faces.
    Returns (vertices, faces) as numpy arrays.
    """
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Parse face indices (handles v, v/vt, v/vt/vn, v//vn formats)
                face_indices = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    face_indices.append(idx)
                
                # Triangulate if needed (fan triangulation)
                for i in range(1, len(face_indices) - 1):
                    faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
    
    return np.array(vertices), np.array(faces)


def obj_to_pyvista(filepath: str) -> pv.PolyData:
    """Load OBJ file and convert to PyVista mesh."""
    vertices, faces = load_obj_simple(filepath)
    
    # PyVista faces format: [n_points, idx0, idx1, ..., n_points, idx0, ...]
    pv_faces = []
    for face in faces:
        pv_faces.extend([3] + list(face))
    
    mesh = pv.PolyData(vertices, pv_faces)
    return mesh


def get_obj_files(input_dir: str) -> List[str]:
    """Get sorted list of OBJ files."""
    pattern = os.path.join(input_dir, "frame_*.obj")
    files = sorted(glob.glob(pattern))
    return files


# ============================================================================
# Rendering
# ============================================================================

def setup_plotter(
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    background: str = DEFAULT_BACKGROUND,
    off_screen: bool = True
) -> pv.Plotter:
    """Create and configure a PyVista plotter."""
    
    # Use off-screen rendering
    plotter = pv.Plotter(
        window_size=window_size,
        off_screen=off_screen
    )
    
    # Set background
    plotter.set_background(background)
    
    # Add lighting
    plotter.add_light(pv.Light(
        position=(20, -20, 30),
        focal_point=(0, 0, 0),
        color='white',
        intensity=0.8
    ))
    plotter.add_light(pv.Light(
        position=(-15, 10, 20),
        color='white',
        intensity=0.4
    ))
    
    return plotter


def add_ground_plane(plotter: pv.Plotter, size: float = 30.0):
    """Add a ground plane to the scene."""
    plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=size,
        j_size=size,
        i_resolution=10,
        j_resolution=10
    )
    plotter.add_mesh(
        plane,
        color='#2d3436',
        opacity=0.8,
        show_edges=True,
        edge_color='#636e72',
        line_width=0.5
    )


def add_boundary_box(plotter: pv.Plotter, bounds: Tuple[float, ...]):
    """Add a wireframe boundary box."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    box = pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
    plotter.add_mesh(
        box,
        style='wireframe',
        color='#dfe6e9',
        line_width=1.5,
        opacity=0.3
    )


def render_frame(
    plotter: pv.Plotter,
    obj_file: str,
    output_path: str,
    object_color: str = DEFAULT_OBJECT_COLOR,
    camera_position: Optional[List] = None,
    show_edges: bool = False
) -> np.ndarray:
    """Render a single frame and return the image."""
    
    # Clear previous meshes (keep lights and plane)
    plotter.clear_actors()
    
    # Add ground
    add_ground_plane(plotter)
    
    # Add boundary box
    add_boundary_box(plotter, (-10, 10, -10, 10, 0, 30))
    
    # Load and add mesh
    mesh = obj_to_pyvista(obj_file)
    
    plotter.add_mesh(
        mesh,
        color=object_color,
        specular=0.5,
        specular_power=30,
        smooth_shading=True,
        show_edges=show_edges,
        edge_color='#2d3436',
        line_width=0.5
    )
    
    # Set camera
    if camera_position:
        plotter.camera_position = camera_position
    
    # Capture frame
    plotter.screenshot(output_path)
    
    return plotter.screenshot(return_img=True)


def render_sequence(
    obj_files: List[str],
    output_dir: str,
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    object_color: str = DEFAULT_OBJECT_COLOR,
    camera_position: Optional[List] = None,
    show_progress: bool = True
) -> List[str]:
    """Render all frames in the sequence."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = setup_plotter(window_size=window_size)
    
    if camera_position is None:
        camera_position = DEFAULT_CAMERA_POSITION
    
    output_files = []
    iterator = tqdm(enumerate(obj_files), total=len(obj_files), desc="Rendering")
    
    for i, obj_file in iterator:
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        render_frame(plotter, obj_file, output_path, object_color, camera_position)
        output_files.append(output_path)
    
    plotter.close()
    
    return output_files


def create_video(
    image_files: List[str],
    output_path: str,
    fps: int = 30
):
    """Create video from rendered images."""
    
    print(f"Creating video: {output_path}")
    
    # Read first image to get dimensions
    first_image = imageio.imread(image_files[0])
    
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    )
    
    for img_path in tqdm(image_files, desc="Encoding video"):
        img = imageio.imread(img_path)
        writer.append_data(img)
    
    writer.close()
    print(f"Video saved: {output_path}")


def create_gif(
    image_files: List[str],
    output_path: str,
    fps: int = 15,
    loop: int = 0
):
    """Create GIF from rendered images."""
    
    print(f"Creating GIF: {output_path}")
    
    images = []
    for img_path in tqdm(image_files[::2], desc="Loading images"):  # Skip every other frame
        images.append(imageio.imread(img_path))
    
    imageio.mimsave(output_path, images, fps=fps, loop=loop)
    print(f"GIF saved: {output_path}")


# ============================================================================
# Interactive Viewer
# ============================================================================

def interactive_view(obj_files: List[str], start_frame: int = 0):
    """Open an interactive viewer for the sequence."""
    
    print("Interactive viewer controls:")
    print("  Space: Play/Pause")
    print("  Left/Right: Previous/Next frame")
    print("  R: Reset camera")
    print("  Q: Quit")
    
    plotter = pv.Plotter()
    plotter.set_background(DEFAULT_BACKGROUND)
    add_ground_plane(plotter)
    
    current_frame = [start_frame]
    mesh_actor = [None]
    playing = [False]
    
    def update_frame(frame_idx):
        if mesh_actor[0] is not None:
            plotter.remove_actor(mesh_actor[0])
        
        mesh = obj_to_pyvista(obj_files[frame_idx])
        mesh_actor[0] = plotter.add_mesh(
            mesh,
            color=DEFAULT_OBJECT_COLOR,
            specular=0.5,
            smooth_shading=True
        )
        plotter.add_text(
            f"Frame: {frame_idx + 1}/{len(obj_files)}",
            position='upper_left',
            font_size=12,
            name='frame_text'
        )
    
    def next_frame():
        current_frame[0] = (current_frame[0] + 1) % len(obj_files)
        update_frame(current_frame[0])
    
    def prev_frame():
        current_frame[0] = (current_frame[0] - 1) % len(obj_files)
        update_frame(current_frame[0])
    
    def toggle_play():
        playing[0] = not playing[0]
    
    plotter.add_key_event('Right', next_frame)
    plotter.add_key_event('Left', prev_frame)
    plotter.add_key_event('space', toggle_play)
    
    update_frame(current_frame[0])
    plotter.camera_position = DEFAULT_CAMERA_POSITION
    
    # Animation callback
    def callback(step):
        if playing[0]:
            next_frame()
    
    plotter.add_callback(callback, interval=33)  # ~30 FPS
    
    plotter.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render OBJ sequence to images/video using PyVista"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing frame_XXXX.obj files"
    )
    parser.add_argument(
        "-o", "--output",
        default="render",
        help="Output directory or video file path"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Create video output (MP4)"
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Create GIF output"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive viewer instead of rendering"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Output width (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Output height (default: 1080)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS (default: 30)"
    )
    parser.add_argument(
        "--color",
        default=DEFAULT_OBJECT_COLOR,
        help="Object color in hex (default: #e94560)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End frame (-1 for all)"
    )
    
    args = parser.parse_args()
    
    # Find OBJ files
    obj_files = get_obj_files(args.input)
    if not obj_files:
        print(f"No OBJ files found in: {args.input}")
        sys.exit(1)
    
    print(f"Found {len(obj_files)} OBJ files")
    
    # Slice if needed
    end = args.end if args.end > 0 else len(obj_files)
    obj_files = obj_files[args.start:end]
    print(f"Processing frames {args.start} to {end-1}")
    
    # Interactive mode
    if args.interactive:
        interactive_view(obj_files)
        return
    
    # Determine output paths
    output_path = Path(args.output)
    if output_path.suffix in ['.mp4', '.gif']:
        video_output = str(output_path)
        image_dir = str(output_path.parent / "frames")
        create_video_flag = output_path.suffix == '.mp4'
        create_gif_flag = output_path.suffix == '.gif'
    else:
        image_dir = str(output_path / "frames")
        video_output = str(output_path / "simulation.mp4")
        create_video_flag = args.video
        create_gif_flag = args.gif
    
    # Render sequence
    print(f"Rendering to: {image_dir}")
    image_files = render_sequence(
        obj_files,
        image_dir,
        window_size=(args.width, args.height),
        object_color=args.color
    )
    
    # Create video if requested
    if create_video_flag:
        create_video(image_files, video_output, fps=args.fps)
    
    # Create GIF if requested
    if create_gif_flag:
        gif_output = video_output.replace('.mp4', '.gif')
        create_gif(image_files, gif_output, fps=args.fps // 2)
    
    print("Done!")


if __name__ == "__main__":
    main()
