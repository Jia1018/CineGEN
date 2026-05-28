"""
Render a single CineGen trajectory to a PNG using Blender 3.6.5.

Renders only the camera-marker trajectory (no character mesh). Reads a JSON
file in the format produced by ``visualize/postprocess.py``.

Usage (must be invoked through Blender, not Python directly)::

    /path/to/blender --background --python visualize/blender_render.py \\
        -- --traj_p path/to/clip.json --out_png path/to/out.png

The ``--`` separates Blender's args from this script's args.
"""
import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np

import bpy

# Add the local Blender library to PYTHONPATH (relative to this file).
HERE = Path(__file__).resolve().parent
LIB_ROOT = HERE / "blender_lib"
sys.path.insert(0, str(LIB_ROOT))

from blender.src.render import render
from blender.src.tools import delete_objs


def load_transform(transform_p):
    """Load the trajectory JSON, apply axis canonicalisation + per-clip normalisation.

    Mirrors the original rendering pipeline's canonical-frame convention.
    """
    with open(transform_p) as f:
        data = json.load(f)
    frames = data["frames"]
    c2ws = np.stack([np.array(it["transform_matrix"]) for it in frames], axis=0)
    # Axis canonicalisation
    c2ws[:, :, 0] = -c2ws[:, :, 0]
    c2ws[:, :, 1] = -c2ws[:, :, 1]
    ref_w2c = np.linalg.inv(c2ws[:1])
    ref_w2c_repeated = np.repeat(ref_w2c, c2ws.shape[0], axis=0)
    c2ws = np.matmul(ref_w2c_repeated, c2ws)[:, :3, :]
    T_norm = np.linalg.norm(c2ws[:, :3, 3], axis=-1).max()
    scale = T_norm + 1e-5
    c2ws[:, :3, 3] /= scale
    c2ws[:, :, 0] = -c2ws[:, :, 0]
    c2ws[:, :, 2] = -c2ws[:, :, 2]
    c2ws[:, :3, 3] = c2ws[:, :3, 3] * 5
    return c2ws  # (N, 3, 4)


def get_meshes_bounds(mesh_objects):
    if not mesh_objects:
        return np.array([-1, -1, 0]), np.array([1, 1, 0.1])
    all_vertices = []
    for obj in mesh_objects:
        if obj.type == "MESH":
            all_vertices.extend([obj.matrix_world @ v.co for v in obj.data.vertices])
    if not all_vertices:
        return np.array([-1, -1, 0]), np.array([1, 1, 0.1])
    arr = np.array(all_vertices)
    return arr.min(axis=0), arr.max(axis=0)


def look_at_rotation(camera_pos, target_pos, up=np.array([0, 0, 1])):
    direction = np.array(target_pos) - np.array(camera_pos)
    direction /= np.linalg.norm(direction) + 1e-8
    right = np.cross(direction, up)
    right /= np.linalg.norm(right) + 1e-8
    new_up = np.cross(right, direction)
    return np.array([right, new_up, -direction]).T


def rotation_matrix_to_euler(matrix):
    sy = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(matrix[2, 1], matrix[2, 2])
        y = np.arctan2(-matrix[2, 0], sy)
        z = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y = np.arctan2(-matrix[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def setup_camera(mesh_objects):
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(camera)
    min_xyz, max_xyz = get_meshes_bounds(mesh_objects)
    look_at = (min_xyz + max_xyz) / 2
    bbox = max_xyz - min_xyz
    distance = float(np.max(bbox)) * 1.0 + 1.0
    camera.location = [look_at[0], look_at[1] + distance, look_at[2] + distance * 1.5]
    camera.rotation_euler = rotation_matrix_to_euler(
        look_at_rotation(np.array(camera.location), look_at)
    )
    bpy.context.scene.camera = camera
    return camera


def set_plane_color(plane_name, color):
    plane = bpy.data.objects.get(plane_name)
    if not plane:
        return
    mat = bpy.data.materials.new(name="PlaneMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1)
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)


def render_camonly(traj_p: str, out_png: str, mode="image", selected_rate=0.2):
    traj = load_transform(traj_p)
    # Axis swap (Y↔Z, then negate Z) to align with Blender's coordinate frame.
    traj = traj[:, [0, 2, 1]]
    traj[:, 2] = -traj[:, 2]
    nframes = traj.shape[0]

    # Zero-placeholder vertices/faces (mesh rendering is disabled in
    # MeshesWithCameras anyway; vertices are only used for floor positioning).
    vertices = np.zeros((nframes, 1, 3), dtype=np.float32)
    faces = np.array([[0, 0, 0]], dtype=np.int64)
    cam_segments = np.zeros(nframes, dtype=np.int64)
    char_segments = np.zeros(nframes, dtype=np.int64)

    if "video" in mode:
        bpy.context.scene.frame_end = nframes - 1

    obj_names = render(
        traj=traj,
        vertices=vertices,
        faces=faces,
        cam_segments=cam_segments,
        char_segments=char_segments,
        denoising=True,
        oldrender=True,
        res="ultra",
        canonicalize=True,
        exact_frame=0.5,
        num=int(selected_rate * nframes),
        mode=mode,
        init=False,
    )
    mesh_objects = [bpy.data.objects[n] for n in obj_names if n in bpy.data.objects]
    setup_camera(mesh_objects)

    # Floor plane setup
    minb, maxb = get_meshes_bounds(mesh_objects)
    center = (minb + maxb) / 2; center[2] = 0
    plane_scale = (maxb - minb) / 2 * 1.2; plane_scale[2] = 1
    for obj in bpy.data.objects:
        if obj.name == "BigPlane":
            obj.scale = (0.01, 0.01, 0.01); obj.location[2] = -0.01
            set_plane_color("BigPlane", (1, 1, 1))
        if obj.name == "SmallPlane":
            obj.scale = plane_scale; obj.location = center; obj.location[2] = 0

    # Output: transparent PNG
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = out_png

    bpy.ops.render.render(write_still=True)


def main():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_p", required=True, help="Path to the trajectory JSON (from postprocess.py)")
    ap.add_argument("--out_png", required=True, help="Where to write the rendered PNG")
    ap.add_argument("--mode", default="image", choices=["image", "video", "video_accumulate"])
    ap.add_argument("--selected_rate", type=float, default=0.2,
                    help="Fraction of frames to render as camera markers")
    args = ap.parse_args(argv)
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    render_camonly(args.traj_p, args.out_png, args.mode, args.selected_rate)


if __name__ == "__main__":
    main()
