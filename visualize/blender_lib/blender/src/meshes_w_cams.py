import bpy
import trimesh
import matplotlib

from .materials import body_material

def prepare_data(cams, vertices):
    # Rough first-pass lift so cam CENTRES sit ≥ 0.02 above the floor plane.
    # A finer-grained "no buried cones" fix is in MeshesWithCameras.__init__,
    # which lifts further based on the actual rotated cam-marker bbox.
    cam_z_min = float(cams[:, 2, 3].min())
    vert_z_min = float(vertices[..., 2].min())
    offset = min(cam_z_min, vert_z_min)
    vertices[..., 2] -= offset
    cams[:, 2, 3] -= offset
    return cams, vertices


class MeshesWithCameras:
    def __init__(
        self,
        cams,
        vertices,
        mode,
        faces,
        oldrender=False,
        mesh_color="Blues",
        # cam_color="viridis",
        cam_color="Blues",
        # cam_color="coolwarm",
        **kwargs,
    ):
        cams, vertices = prepare_data(cams, vertices)

        self.faces = faces
        self.data = vertices
        self.mode = mode
        self.oldrender = oldrender
        self.mesh_color = mesh_color
        self.cam_color = cam_color
        self.cams = cams

        self.N = len(cams)
        self.trajectory = vertices[:, :, [0, 1]].mean(1)

        self.cam_vertices, self.cam_faces = self.load_cam_marker()

        # Compute the LOWEST world-Z that any rotated cam-marker vertex will
        # reach across all frames, and lift the entire trajectory so the deepest
        # point sits above the floor (Z=0.02). The simpler cam-centre lift above
        # leaves frustum extremities buried for cams with downward-tilted poses.
        import numpy as _np
        _all_z = []
        for _i in range(len(self.cams)):
            _rot = self.cams[_i, :3, :3].T
            _world = self.cam_vertices @ _rot + self.cams[_i, :3, 3]
            _all_z.append(_world[:, 2].min())
        _min_marker_z = float(min(_all_z))
        _floor_lift = max(0.0, 0.02 - _min_marker_z)
        if _floor_lift > 0:
            self.cams[:, 2, 3] += _floor_lift

    def load_cam_marker(self):
        # Robust path resolution — load relative to this package so cwd doesn't matter.
        import os
        stl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cam_marker.stl")
        cam_marker = trimesh.load_mesh(stl_path)
        cam_vertices = (cam_marker.vertices / cam_marker.vertices.max()) * 0.2
        cam_vertices[:, 2] *= -1
        cam_vertices[:, 2] += 0.2
        cam_faces = cam_marker.faces
        return cam_vertices, cam_faces

    def get_mesh_sequence_mat(self, frac):
        cmap = matplotlib.cm.get_cmap(self.mesh_color)
        # cmap = sns.color_palette("flare", as_cmap=True)

        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end - begin) * frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat

    def get_cam_sequence_mat(self, frac):
        cmap = matplotlib.cm.get_cmap(self.cam_color)
        # print(self.cam_color, frac)
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end - begin) * frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat

    def get_root(self, index):
        return self.cams[index, :3, 3]

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, cam_mat, mesh_mat, mode, keep_frame):
        from .tools import load_numpy_vertices_into_blender

        # vertices = self.data[index]
        # faces = self.faces
        # mesh_name = f"{str(index).zfill(4)}_mesh"
        # load_numpy_vertices_into_blender(
        #     vertices, faces, mesh_name, mesh_mat, index, mode, keep_frame
        # )
        mesh_name = None

        rot = self.cams[index, :3, :3].T
        marker = self.cam_vertices @ rot
        marker += self.cams[index, :3, 3]
        cam_name = f"{str(index).zfill(4)}_cam"
        load_numpy_vertices_into_blender(
            marker, self.cam_faces, cam_name, cam_mat, index, mode, keep_frame
        )

        return mesh_name, cam_name

    def show_cams(self, index, mode):
        # Render note: only emit ONE curve total — the full trajectory at the
        # final call. Earlier calls would otherwise draw cams[:index] each time,
        # stacking N redundant polylines on top of each other and producing the
        # "noisy outlier line" artifact when downsampling rate > 1/n_frames.
        if index < len(self.cams):
            return f"_skipped_{index}"

        name = f"final_curve"
        curveData = bpy.data.curves.new(name, type="CURVE")
        curveData.dimensions = "3D"
        curveData.resolution_u = 1

        # Patch: POLY spline (straight segs through cam centres). We tried BEZIER
        # auto-tangents but they overshoot near the trajectory ends, drawing arcs
        # to far-away cones that look like outliers. Caller-side Gaussian
        # smoothing already removes per-frame jitter, so straight segments through
        # smoothed points are visually fluent.
        polyline = curveData.splines.new("POLY")
        n_pts = len(self.cams)
        polyline.points.add(n_pts - 1)
        for i, coord in enumerate(self.cams[:, :3, 3]):
            polyline.points[i].co = (float(coord[0]), float(coord[1]), float(coord[2]), 1.0)

        curveOB = bpy.data.objects.new(name, curveData)
        curveData.bevel_depth = 0.012

        bpy.context.collection.objects.link(curveOB)

        if "video" in mode:
            # Initialize object as hidden
            curveOB.hide_viewport = True
            curveOB.hide_render = True
            curveOB.keyframe_insert(data_path="hide_viewport", frame=index - 1)
            curveOB.keyframe_insert(data_path="hide_render", frame=index - 1)

            # Make object visible at the specified frame
            curveOB.hide_viewport = False
            curveOB.hide_render = False
            curveOB.keyframe_insert(data_path="hide_viewport", frame=index)
            curveOB.keyframe_insert(data_path="hide_render", frame=index)

            if "accumulate" not in mode:
                # Hide object after the specified frame
                curveOB.hide_viewport = True
                curveOB.hide_render = True
                curveOB.keyframe_insert(data_path="hide_viewport", frame=index + 1)
                curveOB.keyframe_insert(data_path="hide_render", frame=index + 1)

        # Optional: Set shading to smooth
        bpy.ops.object.select_all(action="DESELECT")
        curveOB.select_set(True)
        bpy.context.view_layer.objects.active = curveOB
        bpy.ops.object.shade_smooth()
        bpy.ops.object.select_all(action="DESELECT")

        return name

    def __len__(self):
        return self.N
