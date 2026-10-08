# Interactive viewer for the simulated Gaussian sequences produced by gs_simulation.py.
#
# Modified from the viewer of OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS),
# which builds on Gaussian-Splatting (https://team.inria.fr/graphdeco) and
# Gaussian-Grouping (https://github.com/lkeab/gaussian-grouping).
#
# Usage:
#   python gs_simulation_gui.py --mpm_path log/gaussians/<scene>/mpm/<setting>/<save_note>/@<timestamp>
import argparse
import glob
import math
import os
import sys

sys.path.append("gsplat")

import cv2
import dearpygui.dearpygui as dpg
import matplotlib
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov, fov2focal
from vis_utils.colormap_utils import thermal_vis


class OrbitCamera:
    """Orbit camera rotating around a movable center at distance `radius`."""

    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_quat([0, 0, 0, 1])
        self.fovy = fovy

    @property
    def pose(self):
        """[R | t] with R the camera-to-world rotation and t the world-to-camera translation (3DGS convention)."""
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius

        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        res[:3, 3] -= self.center
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        return res

    def orbit(self, dx, dy):
        up = self.rot.as_matrix()[:3, 1]
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius -= 0.1 * delta

    def pan(self, dx, dy, dz=0):
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt) -> None:
        self.opt = opt

        self.width = opt.window_size
        self.height = opt.window_size
        self.camera = OrbitCamera(self.width, self.height, r=opt.radius)

        bg_color = [0, 0, 0] if opt.black_background else [1, 1, 1]
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)

        print("loading model file...")
        with torch.no_grad():
            # Static appearance (SH) comes from the filled Gaussians; geometry from the per-frame deformation files
            self.engine = GaussianModel(sh_degree=3)
            self.engine.load_ply(os.path.join(opt.mpm_path, "fill_internal_gaussian.ply"))
            self.gaussian_indices = torch.load(os.path.join(opt.mpm_path, "original_gaussian_indices.pt"))

            self.background_engine = None
            if opt.background_gaussian_path and os.path.exists(opt.background_gaussian_path):
                self.background_engine = GaussianModel(sh_degree=3)
                self.background_engine.load_ply(opt.background_gaussian_path)
                aligned_translate_z = torch.load(opt.align_translate_z_path)
                self.background_engine._xyz = self.background_engine._xyz + torch.tensor(
                    [0.5, 0.5, 0.5 + aligned_translate_z]).cuda()

        self.deformation_files = sorted(glob.glob(os.path.join(opt.mpm_path, "gs_render", "*.pt")))
        assert len(self.deformation_files) > 0, f"No deformation files found in {opt.mpm_path}/gs_render"
        self.current_deformation_info = None
        self.update_gaussian_flag = True
        print("loading model file done.")

        # 0: RGB, 1: depth, 2: normal, 3: alpha
        self.img_mode = 0

        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        self.auto_dynamic_mode = False
        self.show_internal_gaussians = False
        self.show_pseudo_normal = False
        self.show_surface_gaussian = False
        self.show_bg_gaussian = False
        self.flag_use_cov3D_precomp = False
        self.flag_thermal_in_wipe_right = True
        self.flag_rgba = False

        self.flag_render_video = False
        self.render_video_cache = []
        self.flag_capture_frame = False
        self.flag_export_pointcloud = False

        self.render_camera = None
        self.flag_import_camera = False

        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb,
                                tag="_texture")

        with dpg.window(tag="_primary_window", width=self.width + 400, height=self.height):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        def toggle(name):
            return lambda: setattr(self, name, not getattr(self, name))

        def callback_render_video():
            self.flag_render_video = True
            self.auto_dynamic_mode = True
            dpg.set_value(self.current_deformation_time, 0)

        def round_slider(sender, app_data):
            dpg.set_value(sender, round(app_data / 0.1) * 0.1)

        def callback_export_pose():
            np.save("gui_camera_pose.npy", {"c2w": self.render_camera.get_c2w.cpu().numpy(),
                                            "intrinsics": self.render_camera.get_intrinsic.cpu().numpy(),
                                            "width": self.render_camera.image_width,
                                            "height": self.render_camera.image_height})
            print("### Export the Pose ###")

        def callback_import_pose():
            if self.flag_import_camera:
                self.flag_import_camera = False
                return
            camera_data = np.load("gui_camera_pose.npy", allow_pickle=True).item()
            c2w = camera_data["c2w"]
            intrinsics = camera_data["intrinsics"]
            width, height = camera_data["width"], camera_data["height"]
            self.render_camera = Camera(R=c2w[:3, :3], T=np.linalg.inv(c2w)[:3, 3],
                                        FoVx=focal2fov(intrinsics[0, 0], width),
                                        FoVy=focal2fov(intrinsics[1, 1], height),
                                        image_width=width, image_height=height)
            self.flag_import_camera = True
            print("### Import the Pose ###")

        with dpg.window(label="Control", tag="_control_window", width=350, height=550, pos=[self.width + 10, 0]):
            # Display
            dpg.add_button(label="Render_Option (RGB/Depth/Normal/Alpha)",
                           callback=lambda: setattr(self, "img_mode", (self.img_mode + 1) % 4))
            self.wipe_slider = dpg.add_slider_float(label="Thermal Vis Ratio", width=200, default_value=0.75,
                                                    min_value=0.0, max_value=1.0)
            self.wipe_angle_deg = dpg.add_slider_float(label="Wipe Angle (deg)", width=200, default_value=0.0,
                                                       min_value=0.0, max_value=180.0)
            dpg.add_button(label="Thermal_Dir", callback=toggle("flag_thermal_in_wipe_right"))
            dpg.add_checkbox(label="Render Background Gaussians", callback=toggle("show_bg_gaussian"))

            # Playback
            dpg.add_checkbox(label="use_dynamic_render", callback=toggle("auto_dynamic_mode"))
            self.current_deformation_time = dpg.add_slider_int(
                label="Deformation Time", width=200, default_value=0,
                max_value=len(self.deformation_files) - 1,
                callback=lambda: setattr(self, "update_gaussian_flag", True))
            self.deformation_speed = dpg.add_slider_int(label="Deformation Speed", width=200, default_value=1,
                                                        min_value=1, max_value=10)
            dpg.add_button(label="Render_Video", callback=callback_render_video)

            # Gaussian subsets and parameters
            dpg.add_checkbox(label="show_internal_render", callback=toggle("show_internal_gaussians"))
            dpg.add_checkbox(label="show_pseudo_normal", callback=toggle("show_pseudo_normal"))
            dpg.add_checkbox(label="show_surface_gaussian", callback=toggle("show_surface_gaussian"))
            dpg.add_checkbox(label="use_cov3D_precomp", callback=toggle("flag_use_cov3D_precomp"))
            self.scale_ratio = dpg.add_slider_float(label="Gaussian Scale", width=200, default_value=1.0,
                                                    min_value=0.1, max_value=2.0, format="%.1f",
                                                    callback=round_slider)

            # Export
            dpg.add_checkbox(label="RGBA Frame", default_value=False,
                             callback=lambda s, a: setattr(self, 'flag_rgba', a))
            dpg.add_button(label="Capture Frame", callback=lambda: setattr(self, "flag_capture_frame", True))
            dpg.add_button(label="Export Camera Pose", callback=callback_export_pose)
            dpg.add_checkbox(label="Import Camera Pose", callback=callback_import_pose)
            dpg.add_button(label="Export Pointcloud", callback=lambda: setattr(self, 'flag_export_pointcloud', True))

        # Mouse: left drag = orbit, middle drag = pan, wheel = zoom
        def callback_camera_wheel_scale(sender, app_data):
            if dpg.is_item_focused("_primary_window"):
                self.camera.scale(app_data)

        def move_handler(sender, pos):
            if dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if self.moving and (dx != 0.0 or dy != 0.0):
                    self.camera.orbit(-dx * 30, dy * 30)
                if self.moving_middle and (dx != 0.0 or dy != 0.0):
                    self.camera.pan(-dx * 20, dy * 20)
            self.mouse_pos = pos

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=toggle("moving"))
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=toggle("moving"))
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=toggle("moving_middle"))
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=toggle("moving_middle"))
            dpg.add_mouse_move_handler(callback=lambda s, a: move_handler(s, a))

        dpg.create_viewport(title="MeGAS-Viewer", width=self.width + 375, height=self.height, resizable=False)

        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            if self.auto_dynamic_mode:
                current_time = dpg.get_value(self.current_deformation_time)
                speed = dpg.get_value(self.deformation_speed)
                next_time = current_time + speed if current_time < len(self.deformation_files) - speed else 0
                dpg.set_value(self.current_deformation_time, next_time)
                self.update_gaussian_flag = True

            if self.flag_import_camera:
                cam = self.render_camera
            else:
                cam = self.construct_camera()
                self.render_camera = cam
            self.fetch_data(cam)
            dpg.render_dearpygui_frame()

    def construct_camera(self) -> Camera:
        pose = self.camera.pose
        fovy = self.camera.fovy * math.pi / 180.0
        fovx = focal2fov(fov2focal(fovy, self.height), self.width)
        return Camera(R=pose[:3, :3], T=pose[:3, 3], FoVx=fovx, FoVy=fovy,
                      image_width=self.width, image_height=self.height)

    @torch.no_grad()
    def fetch_data(self, view_camera):
        if self.update_gaussian_flag:
            deformation_file = self.deformation_files[dpg.get_value(self.current_deformation_time)]
            self.current_deformation_info = torch.load(deformation_file)
            self.update_gaussian_flag = False

        info = self.current_deformation_info
        xyz = info["xyz"]
        rotation = info["rotation"]
        scaling = info["scaling"]
        colors = info["color"]
        opacity = info["opacity"]
        pseudo_normal = info["pseudo_normal"]
        cov3D_precomp = info.get("cov3D_precomp", None)

        # Load the deformed geometry into the engine to evaluate normals (only argmin/orientation of the
        # scales is used, so the activated scales can be passed directly)
        self.engine.create_from_known_params(xyz, self.engine.get_features_dc, self.engine.get_features_rest,
                                             scaling, rotation, opacity, normal=pseudo_normal)

        # Offset keeps cold regions distinguishable from the empty background in the thermal map
        thermal = torch.clamp_min(info["thermal"][:, None], 0.0) + 0.1

        normal = self.engine.get_pseudo_normal if self.show_pseudo_normal else \
            self.engine.get_normal(cam_origin=view_camera.camera_center)
        W2C_R = torch.from_numpy(view_camera.R).float().cuda().T
        normal = (W2C_R @ normal.T).T
        extra_attributes = torch.cat([normal, thermal], dim=-1)

        scaling = scaling * (dpg.get_value(self.scale_ratio) ** 2)

        if self.flag_export_pointcloud:
            self.flag_export_pointcloud = False
            export_pcd = o3d.geometry.PointCloud()
            export_pcd.points = o3d.utility.Vector3dVector(xyz[self.gaussian_indices].cpu().numpy())
            export_pcd.colors = o3d.utility.Vector3dVector(colors[self.gaussian_indices].cpu().numpy())
            export_pcd.normals = o3d.utility.Vector3dVector(pseudo_normal[self.gaussian_indices].cpu().numpy())
            o3d.io.write_point_cloud("gaussian_pointcloud.ply", export_pcd)

        # By default only the original surface Gaussians are shown
        show_gaussian_indices = self.gaussian_indices
        if self.show_internal_gaussians:
            show_gaussian_indices = torch.ones_like(xyz[:, 0], dtype=bool)
        elif self.show_surface_gaussian and 'surface_internal_indices' in info:
            show_gaussian_indices = torch.cat([show_gaussian_indices, info['surface_internal_indices']], dim=0)

        xyz = xyz[show_gaussian_indices]
        opacity = opacity[show_gaussian_indices]
        scaling = scaling[show_gaussian_indices]
        rotation = rotation[show_gaussian_indices]
        colors = colors[show_gaussian_indices]
        extra_attributes = extra_attributes[show_gaussian_indices]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[show_gaussian_indices]

        # Discard Gaussians with degenerate (NaN) rotations
        valid_rotation_mask = ~rotation.sum(-1).isnan()
        xyz = xyz[valid_rotation_mask]
        opacity = opacity[valid_rotation_mask]
        scaling = scaling[valid_rotation_mask]
        rotation = rotation[valid_rotation_mask]
        colors = colors[valid_rotation_mask]
        extra_attributes = extra_attributes[valid_rotation_mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[valid_rotation_mask]

        # Foreground alpha (without background Gaussians) for RGBA export
        if self.flag_rgba:
            outputs_foreground = render(view_camera, xyz, None, opacity, None, None,
                                        torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                                        cov3D_precomp=cov3D_precomp, colors_precomp=colors,
                                        extra_attributes=extra_attributes, only_rgb=False)
            foreground_alpha = outputs_foreground['render_alpha'].squeeze().detach().cpu().numpy()

        if self.background_engine is not None and self.show_bg_gaussian:
            bg = self.background_engine
            xyz = torch.cat([xyz, bg.get_xyz], dim=0)
            opacity = torch.cat([opacity, bg.get_opacity], dim=0)
            scaling = torch.cat([scaling, bg.get_scaling], dim=0)
            rotation = torch.cat([rotation, bg.get_rotation], dim=0)
            colors = torch.cat([colors, bg.get_colors(view_camera)], dim=0)
            extra_attributes_bg = torch.cat([(W2C_R @ bg.get_pseudo_normal.T).T,
                                             torch.ones_like(bg.get_xyz[:, :1]) * 0.05], dim=-1)
            extra_attributes = torch.cat([extra_attributes, extra_attributes_bg], dim=0)
            if cov3D_precomp is not None:
                cov3D_precomp = torch.cat([cov3D_precomp, bg.get_covariance()], dim=0)

        if self.flag_use_cov3D_precomp and cov3D_precomp is not None:
            scaling = None
            rotation = None
        else:
            cov3D_precomp = None

        outputs = render(view_camera, xyz, None, opacity, scaling, rotation, self.bg_color,
                         cov3D_precomp=cov3D_precomp, colors_precomp=colors,
                         extra_attributes=extra_attributes, only_rgb=False)

        if self.img_mode == 0:
            render_buffer = outputs["render"].permute(1, 2, 0).clip(0, 1).cpu().numpy()
        elif self.img_mode == 1:
            depth = outputs['render_depth'][0].squeeze().detach().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-7)
            render_buffer = matplotlib.colormaps["Spectral"](depth, bytes=False)[:, :, 0:3].astype(np.float32)
        elif self.img_mode == 2:
            normal = outputs["extra_attri"][:3]
            render_buffer = (-normal.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        elif self.img_mode == 3:
            render_buffer = outputs['render_alpha'].repeat([3, 1, 1]).permute(1, 2, 0).clip(0, 1).cpu().numpy()

        if self.flag_rgba:
            render_buffer = render_buffer * foreground_alpha[..., None] + (1.0 - foreground_alpha[..., None])

        render_buffer = self.blend_thermal(render_buffer, outputs["extra_attri"][3:],
                                           foreground_alpha if self.flag_rgba else None)

        self.render_buffer = render_buffer.reshape(-1)
        dpg.set_value("_texture", self.render_buffer)

        if self.flag_render_video:
            self.render_video_cache.append(Image.fromarray(np.uint8(255 * render_buffer)))
            if len(self.render_video_cache) == len(self.deformation_files) // dpg.get_value(self.deformation_speed):
                video_writer = cv2.VideoWriter(os.path.join(self.opt.mpm_path, 'render_video.mp4'),
                                               cv2.VideoWriter_fourcc(*'mp4v'), 24,
                                               self.render_video_cache[0].size)
                for frame in self.render_video_cache:
                    video_writer.write(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
                video_writer.release()

                self.render_video_cache = []
                self.flag_render_video = False
                self.auto_dynamic_mode = False

        if self.flag_capture_frame:
            self.flag_capture_frame = False
            frame = self.render_buffer.reshape(self.height, self.width, 3)
            if self.flag_rgba:
                frame = np.concatenate([frame, foreground_alpha[..., None]], axis=-1)
            os.makedirs("render_capture_frame", exist_ok=True)
            Image.fromarray(np.uint8(255 * frame)).save("render_capture_frame/capture_frame.png")

    def blend_thermal(self, render_buffer, render_thermal, foreground_alpha=None):
        """
        Wipe transition between the current render and the thermal map. The split line is
        perpendicular to the direction given by `Wipe Angle` (0 deg: vertical line) and is placed
        at `Thermal Vis Ratio` along it; `Thermal_Dir` selects which side shows temperature.
        """
        render_thermal_vis = thermal_vis(render_thermal, min_value=0.0, max_value=1.1
                                         ).permute(1, 2, 0).detach().cpu().numpy()

        wipe_ratio = dpg.get_value(self.wipe_slider)
        theta = np.deg2rad(dpg.get_value(self.wipe_angle_deg))

        H, W, _ = render_buffer.shape
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = np.cos(theta), np.sin(theta)
        s = xs * nx + ys * ny

        corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=np.float32)
        s_corners = corners @ np.array([nx, ny], dtype=np.float32)
        s_min, s_max = float(np.min(s_corners)), float(np.max(s_corners))

        k = s_min + wipe_ratio * (s_max - s_min)
        feather = 1.0
        a = np.clip((s - (k - feather)) / (2.0 * feather), 0.0, 1.0).astype(np.float32)[..., None]

        if self.flag_thermal_in_wipe_right:
            blended = (1.0 - a) * render_buffer + a * render_thermal_vis
        else:
            blended = (1.0 - a) * render_thermal_vis + a * render_buffer

        if foreground_alpha is not None:
            render_buffer[foreground_alpha > 0.5] = blended[foreground_alpha > 0.5]
        else:
            render_buffer[:] = blended
        return render_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive viewer for MeGAS simulation results.")
    parser.add_argument("--mpm_path", type=str, required=True,
                        help="output directory of gs_simulation.py, i.e. .../<save_note>/@<timestamp>")
    parser.add_argument("--background_gaussian_path", type=str, default="",
                        help="optional static background Gaussians (.ply)")
    parser.add_argument("--align_translate_z_path", type=str, default="",
                        help="z-offset (.pt) aligning the background with the simulation domain")
    parser.add_argument("--window_size", type=int, default=800)
    parser.add_argument("--radius", type=float, default=2.0, help="initial camera distance")
    parser.add_argument("--black_background", action="store_true")
    opt = parser.parse_args()

    gui = GaussianSplattingGUI(opt)
    gui.render()
