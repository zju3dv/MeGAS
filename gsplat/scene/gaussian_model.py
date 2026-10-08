#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import torch
import trimesh
from plyfile import PlyData, PlyElement
from torch import nn
from simple_knn._C import distCUDA2
from gstracer import GaussianTracer

from utils.general_utils import inverse_sigmoid, build_rotation, strip_symmetric, build_scaling_rotation
from utils.sh_utils import SH2RGB, eval_sh_wobase
from utils.mesh_utils import normal2rotation, rotmat2quaternion


class GaussianModel:
    """
    3D Gaussians with an additional per-Gaussian pseudo normal (stored as `nx, ny, nz` in the PLY file).
    Raw parameters are stored in their unconstrained form and activated by the `get_*` accessors.
    """

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.scaling_activation = torch.exp
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, use_raytracing=False, transmittance_min=0.001):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = None
        self.flag_load_normal = False
        self.setup_functions()

        self.render_mode = "raster"  # "raster" | "raytracing"
        if use_raytracing:
            # Bounding proxy for each Gaussian: an icosahedron whose inscribed sphere has unit radius
            icosahedron = trimesh.creation.icosahedron()
            self.unit_icosahedron_vertices = torch.from_numpy(icosahedron.vertices).float().cuda() * 1.2584
            self.unit_icosahedron_faces = torch.from_numpy(icosahedron.faces).long().cuda()

            self.gaussian_tracer = GaussianTracer(transmittance_min=transmittance_min)
            self.alpha_min = 1 / 255

    @property
    def get_scaling(self):
        if self.render_mode == "raster":
            return self.scaling_activation(self._scaling)
        elif self.render_mode == "raytracing":
            return self.scaling_activation(self._scaling).clamp(1e-6, 100)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_mat(self):
        return build_rotation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_pseudo_normal(self):
        return self.normal_activation(self._normal)

    def get_normal(self, cam_origin=None):
        """
        Geometric normal = direction of the shortest Gaussian axis, oriented towards `cam_origin`,
        or away from the centroid if no camera is given.
        """
        pts = self.get_xyz
        scales = self.get_scaling
        min_axis_id = torch.argmin(scales, dim=-1, keepdim=True)
        min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1)
        rot_matrix = build_rotation(self.get_rotation)
        ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)
        if cam_origin is None:
            o2p = pts - pts.mean(0)
        else:
            o2p = cam_origin - pts
        neg_msk = torch.sum(o2p * ndir, dim=-1) < 0
        ndir[neg_msk] = -ndir[neg_msk]
        return ndir

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    def get_colors(self, camera=None, shs_rotation=None):
        """
        RGB colors evaluated from the SH coefficients. View dependence is only included if a
        camera is given; `shs_rotation` [N, 3, 3] rotates the SH frame of each Gaussian, which is
        applied as the inverse rotation of the viewing direction.
        """
        if shs_rotation is not None:
            n_valid_gaussian = len(shs_rotation)
        else:
            n_valid_gaussian = len(self._xyz)
        colors = SH2RGB(self.get_features_dc)[:n_valid_gaussian]
        if self.active_sh_degree > 0 and camera is not None:
            xyz = self.get_xyz.detach()[:n_valid_gaussian]
            dir_pp = xyz - camera.camera_center[None]
            dir_pp = dir_pp / torch.norm(dir_pp, dim=-1, keepdim=True)
            shs_rest = self.get_features_rest[:n_valid_gaussian]
            if shs_rotation is not None:
                dir_pp = torch.matmul(shs_rotation.transpose(1, 2), dir_pp.unsqueeze(2)).squeeze(2)

            colors = colors.squeeze() + eval_sh_wobase(dir_pp, shs_rest, degree=self.active_sh_degree)
        return colors.squeeze().clamp(0, 1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def prepare_params(self):
        """Activated (xyz, SH features, opacity, scaling, rotation) of all Gaussians."""
        return self.get_xyz, self.get_features, self.get_opacity, self.get_scaling, self.get_rotation

    # ------------------------------------------------------------------ #
    # Construction and I/O
    # ------------------------------------------------------------------ #
    def create_from_known_params(self, xyz, features_dc, features_rest, scaling, rotation, opacity, normal=None):
        """Build from raw (unactivated) parameters."""
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        if normal is not None:
            self._normal = nn.Parameter(normal.requires_grad_(True))

    def create_from_known_gaussian(self, gaussian_model):
        self._xyz = gaussian_model._xyz.clone()
        self._features_dc = gaussian_model._features_dc.clone()
        self._features_rest = gaussian_model._features_rest.clone()
        self._scaling = gaussian_model._scaling.clone()
        self._rotation = gaussian_model._rotation.clone()
        self._opacity = gaussian_model._opacity.clone()
        self._normal = gaussian_model._normal.clone()

    def combine_gaussians(self, new_gaussian):
        self._xyz = nn.Parameter(torch.cat((self._xyz, new_gaussian._xyz), dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.cat((self._features_dc, new_gaussian._features_dc), dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.cat((self._features_rest, new_gaussian._features_rest), dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat((self._scaling, new_gaussian._scaling), dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat((self._rotation, new_gaussian._rotation), dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat((self._opacity, new_gaussian._opacity), dim=0).requires_grad_(True))

        if self._normal is not None and ((torch.linalg.norm(self._normal, dim=-1)).sum() > 0) and \
                new_gaussian._normal is not None and ((torch.linalg.norm(new_gaussian._normal, dim=-1)).sum() > 0):
            self._normal = nn.Parameter(torch.cat((self._normal, new_gaussian._normal), dim=0).requires_grad_(True))

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, crop_mask=None, save_colorful_pcd=True, use_pseudo_normal=True):
        """Save in the 3DGS PLY format; optionally also write `<name>_color.ply` as a colored point cloud."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if crop_mask is not None:
            valid_mask = crop_mask.detach().cpu()
        else:
            valid_mask = np.ones((len(self.get_xyz)), dtype=bool)

        xyz = self._xyz.detach().cpu().numpy()[valid_mask]
        if use_pseudo_normal:
            normals = self.get_pseudo_normal.detach().cpu().numpy()[valid_mask]
        else:
            normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[valid_mask]
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[
            valid_mask]
        opacities = self._opacity.detach().cpu().numpy()[valid_mask]
        scale = self._scaling.detach().cpu().numpy()[valid_mask]
        rotation = self._rotation.detach().cpu().numpy()[valid_mask]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scale, rotation], axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if save_colorful_pcd:
            import open3d as o3d
            o3d_pointcloud = o3d.geometry.PointCloud()
            o3d_pointcloud.points = o3d.utility.Vector3dVector(xyz)
            o3d_pointcloud.colors = o3d.utility.Vector3dVector(SH2RGB(f_dc).clip(0., 1.))
            if self.flag_load_normal:
                o3d_pointcloud.normals = o3d.utility.Vector3dVector(normals)
            o3d.io.write_point_cloud(os.path.abspath(path).split(".")[0] + "_color.ply", o3d_pointcloud)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # (P, F * SH_coeffs) -> (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Vanilla 3DGS files store all-zero normals; fall back to random unit vectors
        self.flag_load_normal = True
        normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])), axis=1)
        if normals.sum() == 0:
            self.flag_load_normal = False
            normals = np.random.random((len(xyz), 3)) * 2 - 1
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        print(f"Number of points loaded:  {len(self.get_xyz)}")

    # ------------------------------------------------------------------ #
    # Ray tracing (GSTracer)
    # ------------------------------------------------------------------ #
    def get_boundings(self, alpha_min=0.01):
        """Icosahedral proxies enclosing each Gaussian up to the level set where its alpha falls to alpha_min."""
        mu = self.get_xyz
        opacity = self.get_opacity

        L = build_scaling_rotation(self.get_scaling, self._rotation)

        vertices_b = (2 * (opacity / alpha_min).log()).sqrt()[:, None] * (
                self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces_b = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device="cuda")[:, None, None] * 12
        gs_id = torch.arange(mu.shape[0], device="cuda")[:, None].expand(-1, faces_b.shape[1])
        return vertices_b.reshape(-1, 3), faces_b.reshape(-1, 3), gs_id.reshape(-1)

    def get_SinvR(self):
        return build_scaling_rotation(1 / self.get_scaling, self._rotation)

    def build_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.gaussian_tracer.build_bvh(vertices_b, faces_b, gs_id)

    def trace(self, rays_o, rays_d):
        colors, depth, alpha, normal = self.gaussian_tracer.trace(rays_o, rays_d,
                                                                  self.get_xyz, self.get_opacity, self.get_SinvR(),
                                                                  self.get_features, self.get_pseudo_normal,
                                                                  alpha_min=self.alpha_min,
                                                                  deg=self.active_sh_degree)
        return {
            "render": colors,
            "depth": depth,
            "alpha": alpha,
            "normal": normal
        }

    # ------------------------------------------------------------------ #
    # In-place parameter updates; `update_mask` restricts the update to a subset
    # ------------------------------------------------------------------ #
    def set_normal(self, normal, update_mask=None):
        if update_mask is None:
            self._normal = nn.Parameter(normal.requires_grad_(True))
        else:
            _normal = self._normal.detach()
            _normal[update_mask] = normal
            self._normal = nn.Parameter(_normal.requires_grad_(True))

    def reset_scaling_with_knn(self, scale_ratio=1.0, update_mask=None):
        """Isotropic scales set to `scale_ratio` times the mean distance to the 3 nearest neighbors."""
        dist = torch.sqrt(torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)) * scale_ratio
        scales = torch.log(dist)[..., None].repeat(1, 3)

        if update_mask is not None:
            scales[~update_mask] = self._scaling[~update_mask].detach()
        self._scaling = nn.Parameter(scales.requires_grad_(True))

    def set_scaling(self, scales, update_mask=None, use_dist_clamp=False):
        """Set activated scales [N] (isotropic) or [N, 3]; optionally clamped by the k-NN distance."""
        if use_dist_clamp:
            dist = torch.sqrt(torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001))
            if update_mask is not None:
                scales = torch.minimum(scales, dist[update_mask])
            else:
                scales = torch.minimum(scales, dist)

        if len(scales.shape) == 1:
            scales = scales[:, None].repeat(1, 3)

        scales = torch.log(scales)
        if update_mask is None:
            self._scaling = nn.Parameter(scales.requires_grad_(True))
        else:
            scaling = self._scaling.detach()
            scaling[update_mask] = scales
            self._scaling = nn.Parameter(scaling.requires_grad_(True))

    def set_rotation(self, rotation_mat, update_mask=None):
        """Set rotations from matrices [N, 3, 3] or quaternions [N, 4]."""
        if len(rotation_mat.shape) == 3:
            rotation = rotmat2quaternion(rotation_mat)
        elif len(rotation_mat.shape) == 2 and rotation_mat.shape[-1] == 4:
            rotation = rotation_mat

        rotation = rotation.cuda()

        if update_mask is None:
            self._rotation = nn.Parameter(rotation.requires_grad_(True))
        else:
            _rotation = self._rotation.detach()
            _rotation[update_mask] = rotation
            self._rotation = nn.Parameter(_rotation.requires_grad_(True))

    def set_opacity(self, opacities):
        opacities = self.inverse_opacity_activation(opacities)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

    def set_features(self, features, update_mask=None):
        if update_mask is None:
            self._features_dc = nn.Parameter(features[:, 0:1].requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, 1:].requires_grad_(True))
        else:
            _features_dc = self._features_dc.detach()
            _features_rest = self._features_rest.detach()
            _features_dc[update_mask] = features[:, 0:1]
            _features_rest[update_mask] = features[:, 1:]

            self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
            self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))

    def set_scaling_rotation_from_normals(self, normals, update_mask=None):
        """Align the shortest axis with `normals` and flatten it into a surfel (scale_z = 1e-7)."""
        rotations = normal2rotation(normals.cpu()).cuda()
        if update_mask is None:
            self._rotation = nn.Parameter(rotations.requires_grad_(True))

            scaling = self.get_scaling.detach()
            scaling[:, 2] = 1e-7
            self._scaling = nn.Parameter(torch.log(scaling).requires_grad_(True))

            self._normal = nn.Parameter(normals.requires_grad_(True))
        else:
            _rotation = self._rotation.detach()
            _rotation[update_mask] = rotations
            self._rotation = nn.Parameter(_rotation.requires_grad_(True))

            scaling = self.get_scaling.detach()
            scaling[update_mask, 2] = 1e-7
            self._scaling = nn.Parameter(torch.log(scaling).requires_grad_(True))

            _normal = self._normal.detach()
            _normal[update_mask] = normals
            self._normal = nn.Parameter(_normal.requires_grad_(True))

    def rotate_transform(self, transform_rotation_mat, update_mask=None):
        """Left-multiply the Gaussian rotations by `transform_rotation_mat` [N, 3, 3]."""
        if update_mask is None:
            update_rotation = transform_rotation_mat @ self.get_rotation_mat
            self.set_rotation(update_rotation)
        else:
            update_rotation = transform_rotation_mat @ self.get_rotation_mat[update_mask]
            self.set_rotation(update_rotation, update_mask)
