import copy
import os

import numpy as np
import torch
import warp as wp
from PIL import Image
from scipy.spatial import cKDTree

from mpm_modules.MPMStruct import MPMParticle_State, MPMGrid_State, MPMMaterial_State
from mpm_modules.MPMCollider import *
from mpm_utils.mpm_utils import *
from mpm_utils.warp_utils import *
from mpm_utils.config_utils import load_config
from mpm_utils.mls_utils import compute_mls_surface
from visual_utils.camera_utils import get_viewmatrix

from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from utils.mesh_utils import rotmat2quaternion
from utils.transform_utils import normals_rotmat_from_B_to_A
from vis_utils.colormap_utils import thermal_vis


class MPM_Simulator:
    """
    Thermomechanical MLS-MPM simulator whose particles are the (surface + interior) Gaussians.

    Each step performs P2G -> grid update (momentum, collisions, implicit heat advection-diffusion)
    -> G2P -> plastic return mapping -> thermal boundary conditions and solid-liquid phase update,
    and finally deforms the Gaussian covariances with the deformation gradient.

    Particle state:  x, v, C (APIC affine matrix), F (deformation gradient), b_e_hat, T, phase.
    Grid state:      m, v (momentum / velocity), T.
    Material:        E (Young's modulus), nu (Poisson's ratio), mu / lam (Lame parameters), bulk (bulk modulus).
    """

    def __init__(self,
                 input_gaussian_model: GaussianModel,
                 original_gaussian_indices: torch.Tensor,
                 n_grid=100,
                 device="cuda:0"):
        self.device = device
        self.use_damping = False

        input_particles = input_gaussian_model.get_xyz.detach().cpu().numpy()
        self.gaussian_model = input_gaussian_model
        self.original_gaussian_indices = original_gaussian_indices

        # Interior Gaussians are rendered as isotropic kernels with identity rotation
        self.internal_gaussian_mask = torch.ones(len(input_particles), dtype=torch.bool, device="cuda")
        self.internal_gaussian_mask[self.original_gaussian_indices] = False
        self.internal_gaussian_rotation = torch.eye(3, device="cuda")[None].repeat(
            int(self.internal_gaussian_mask.sum()), 1, 1)

        # Grid
        self.mpm_grid_state = MPMGrid_State()
        self.n_grid = n_grid
        grid_shape = (self.n_grid, self.n_grid, self.n_grid)
        self.mpm_grid_state.n_grid = n_grid
        self.mpm_grid_state.dx, self.mpm_grid_state.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.mpm_grid_state.grid_m = wp.zeros(shape=grid_shape, dtype=float, device=device)
        self.mpm_grid_state.grid_v = wp.zeros(shape=grid_shape, dtype=wp.vec3, device=device)
        self.mpm_grid_state.grid_thermal = wp.zeros(shape=grid_shape, dtype=float, device=device)
        self.mpm_grid_state.grid_rhs = wp.zeros(shape=grid_shape, dtype=float, device=device)
        self.mpm_grid_state.cg_r = wp.zeros(shape=grid_shape, dtype=float, device=device)
        self.mpm_grid_state.cg_p = wp.zeros(shape=grid_shape, dtype=float, device=device)
        self.mpm_grid_state.cg_Ap = wp.zeros(shape=grid_shape, dtype=float, device=device)
        self.mpm_grid_state.cg_y = wp.zeros(shape=grid_shape, dtype=float, device=device)
        self.mpm_grid_state.cg_dot_buf = wp.zeros(shape=1, dtype=float, device=self.device)

        # Particles
        self.mpm_particle_state = MPMParticle_State()
        self.input_particles = input_particles
        self.n_particles = len(input_particles)
        self.mpm_particle_state.n_particles = self.n_particles
        print(f"### Init Point Num: {self.n_particles}, "
              f"Range: {input_particles.min(0).tolist()}, {input_particles.max(0).tolist()}  ###")

        self.mpm_particle_state.particle_x = wp.from_numpy(input_particles, dtype=wp.vec3, device=device)
        self.mpm_particle_state.particle_v = wp.zeros(shape=self.n_particles, dtype=wp.vec3, device=device)
        self.mpm_particle_state.particle_F = wp.zeros(shape=self.n_particles, dtype=wp.mat33, device=device)
        wp.launch(kernel=set_mat33_to_identity, dim=self.n_particles, inputs=[self.mpm_particle_state.particle_F],
                  device=device)
        self.mpm_particle_state.particle_C = wp.zeros(shape=self.n_particles, dtype=wp.mat33, device=device)
        self.mpm_particle_state.particle_be_hat = wp.zeros(shape=self.n_particles, dtype=wp.mat33, device=device)
        wp.launch(kernel=set_mat33_to_identity, dim=self.n_particles, inputs=[self.mpm_particle_state.particle_be_hat],
                  device=device)

        self.mpm_particle_state.particle_thermal = wp.zeros(shape=self.n_particles, dtype=float, device=device)
        self.mpm_particle_state.delta_particle_thermal = wp.zeros(shape=self.n_particles, dtype=wp.vec3, device=device)
        self.mpm_particle_state.phase_state = wp.zeros(shape=self.n_particles, dtype=int, device=device)

        for material_state in [self.mpm_particle_state.solid_material_state,
                               self.mpm_particle_state.liquid_material_state]:
            material_state.material_type = 0
            material_state.E = wp.zeros(shape=self.n_particles, dtype=float, device=device)
            material_state.nu = wp.zeros(shape=self.n_particles, dtype=float, device=device)
            material_state.mu = wp.zeros(shape=self.n_particles, dtype=float, device=device)
            material_state.lam = wp.zeros(shape=self.n_particles, dtype=float, device=device)
            material_state.bulk = wp.zeros(shape=self.n_particles, dtype=float, device=device)
            material_state.elastic_method_type = wp.zeros(shape=self.n_particles, dtype=int, device=device)
            material_state.yield_threshold = 0.0
            material_state.viscosity = 0.0
            material_state.shear_ratio = 0.0

        self.mpm_particle_state.particle_density = wp.zeros(shape=self.n_particles, dtype=float, device=device)
        self.mpm_particle_state.particle_vol = wp.zeros(shape=self.n_particles, dtype=float, device=device)
        self.mpm_particle_state.particle_mass = wp.zeros(shape=self.n_particles, dtype=float, device=device)

        self.init_particle_volume()

        self.gravity = wp.vec3(0.0, 0.0, -9.8)

        self.elastic_method_index = {
            "fixed_corotated": 0,
            "stvk": 1,
            "neo_hookean": 2,
            "herschel": 3
        }
        self.material_type_index = {
            "jelly": 0,
            "foam": 5,
        }
        self.collision_type_index = {
            "sticky": 0,
        }

        # Gaussian kernel deformation
        gaussian_cov3D_precomp = vec_to_mat33(self.gaussian_model.get_covariance().detach())
        self.mpm_particle_state.particle_init_cov3D = torch_to_wp_mat33(gaussian_cov3D_precomp)
        self.mpm_particle_state.particle_cov3D = wp.zeros(shape=self.n_particles, dtype=wp.mat33, device=device)
        self.mpm_particle_state.particle_R = wp.zeros(shape=self.n_particles, dtype=wp.mat33, device=device)
        self.mpm_particle_state.particle_Scale = wp.zeros(shape=self.n_particles, dtype=wp.vec3, device=device)

        # `updated_info_gaussian_model` tracks the physically deformed Gaussians across steps;
        # `updated_gaussian_model` additionally carries the MLS-refined surface used for rendering.
        self.updated_info_gaussian_model = copy.copy(self.gaussian_model)
        self.updated_gaussian_model = copy.copy(self.gaussian_model)
        self.surface_view_cameras = None
        self.surface_gaussian_indices = set()
        self.surface_gaussian_from_origin_indices = set()
        self.surface_gaussian_from_internal_indices = set()
        self.shs_rotation_mats = torch.eye(3, device="cuda")[None].repeat(len(self.gaussian_model.get_xyz), 1, 1)

        self.times = 0

    def init_particle_volume(self):
        """Particle volume = cell volume / number of particles in the cell."""
        grid_counts = wp.zeros(shape=(self.n_grid, self.n_grid, self.n_grid),
                               dtype=int, device=self.device)

        @wp.kernel
        def compute_grid_particle_num(grid_particle_counts: wp.array(dtype=int, ndim=3),
                                      mpm_particle_state: MPMParticle_State,
                                      mpm_grid_state: MPMGrid_State):
            p = wp.tid()

            grid_pos = mpm_particle_state.particle_x[p] * mpm_grid_state.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
            wp.atomic_add(grid_particle_counts, base_pos_x, base_pos_y, base_pos_z, 1)

        wp.launch(kernel=compute_grid_particle_num, dim=self.n_particles,
                  inputs=[grid_counts, self.mpm_particle_state, self.mpm_grid_state],
                  device=self.device)

        @wp.kernel
        def compute_particle_volume(grid_particle_counts: wp.array(dtype=int, ndim=3),
                                    mpm_particle_state: MPMParticle_State,
                                    mpm_grid_state: MPMGrid_State):
            p = wp.tid()
            grid_pos = mpm_particle_state.particle_x[p] * mpm_grid_state.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)

            mpm_particle_state.particle_vol[p] = (mpm_grid_state.dx * mpm_grid_state.dx * mpm_grid_state.dx) / \
                                                 wp.float(grid_particle_counts[base_pos_x, base_pos_y, base_pos_z])

        wp.launch(kernel=compute_particle_volume, dim=self.n_particles,
                  inputs=[grid_counts, self.mpm_particle_state, self.mpm_grid_state],
                  device=self.device)

    def init_material_from_config(self, material_config_file: str, material_type: str = "jelly", device="cuda:0",
                                  solid_type=False):
        """
        Load a material from a HOCON config. The solid material additionally defines gravity
        and density (hence particle mass); the liquid material defines the heat diffusivity
        and the melting temperature.
        """
        material_kwargs = load_config(material_config_file)["materials"]
        if "jelly" in material_type:
            material_type = "jelly"
        if "foam" in material_type:
            material_type = "foam"

        if solid_type:
            material_state = self.mpm_particle_state.solid_material_state
        else:
            material_state = self.mpm_particle_state.liquid_material_state

        material_state.material_type = self.material_type_index[material_type]

        if "elastic_method" in material_kwargs:
            material_state.elastic_method_type.fill_(self.elastic_method_index[material_kwargs["elastic_method"]])

        if "E" in material_kwargs:
            wp.launch(kernel=set_value_to_float_array, dim=self.n_particles,
                      inputs=[material_state.E, material_kwargs["E"]], device=device)
        if "nu" in material_kwargs:
            wp.launch(kernel=set_value_to_float_array, dim=self.n_particles,
                      inputs=[material_state.nu, material_kwargs["nu"]], device=device)

        if "yield_threshold" in material_kwargs:
            material_state.yield_threshold = material_kwargs["yield_threshold"]
        if "viscosity" in material_kwargs:
            material_state.viscosity = material_kwargs["viscosity"]
        if "shear_ratio" in material_kwargs:
            material_state.shear_ratio = material_kwargs["shear_ratio"]

        if solid_type:
            if "gravity" in material_kwargs:
                self.gravity[2] = -material_kwargs["gravity"]

            if "density" in material_kwargs:
                wp.launch(kernel=set_value_to_float_array, dim=self.n_particles,
                          inputs=[self.mpm_particle_state.particle_density, material_kwargs["density"]],
                          device=device)

                @wp.kernel
                def compute_mass(particle_mass: wp.array(dtype=float), particle_density: wp.array(dtype=float),
                                 particle_vol: wp.array(dtype=float)):
                    p = wp.tid()
                    particle_mass[p] = particle_density[p] * particle_vol[p]

                wp.launch(kernel=compute_mass, dim=self.n_particles,
                          inputs=[self.mpm_particle_state.particle_mass, self.mpm_particle_state.particle_density,
                                  self.mpm_particle_state.particle_vol],
                          device=device)

        # Derive (mu, lam, bulk) from (E, nu) first; explicitly given moduli override the derived ones
        if "mu" not in material_kwargs or "lam" not in material_kwargs or "bulk" not in material_kwargs:
            self.compute_lame_bulk_modulus(solid_type)

        if "mu" in material_kwargs:
            wp.launch(kernel=set_value_to_float_array, dim=self.n_particles,
                      inputs=[material_state.mu, material_kwargs["mu"]], device=device)
        if "lam" in material_kwargs:
            wp.launch(kernel=set_value_to_float_array, dim=self.n_particles,
                      inputs=[material_state.lam, material_kwargs["lam"]], device=device)
        if "bulk" in material_kwargs:
            wp.launch(kernel=set_value_to_float_array, dim=self.n_particles,
                      inputs=[material_state.bulk, material_kwargs["bulk"]], device=device)

        self.mpm_grid_state.heat_diff_ratio = 0.01
        if "heat_diff_ratio" in material_kwargs:
            self.mpm_grid_state.heat_diff_ratio = material_kwargs["heat_diff_ratio"]

        self.mpm_particle_state.thermal_phase_threshold = 0.8
        if "thermal_phase_threshold" in material_kwargs:
            self.mpm_particle_state.thermal_phase_threshold = material_kwargs["thermal_phase_threshold"]

    def compute_lame_bulk_modulus(self, solid_type=False):
        @wp.kernel
        def compute_modulus(material_state: MPMMaterial_State):
            p = wp.tid()
            material_state.mu[p] = material_state.E[p] / (2.0 * (1.0 + material_state.nu[p]))
            material_state.lam[p] = material_state.E[p] * material_state.nu[p] / (
                    (1.0 + material_state.nu[p]) * (1.0 - 2.0 * material_state.nu[p]))
            material_state.bulk[p] = material_state.E[p] / (
                    3.0 * (1.0 - 2.0 * material_state.nu[p]))

        material_state = self.mpm_particle_state.solid_material_state if solid_type \
            else self.mpm_particle_state.liquid_material_state
        wp.launch(kernel=compute_modulus, dim=self.n_particles, inputs=[material_state], device=self.device)

    def init_collision_from_config(self, collision_config_file: str):
        """Planar colliders; a sticky collider sets the velocity of grid nodes behind the plane to zero."""
        plane_collision_kwargs = load_config(collision_config_file)["collision"]["planes"]

        self.colliders = []
        for plane_key in plane_collision_kwargs.keys():
            plane_collision_kwarg = plane_collision_kwargs[plane_key]
            point = plane_collision_kwarg["point"]
            normal = plane_collision_kwarg["normal"]

            collider = Colliders()
            collider.point = wp.vec3(point[0], point[1], point[2])
            collider.normal = wp.vec3(normal[0], normal[1], normal[2])
            collider.collider_type = self.collision_type_index[plane_collision_kwarg["type"]]
            self.colliders.append(collider)

        self.colliders = wp.array(self.colliders, dtype=Colliders, device=self.device)

    def init_thermal_boundary_from_config(self, thermal_config_file: str, thermal_sphere_boundary: str,
                                          thermal_boundary_value: float):
        """
        Thermal boundary conditions:
            planes:  Dirichlet condition on the grid nodes behind each plane.
            spheres: heat source on the particles inside a sphere (cx, cy, cz, r) or an
                     axis-aligned ellipsoid (cx, cy, cz, rx, ry, rz) read from
                     `mpm_configs/thermal/sphere_ranges/<thermal_sphere_boundary>`, held at
                     `thermal_boundary_value`.
        """
        thermal_kwargs = load_config(thermal_config_file)["thermal"]

        self.thermal_bounds = []
        if "planes" in thermal_kwargs:
            plane_thermal_kwargs = thermal_kwargs["planes"]
            for plane_key in plane_thermal_kwargs.keys():
                plane_thermal_kwarg = plane_thermal_kwargs[plane_key]
                point = plane_thermal_kwarg["point"]
                normal = plane_thermal_kwarg["normal"]

                thermal_bound = Plane_Thermal_Boundary()
                thermal_bound.boundary_type = 0
                thermal_bound.point = wp.vec3(point[0], point[1], point[2])
                thermal_bound.normal = wp.vec3(normal[0], normal[1], normal[2])
                thermal_bound.thermal = plane_thermal_kwarg.get("thermal", 0.0)
                self.thermal_bounds.append(thermal_bound)

        self.sphere_bounds = []
        if "spheres" in thermal_kwargs:
            for _ in thermal_kwargs["spheres"].keys():
                sphere_bound = Particle_Thermal_Boundary()
                sphere_bound.boundary_type = 1

                sphere_range = np.loadtxt(os.path.join("mpm_configs/thermal/sphere_ranges", thermal_sphere_boundary))
                if sphere_range.shape[0] == 4:
                    sphere_center, sphere_radius = sphere_range[:3], sphere_range[3]
                    range_particle_mask = np.linalg.norm((self.input_particles - sphere_center),
                                                         axis=-1) <= sphere_radius
                elif sphere_range.shape[0] == 6:
                    sphere_center, sphere_scale = sphere_range[:3], sphere_range[3:]
                    rel = self.input_particles - sphere_center
                    range_particle_mask = ((rel[:, 0] / sphere_scale[0]) ** 2 +
                                           (rel[:, 1] / sphere_scale[1]) ** 2 +
                                           (rel[:, 2] / sphere_scale[2]) ** 2) <= 1.0

                sphere_bound.particle_mask = wp.from_numpy(range_particle_mask, dtype=int, device=self.device)
                sphere_bound.thermal = thermal_boundary_value
                self.sphere_bounds.append(sphere_bound)

    def update_thermal_boundary_incremental(self, delta_thermal=0.01):
        """Gradually heat the source, capped at 1.0."""
        for sphere_bound in self.sphere_bounds:
            sphere_bound.thermal += delta_thermal
            sphere_bound.thermal = min(sphere_bound.thermal, 1.0)

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def Zero_Grid(self):
        wp.launch(kernel=zero_grid,
                  dim=(self.n_grid, self.n_grid, self.n_grid),
                  inputs=[self.mpm_grid_state],
                  device=self.device)

    def P2G(self):
        launch_sync(kernel=p2g_apic,
                    dim=self.n_particles,
                    inputs=[self.mpm_particle_state, self.mpm_grid_state, self.dt,
                            self.use_damping and self.times < 100],
                    device=self.device)

    def solve_implicit_diffusion_cg(self, max_iters=200, tol=1e-6):
        """
        Implicit backward-Euler heat diffusion solved by conjugate gradients:

            (I - alpha * dt * Laplacian) T^{n+1} = T^n - dt * (v . grad T^n)

        Init:  x_0 = T (current grid temperature),  r_0 = b - A x_0,  p_0 = r_0
        Iter:  a_k     = (r_k . r_k) / (p_k . A p_k)
               x_{k+1} = x_k + a_k p_k
               r_{k+1} = r_k - a_k A p_k        (stop if ||r_{k+1}||^2 < tol^2)
               b_k     = (r_{k+1} . r_{k+1}) / (r_k . r_k)
               p_{k+1} = r_{k+1} + b_k p_k
        """
        N = (self.n_grid, self.n_grid, self.n_grid)
        grid = self.mpm_grid_state

        wp.launch(apply_A, dim=N, inputs=[grid, self.dt, grid.grid_thermal, grid.cg_y], device=self.device)
        wp.launch(vec_sub, dim=N, inputs=[grid.cg_r, grid.grid_rhs, grid.cg_y], device=self.device)
        wp.launch(vec_copy, dim=N, inputs=[grid.cg_p, grid.cg_r], device=self.device)

        wp.launch(clear_array, dim=1, inputs=[grid.cg_dot_buf], device=self.device)
        wp.launch(vec_dot, dim=N, inputs=[grid.cg_r, grid.cg_r, grid.cg_dot_buf], device=self.device)
        rsold = wp.to_torch(grid.cg_dot_buf)[0].item()

        for it in range(max_iters):
            wp.launch(apply_A, dim=N, inputs=[grid, self.dt, grid.cg_p, grid.cg_Ap], device=self.device)

            wp.launch(clear_array, dim=1, inputs=[grid.cg_dot_buf], device=self.device)
            wp.launch(vec_dot, dim=N, inputs=[grid.cg_p, grid.cg_Ap, grid.cg_dot_buf], device=self.device)
            pAp = wp.to_torch(grid.cg_dot_buf)[0].item()
            alpha = rsold / max(pAp, 1e-30)

            wp.launch(vec_axpy, dim=N, inputs=[grid.grid_thermal, alpha, grid.cg_p], device=self.device)
            wp.launch(vec_axpy, dim=N, inputs=[grid.cg_r, -alpha, grid.cg_Ap], device=self.device)

            wp.launch(clear_array, dim=1, inputs=[grid.cg_dot_buf], device=self.device)
            wp.launch(vec_dot, dim=N, inputs=[grid.cg_r, grid.cg_r, grid.cg_dot_buf], device=self.device)
            rsnew = wp.to_torch(grid.cg_dot_buf)[0].item()
            if rsnew < tol * tol:
                break

            beta = rsnew / max(rsold, 1e-30)
            wp.launch(vec_scale, dim=N, inputs=[grid.cg_p, beta], device=self.device)
            wp.launch(vec_axpy, dim=N, inputs=[grid.cg_p, 1.0, grid.cg_r], device=self.device)

            rsold = rsnew

    def Grid_Update(self):
        """
        Momentum update with gravity and collisions, followed by the thermal update following
        "A Thermomechanical Hybrid Incompressible Material Point Method": explicit upwind
        advection builds the right-hand side, implicit diffusion is solved with CG, and the
        Dirichlet planes are enforced last.
        """
        launch_sync(kernel=grid_update,
                    dim=(self.n_grid, self.n_grid, self.n_grid),
                    inputs=[self.mpm_grid_state, self.gravity, self.dt, self.colliders,
                            self.use_damping and self.times < 100],
                    device=self.device)

        launch_sync(kernel=grid_build_rhs_advect,
                    dim=(self.n_grid, self.n_grid, self.n_grid),
                    inputs=[self.mpm_grid_state, self.dt],
                    device=self.device)

        self.solve_implicit_diffusion_cg(max_iters=100, tol=1e-6)

        for thermal_bound in self.thermal_bounds:
            launch_sync(kernel=grid_update_thermal_bc,
                        dim=(self.n_grid, self.n_grid, self.n_grid),
                        inputs=[self.mpm_grid_state, thermal_bound],
                        device=self.device)

    def G2P(self):
        launch_sync(kernel=g2p,
                    dim=self.n_particles,
                    inputs=[self.mpm_particle_state, self.mpm_grid_state, self.dt],
                    device=self.device)

    def Correct_Plastic_Gradient(self):
        launch_sync(kernel=compute_correct_gradient,
                    dim=self.n_particles,
                    inputs=[self.mpm_particle_state, self.dt],
                    device=self.device)

    def Particle_Update(self):
        """Hold heat-source particles at their temperature, then update the solid / liquid phase."""
        for sphere_bound in self.sphere_bounds:
            launch_sync(kernel=particle_update_thermal_bc,
                        dim=self.n_particles,
                        inputs=[self.mpm_particle_state, sphere_bound],
                        device=self.device)

        launch_sync(kernel=particle_phase_update,
                    dim=self.n_particles,
                    inputs=[self.mpm_particle_state],
                    device=self.device)

    def Gaussian_Cov3D_Update(self):
        launch_sync(kernel=update_cov3D,
                    dim=self.n_particles,
                    inputs=[self.mpm_particle_state],
                    device=self.device)

    def simulation(self, dt):
        self.dt = dt

        self.Zero_Grid()
        self.P2G()
        self.Grid_Update()
        self.G2P()
        self.Correct_Plastic_Gradient()
        self.Particle_Update()
        self.Gaussian_Cov3D_Update()

        self.times += 1

    # ------------------------------------------------------------------ #
    # Gaussian update and rendering
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_gaussian_with_mlssurface(self):
        """
        Topology-adaptive Gaussian update after each simulation step.

        1. Axis alignment: the eigenvectors (R, s) of F Sigma_0 F^T are reordered and sign-flipped to
           best match the axes of the Gaussians from the previous step.
        2. Normal propagation: the shortest-axis normals are oriented consistently with the previous
           pseudo normals.
        3. Surface detection: Gaussians that are the maximum-weight contributor of any pixel in a set
           of surrounding views form the visible surface; surface Gaussians that originate from the
           interior are newly exposed by deformation (e.g. cracks or melting).
        4. MLS refinement: an RIMLS surface fitted to the original surface Gaussians re-orients both the
           original and the newly exposed surface Gaussians, which are flattened onto that surface.
        5. Appearance transfer: newly exposed interior Gaussians copy the SH coefficients of a nearby,
           highly opaque original Gaussian, and the SH frames of all surface Gaussians are rotated by
           the change of their normals.
        """
        # 1. Axis alignment
        mpm_rotation, mpm_scaling = wp.to_torch(self.mpm_particle_state.particle_R), wp.to_torch(
            self.mpm_particle_state.particle_Scale)
        gaussian_rotation = self.updated_info_gaussian_model.get_rotation_mat
        # Interior Gaussians have identity rotation, so the alignment is a pure axis reordering there
        score = (mpm_rotation.transpose(1, 2) @ gaussian_rotation).abs()  # [N, i, j]: |<R_mpm[:, i], R_gs[:, j]>|
        best_idx = torch.argmax(score, dim=1).long()

        idx = best_idx.unsqueeze(1).expand(-1, mpm_rotation.size(1), -1)
        aligned = torch.gather(mpm_rotation, dim=2, index=idx)
        dots = (aligned * gaussian_rotation).sum(dim=1)
        sign_cols = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
        mpm_rotation_align = aligned * sign_cols.unsqueeze(1)
        mpm_scaling_align = torch.gather(mpm_scaling, dim=1, index=best_idx)

        particles = wp.to_torch(self.mpm_particle_state.particle_x).clone()

        updated_gaussian = GaussianModel(sh_degree=3)
        updated_gaussian.create_from_known_params(
            xyz=particles.clone(),
            features_dc=self.gaussian_model.get_features_dc.clone(),
            features_rest=self.gaussian_model.get_features_rest.clone(),
            scaling=torch.log(mpm_scaling_align).clone(),
            rotation=rotmat2quaternion(mpm_rotation_align).clone(),
            opacity=self.gaussian_model._opacity.clone(),
        )
        updated_gaussian.set_rotation(self.internal_gaussian_rotation, self.internal_gaussian_mask)

        # 2. Normal propagation
        last_pseudo_normal = self.updated_info_gaussian_model.get_pseudo_normal.clone()
        current_normal = updated_gaussian.get_normal().clone()
        normal_dir_mask = (current_normal * last_pseudo_normal).sum(-1) < 0
        current_normal[normal_dir_mask] *= -1
        updated_gaussian.set_normal(current_normal.clone())
        self.updated_info_gaussian_model = GaussianModel(sh_degree=3)
        self.updated_info_gaussian_model.create_from_known_gaussian(updated_gaussian)

        # 3. Surface detection
        last_surface_gaussian_indices = copy.copy(self.surface_gaussian_indices)
        self.surface_gaussian_indices = self.extract_surface_gaussian(copy.copy(updated_gaussian),
                                                                      self.original_gaussian_indices)
        repeat_surface_gaussian_indices = last_surface_gaussian_indices.intersection(self.surface_gaussian_indices)

        # 4. MLS refinement, fitted on the original surface Gaussians only
        original_gaussian_indices = set(self.original_gaussian_indices.tolist())
        surface_gaussian_from_origin_indices = self.surface_gaussian_indices.intersection(original_gaussian_indices)
        surface_gaussian_from_internal_indices = self.surface_gaussian_indices - surface_gaussian_from_origin_indices
        self.surface_gaussian_from_origin_indices = surface_gaussian_from_origin_indices
        self.surface_gaussian_from_internal_indices = surface_gaussian_from_internal_indices

        surface_from_origin_indices_tensor = torch.tensor(list(surface_gaussian_from_origin_indices), device="cuda")
        surface_from_internal_indices_tensor = torch.tensor(list(surface_gaussian_from_internal_indices), device="cuda")

        # Newly exposed interior Gaussians are flattened onto the MLS surface
        _, _, internal_mls_normals = compute_mls_surface(
            refer_points=particles[surface_from_origin_indices_tensor].detach().cpu().numpy(),
            refer_normals=current_normal[surface_from_origin_indices_tensor].detach().cpu().numpy(),
            query_points=particles[surface_from_internal_indices_tensor].detach().cpu().numpy(),
            return_tensor=True
        )
        updated_gaussian.set_scaling_rotation_from_normals(internal_mls_normals, surface_from_internal_indices_tensor)

        # Original surface Gaussians are rotated to the MLS normals
        _, _, origin_mls_normals = compute_mls_surface(
            refer_points=particles[surface_from_origin_indices_tensor].detach().cpu().numpy(),
            refer_normals=current_normal[surface_from_origin_indices_tensor].detach().cpu().numpy(),
            query_points=particles[surface_from_origin_indices_tensor].detach().cpu().numpy(),
            return_tensor=True
        )
        updated_gaussian.rotate_transform(normals_rotmat_from_B_to_A(
            current_normal[surface_from_origin_indices_tensor].detach(),
            origin_mls_normals),
            surface_from_origin_indices_tensor)
        updated_gaussian.set_normal(origin_mls_normals, surface_from_origin_indices_tensor)

        # 5. Appearance transfer, skipped for interior Gaussians already exposed in the previous step
        update_color_internal_indices_tensor = torch.tensor(
            list(surface_gaussian_from_internal_indices - repeat_surface_gaussian_indices),
            device="cuda")
        if len(update_color_internal_indices_tensor) == 0:
            self.updated_gaussian_model = GaussianModel(sh_degree=3)
            self.updated_gaussian_model.create_from_known_gaussian(updated_gaussian)
            return

        # Source selection score: opacity^gamma / normalized distance over the K nearest originals
        origin_gaussian_xyz = particles[self.original_gaussian_indices].cpu().numpy()
        origin_gaussian_opacity = self.gaussian_model.get_opacity[self.original_gaussian_indices, 0].cpu().numpy()
        update_surface_internal_gaussian_xyz = particles[update_color_internal_indices_tensor].cpu().numpy()
        query_tree = cKDTree(origin_gaussian_xyz)
        neareast_dists, neareast_indices = query_tree.query(update_surface_internal_gaussian_xyz, k=12, workers=-1)

        eps = 1e-12
        gamma = 2.0
        nn1 = neareast_dists[:, 0]
        scale = np.median(nn1[np.isfinite(nn1)]) + eps
        op_cand = origin_gaussian_opacity[neareast_indices]
        score = (np.power(op_cand, gamma)) / (neareast_dists / scale + eps)
        internal_update_indices = neareast_indices[np.arange(neareast_indices.shape[0]), np.argmax(score, axis=1)]
        internal_update_from_origin_indices = self.original_gaussian_indices[internal_update_indices]

        internal_update_features = updated_gaussian.get_features[internal_update_from_origin_indices]
        updated_gaussian.set_features(internal_update_features.clone(), update_color_internal_indices_tensor)

        # SH frame rotation: rest-state pseudo normal -> current MLS normal
        surface_from_origin_rotation_mat = normals_rotmat_from_B_to_A(
            self.gaussian_model.get_pseudo_normal[surface_from_origin_indices_tensor],
            origin_mls_normals)
        self.shs_rotation_mats[surface_from_origin_indices_tensor] = surface_from_origin_rotation_mat

        surface_from_internal_rotation_mat = normals_rotmat_from_B_to_A(
            self.gaussian_model.get_pseudo_normal[surface_from_internal_indices_tensor],
            internal_mls_normals)
        self.shs_rotation_mats[surface_from_internal_indices_tensor] = surface_from_internal_rotation_mat

        self.updated_gaussian_model = GaussianModel(sh_degree=3)
        self.updated_gaussian_model.create_from_known_gaussian(updated_gaussian)

    def create_surface_cameras(self,
                               focal,
                               image_wh,
                               azimuths,
                               elevations,
                               radius,
                               center):
        """Cameras on a sphere around `center`, looking at it, used for visibility-based surface detection."""
        view_cameras = []

        image_width, image_height = image_wh, image_wh
        camera_fov = focal2fov(focal, image_width)

        for elevation in elevations:
            for azimuth in azimuths:
                theta = np.deg2rad(azimuth)
                phi = np.deg2rad(elevation)

                cam_x = radius * np.cos(phi) * np.cos(theta) + center[0]
                cam_y = radius * np.cos(phi) * np.sin(theta) + center[1]
                cam_z = radius * np.sin(phi) + center[2]

                look_from = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
                look_at = np.array(center, dtype=np.float32)

                c2w = get_viewmatrix(look_from, look_at)
                view_cameras.append(Camera(R=c2w[:3, :3], T=np.linalg.inv(c2w)[:3, 3],
                                           FoVx=camera_fov, FoVy=camera_fov,
                                           image_width=image_width, image_height=image_height))

        return view_cameras

    def extract_surface_gaussian(self, gs_model, original_gaussian_indices):
        """
        Indices of the Gaussians visible from the surrounding cameras, i.e. those that are the
        maximum-weight contributor of at least one pixel. The original surface Gaussians are always
        included. Scales are first reset from the k-NN spacing so that stretched interior kernels
        with arbitrary orientation do not occlude the true surface.
        """
        if self.surface_view_cameras is None:
            self.surface_view_cameras = self.create_surface_cameras(
                focal=560, image_wh=512,
                azimuths=[i * 45 for i in range(int(360 // 45))],
                elevations=[-75, -50, -25, 0, 25, 50, 75],
                radius=1.25,
                center=(gs_model.get_xyz.max(0)[0] + gs_model.get_xyz.min(0)[0]).detach().cpu().numpy() / 2
            )

        gs_model.reset_scaling_with_knn(scale_ratio=0.5)

        surface_gaussian_indices = set(original_gaussian_indices.tolist())
        for view_camera in self.surface_view_cameras:
            xyz, features, opacity, scaling, rotation = gs_model.prepare_params()
            outputs = render(view_camera, xyz, features, opacity, scaling, rotation,
                             torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"))
            surface_gaussian_indices.update(set(outputs["point_id"].tolist()))

        return surface_gaussian_indices

    def set_render_camera(self, look_from, look_at, resolution=800):
        c2w = get_viewmatrix(look_from, look_at)
        fov = focal2fov(resolution, resolution)
        self.gaussian_view_camera = Camera(R=c2w[:3, :3], T=np.linalg.inv(c2w)[:3, 3],
                                           FoVx=fov, FoVy=fov,
                                           image_width=resolution, image_height=resolution)

    @torch.no_grad()
    def render_gaussians(self):
        """
        Render the current state from the preview camera.

        :return: side-by-side RGB / temperature preview image, and the per-Gaussian state of this
                 frame, which is consumed by the interactive viewer `gs_simulation_gui.py`.
        """
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        xyz, _, opacity, scaling, rotation = self.updated_gaussian_model.prepare_params()
        colors = self.updated_gaussian_model.get_colors(shs_rotation=self.shs_rotation_mats)
        pseudo_normals = self.updated_gaussian_model.get_pseudo_normal
        cov3D_precomp = mat33_to_vec(wp.to_torch(self.mpm_particle_state.particle_cov3D))
        phase_state = wp.to_torch(self.mpm_particle_state.phase_state)
        thermal = wp.to_torch(self.mpm_particle_state.particle_thermal)

        deform_infos = {
            "xyz": xyz,
            "rotation": rotation,
            "scaling": scaling,
            "cov3D_precomp": cov3D_precomp,
            "color": colors,
            "opacity": opacity,
            "pseudo_normal": pseudo_normals,
            "thermal": thermal,
            "phase_state": phase_state,
            "surface_origin_indices": torch.tensor(list(self.surface_gaussian_from_origin_indices)),
            "surface_internal_indices": torch.tensor(list(self.surface_gaussian_from_internal_indices))
        }

        # Degenerate kernels can yield NaN quaternions
        valid_rotation_mask = ~rotation.sum(-1).isnan()

        outputs = render(self.gaussian_view_camera,
                         xyz[valid_rotation_mask],
                         None,
                         opacity[valid_rotation_mask],
                         scaling[valid_rotation_mask],
                         rotation[valid_rotation_mask],
                         background,
                         colors_precomp=colors[valid_rotation_mask],
                         extra_attributes=thermal[valid_rotation_mask][:, None])
        render_buffer = outputs["render"].permute(1, 2, 0).detach().cpu().numpy()
        thermal_buffer = thermal_vis(outputs["extra_attri"], min_value=0.0,
                                     max_value=1.0).permute(1, 2, 0).detach().cpu().numpy()
        render_buffer = Image.fromarray(np.uint8(255 * np.concatenate([render_buffer, thermal_buffer], axis=1)))
        return render_buffer, deform_infos
