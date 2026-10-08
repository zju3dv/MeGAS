"""
Thermomechanical MPM simulation of a 3D Gaussian Splatting scene.

Pipeline:
    1. Load the aligned Gaussians `log/gaussians/<gs_file>/align/align_gaussian.ply`.
    2. Fill the interior with isotropic Gaussians, which together with the surface Gaussians
       serve as MPM particles.
    3. Simulate heating, melting and flow, and re-fit the surface Gaussians to the deformed
       particles via RIMLS after every frame.
    4. Every 10 frames, save a rendering and the deformed Gaussian attributes
       (`gs_render/deform_XXXX.pt`), which can be replayed with `gs_simulation_gui.py`.
"""
import sys

sys.path.append("gsplat/")

import argparse
import os
import time

import numpy as np
import torch
import warp as wp

from mpm_modules.MPMSolver import MPM_Simulator
from mpm_utils.internal_filling_utils import fill_internal_gaussian
from data_utils.save_utils import save_code
from scene.gaussian_model import GaussianModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thermomechanical MPM simulation of 3D Gaussians.")
    parser.add_argument('--gs_file', type=str, default="candy_house",
                        help="scene name under log/gaussians/")
    parser.add_argument("--solid_material", type=str, default="jelly",
                        choices=["jelly", "jelly_smooth"])
    parser.add_argument("--liquid_material", type=str, default="foam",
                        choices=["foam", "foam_smooth"])
    parser.add_argument("--collision", type=str, default="plane",
                        help="collision config in mpm_configs/collision/")

    parser.add_argument("--thermal_boundary", type=str, default="sphere", choices=["sphere", "plane"],
                        help="sphere: particle heat source; plane: constant-temperature plane")
    parser.add_argument("--thermal_sphere_boundary", type=str, default="candy_house.txt",
                        help="heat source range in mpm_configs/thermal/sphere_ranges/")
    parser.add_argument("--thermal_boundary_value", type=float, default=1.0)
    parser.add_argument("--incremental_thermal_bc", action='store_true',
                        help="raise the heat source temperature every 10 frames")
    parser.add_argument("--incremental_thermal_bc_delta", type=float, default=0.01)

    parser.add_argument("--grid_reso", type=int, default=100)
    parser.add_argument("--simulation_step", type=int, default=300, help="number of frames")
    parser.add_argument("--use_damping", action='store_true', help="damp velocities in the first 100 substeps")
    parser.add_argument("--save_note", type=str, default="test")

    args = parser.parse_args()

    wp.init()

    save_dir = os.path.join("log/gaussians", args.gs_file, "mpm",
                            f"solid-{args.solid_material}_liquid-{args.liquid_material}_thermal-{args.thermal_boundary}",
                            args.save_note)
    os.makedirs(save_dir, exist_ok=True)
    record_time = save_code(save_dir)
    save_record_dir = os.path.join(save_dir, record_time)
    save_gs_render_dir = os.path.join(save_record_dir, "gs_render")
    os.makedirs(save_gs_render_dir, exist_ok=True)
    print(f"\033[96m### Saving at: {save_record_dir} ### \033[0m")

    solid_mpm_confs = os.path.join("mpm_configs/materials", args.solid_material + ".conf")
    liquid_mpm_confs = os.path.join("mpm_configs/materials", args.liquid_material + ".conf")
    input_gaussian_file = os.path.join("log/gaussians", args.gs_file, "align", "align_gaussian.ply")

    input_gaussian = GaussianModel(sh_degree=3, use_raytracing=True)
    with torch.no_grad():
        input_gaussian.load_ply(input_gaussian_file)

    # Interior filling
    t1 = time.time()
    filled_gaussian, internal_gaussian, original_gaussian_indices = \
        fill_internal_gaussian(input_gaussian, resolution=128)
    print("\033[96mInternal Filling Time: ", time.time() - t1, "\033[0m")
    filled_gaussian.save_ply(os.path.join(save_record_dir, "fill_internal_gaussian.ply"))
    internal_gaussian.save_ply(os.path.join(save_record_dir, "internal_gaussian.ply"))
    torch.save(original_gaussian_indices, os.path.join(save_record_dir, "original_gaussian_indices.pt"))

    # Solver setup
    t1 = time.time()
    mpm_solver = MPM_Simulator(input_gaussian_model=filled_gaussian,
                               original_gaussian_indices=original_gaussian_indices,
                               n_grid=args.grid_reso)
    mpm_solver.use_damping = args.use_damping
    mpm_solver.init_material_from_config(solid_mpm_confs, args.solid_material, solid_type=True)
    mpm_solver.init_material_from_config(liquid_mpm_confs, args.liquid_material)
    print("\033[96mInit Time: ", time.time() - t1, "\033[0m")

    mpm_solver.init_collision_from_config(os.path.join("mpm_configs/collision", args.collision + ".conf"))
    mpm_solver.init_thermal_boundary_from_config(os.path.join("mpm_configs/thermal", args.thermal_boundary + ".conf"),
                                                 args.thermal_sphere_boundary,
                                                 args.thermal_boundary_value)

    # Render from the +x side of the unit domain
    mpm_solver.set_render_camera(look_from=np.array([2.0, 0.5, 0.5]),
                                 look_at=np.array([0.5, 0.5, 0.5]))

    for k in range(1, args.simulation_step):
        t1 = time.time()
        # One frame = 100 substeps of 1e-4 s
        for _ in range(100):
            mpm_solver.simulation(dt=1e-4)
        t2 = time.time()

        mpm_solver.update_gaussian_with_mlssurface()
        t3 = time.time()

        if k % 10 == 0 and args.incremental_thermal_bc:
            mpm_solver.update_thermal_boundary_incremental(delta_thermal=args.incremental_thermal_bc_delta)

        print(f"\033[96mStep {k}: Total {t3 - t1:.4f} s, Simulation {t2 - t1:.4f} s, "
              f"Surface Update {t3 - t2:.4f} s\033[0m")

        if k % 10 == 0:
            render_buffer, deform_infos = mpm_solver.render_gaussians()
            render_buffer.save(os.path.join(save_gs_render_dir, f"{k:04d}.png"))
            torch.save(deform_infos, os.path.join(save_gs_render_dir, f"deform_{k:04d}.pt"))
            print("\033[91mThe result is saved at \033[0m", save_record_dir)

    print("\033[91mThe result is saved at \033[0m", save_record_dir)
