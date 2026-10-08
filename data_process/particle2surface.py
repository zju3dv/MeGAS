import torch
import numpy as np
import warp as wp
import open3d as o3d
import mcubes
import trimesh
import os
import pyvista as pv
from skimage.measure import marching_cubes
import glob

device = wp.get_device()


@wp.kernel
def compute_min_dist(grid_points: wp.array(dtype=wp.vec3),  # [N,3]
                     gt_points: wp.array(dtype=wp.vec3),  # [N,3]
                     out_dists: wp.array(dtype=float)):  # [N,]
    i = wp.tid()
    p = grid_points[i]
    min_dist_sq = float(1e9)

    for j in range(gt_points.shape[0]):
        q = gt_points[j]
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        dz = p[2] - q[2]
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    out_dists[i] = wp.sqrt(min_dist_sq)


if __name__ == "__main__":
    wp.init()
    wp.config.verify_cuda = True

    resolution = 512

    x = torch.linspace(0, 1, resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).contiguous().cuda()

    grid_points_wp = wp.from_torch(grid_points.contiguous(), dtype=wp.vec3)  # .contiguous()很重要

    ply_files = glob.glob(
        "/home/zesongyang/Projects/3DGS/MPM/LoG-Split/output/blender/lego/iso_split_clamp++/point_cloud/latest/point_cloud.ply")
    for ply_file in sorted(ply_files):
        print("Processing file:", os.path.basename(ply_file))

        ply = o3d.io.read_point_cloud(ply_file)

        gt_ply = torch.tensor(np.array(ply.points), dtype=torch.float32).cuda()
        gt_ply_wp = wp.from_torch(gt_ply.contiguous(), dtype=wp.vec3)

        out_dists_wp = wp.empty(grid_points.shape[0], dtype=wp.float32)

        # 调用 kernel（全并行）
        wp.launch(kernel=compute_min_dist,
                  dim=grid_points.shape[0],  # 计算每一个grid的距离
                  inputs=[grid_points_wp, gt_ply_wp, out_dists_wp],
                  device=wp.get_device())

        # 转回 torch
        min_dists = wp.to_torch(out_dists_wp).cpu().numpy()

        # 将距离转换为SDF

        sdf = min_dists.reshape(resolution, resolution, resolution)

        vertices, triangles = mcubes.marching_cubes(sdf, 0.01)
        vertices = vertices / (resolution - 1)  # * 2 - 1
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.export(ply_file.replace("particles_", "sdf_").replace(".ply", ".obj"))

    '''
    verts, faces, _, _ = marching_cubes(sdf, 0.01)
    verts = verts / (resolution - 1) * 2 - 1
    mesh = pv.PolyData(verts)  # 创建一个网格对象
    mesh.faces = np.hstack((np.full((faces.shape[0], 1), 3), faces))  # 每个面由3个顶点构成
    mesh.plot()  # 可视化网格
    '''
