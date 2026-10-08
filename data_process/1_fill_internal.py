import numpy as np
import os
import open3d as o3d
import warp as wp
import torch
import mcubes
import trimesh
from tqdm import tqdm
import raytracing
import argparse


# 将mesh变为SDF，再将内部进行填充


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


def compute_grid_inside_mesh(mesh, grid_coords, num_dirs=6):
    ## num_dir = [6,26]
    # 将grid从四面八方射出光线，如果和mesh的交点对应的法向量都是同向，说明在内部
    # ! 或者只考虑一个方向，看和mesh交点的个数是奇数还是偶数
    mesh_tracer = raytracing.RayTracer(mesh.vertices, mesh.faces)
    rays_o = torch.from_numpy(grid_coords).cuda()

    if num_dirs == 26:
        grid_dir = torch.tensor([-1, 0, 1])
        x, y, z = torch.meshgrid(grid_dir, grid_dir, grid_dir, indexing="ij")
        grid = torch.stack([x, y, z], dim=-1).reshape(-1, 3).float().cuda()
        rays_d = grid[(grid != 0).any(dim=-1)]
    elif num_dirs == 6:
        rays_d = torch.tensor([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [-1, 0, 0], [0, -1, 0], [0, 0, -1]
        ]).float().cuda()
    rays_d = rays_d / torch.norm(rays_d, dim=-1)[..., None]

    grid_intersects = torch.zeros((len(grid_coords), len(rays_d)), dtype=torch.uint8).cuda()  # 0表示未交，1表示从内部，2表示从外部
    grid_indexs = torch.arange(0, len(grid_intersects)).cuda()

    rays_chunk = 81920
    for b in range(len(grid_coords) // rays_chunk):
        len_rays = len(rays_o[b * rays_chunk:(b + 1) * rays_chunk])
        rays_o_chunk = rays_o[b * rays_chunk:(b + 1) * rays_chunk][:, None].repeat(1, len(rays_d), 1).reshape(-1, 3)
        rays_d_chunk = rays_d[None].repeat(len(rays_o[b * rays_chunk:(b + 1) * rays_chunk]), 1, 1).reshape(-1, 3)

        chunk_index = grid_indexs[b * rays_chunk:(b + 1) * rays_chunk]

        intersections, face_normals, face_id, depth = mesh_tracer.trace(rays_o_chunk, rays_d_chunk)

        # 物体内部
        inside_intersects = (face_normals * rays_d_chunk).sum(-1) > 0  # 法向量和射线同向，说明在物体内部
        inside_intersects = inside_intersects.reshape(len_rays, -1)  # 各个方向的情况
        grid_indices = torch.where(inside_intersects)
        grid_intersects[chunk_index[grid_indices[0]], grid_indices[1]] = 1

        # 物体外部
        grid_indices = torch.where(~inside_intersects)
        grid_intersects[chunk_index[grid_indices[0]], grid_indices[1]] = 2

        # 无交点
        grid_indices = torch.where(face_id.reshape(len_rays, -1) == -1)
        grid_intersects[chunk_index[grid_indices[0]], grid_indices[1]] = 0

    # 有可能非密闭，那考虑70%的点都是1
    return ((grid_intersects == 1).sum(-1) >= int(num_dirs * 0.7)).cpu().numpy()


if __name__ == '__main__':
    # ! TODO:
    ##  将其换到3DGS -> 要考虑非封闭的问题
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('--mesh_file', type=str, default="../assets/Warrior.obj")
    parser.add_argument('--init_state', type=str, default="ground", help="drop, ground")
    parser.add_argument('--resolution', type=int, default=256, help="drop, ground")
    args = parser.parse_args()

    mesh_file = args.mesh_file
    init_state = args.init_state  # ["drop","ground"]
    resolution = args.resolution

    save_dir = os.path.join("../log/mesh", os.path.basename(mesh_file).split(".")[0], "align", init_state)
    os.makedirs(save_dir, exist_ok=True)

    mesh_file = os.path.join(save_dir, "align_mesh.ply")
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)

    # voxel_size = 1 / args.resolution  # resolutions[-1]
    # ply_downsample = ply  # .voxel_down_sample(voxel_size)
    ply = o3d.geometry.PointCloud(o3d_mesh.vertices)
    gt_ply = torch.tensor(np.array(o3d_mesh.vertices), dtype=torch.float32).cuda()

    wp.init()

    # 构建 voxel grid
    # resolution = 256

    x = torch.linspace(0, 1, resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).contiguous().cuda()

    # 转为 Warp array
    if False:
        sdf_save_file = os.path.join(save_dir, f"sdf_{resolution}")
        if os.path.exists(sdf_save_file + ".npz"):
            sdf = np.load(sdf_save_file + ".npz")["arr_0"]
        else:
            grid_points_wp = wp.from_torch(grid_points, dtype=wp.vec3)
            gt_ply = torch.tensor(np.array(ply_downsample.points), dtype=torch.float32).cuda()
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
            mesh = trimesh.load_mesh(mesh_file)
            is_inside = mesh.contains(grid_points.cpu().numpy())
            signs = np.where(is_inside, -1, 1)

            sdf = (min_dists * signs).reshape(resolution, resolution, resolution)

            np.savez_compressed(sdf_save_file, sdf)  # np.load()["arr_0"]

    # vertices, triangles = mcubes.marching_cubes(sdf, 0.005)
    # vertices = vertices / (resolution - 1) * 2 - 1
    # mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # mesh.export(os.path.join(save_dir, f"{resolution}.ply"))

    # 填充内部的点
    mesh = trimesh.load_mesh(mesh_file)
    is_inside = compute_grid_inside_mesh(mesh, grid_points.cpu().numpy())
    # 将grid从四面八方射出光线，如果和mesh的交点对应的法向量都是同向，说明在内部

    # is_inside = mesh.contains()
    inside_points = grid_points.cpu().numpy()[is_inside]
    inside_ply = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inside_points))

    ply.paint_uniform_color([1, 0.5, 0.5])  # 灰色
    inside_ply.paint_uniform_color([0.5, 0.5, 1])
    # o3d.visualization.draw_geometries([ply_downsample])
    # o3d.visualization.draw_geometries([inside_ply])
    # o3d.visualization.draw_geometries([ply_downsample, inside_ply])
    surface_internel_pcd_indices = None

    # combine_ply = o3d.geometry.PointCloud()
    # combine_ply.points = o3d.utility.Vector3dVector(np.concatenate([np.array(ply.points), np.array(inside_ply.points)], axis=0))

    o3d.io.write_point_cloud(os.path.join(save_dir, "internal_filled.ply"), ply + inside_ply)

    mesh_indices = np.arange(len(ply.points))
    mesh_triangles = np.array(o3d_mesh.triangles)
    np.savez_compressed(os.path.join(save_dir, "mesh_indices_triangles.npz"), mesh_indices=mesh_indices,
                        mesh_triangles=mesh_triangles)

    '''
    gt_sdf_voxels = []
    resolution = 256  # , 128, 256

    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    z = torch.linspace(0, 1, resolution)

    grid_x, grid_y, grid_z = torch.meshgrid(torch.linspace(0, 1, resolution),
                                            torch.linspace(0, 1, resolution),
                                            torch.linspace(0, 1, resolution), indexing='ij')

    # 将网格坐标转换为3D点云形式
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).cuda()

    chunk = 2 * 1024
    min_dists = []
    for i in tqdm(range(0, grid_points.shape[0], chunk)):
        dists = torch.cdist(grid_points[i:i + chunk], gt_ply)
        min_dists.append(dists.min(dim=1).values.cpu())
        del dists
        torch.cuda.empty_cache()
    min_dists = torch.concat(min_dists)

    sdf = min_dists.reshape(resolution, resolution, resolution)

    vertices, triangles = mcubes.marching_cubes(sdf.numpy(), 0.005)
    vertices = vertices / (resolution - 1) * 2 - 1
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.export(os.path.join(save_dir, f"{resolution}.ply"))
    '''
