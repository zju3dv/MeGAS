import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree

from scene.gaussian_model import GaussianModel


def find_nearest(target_points, query_points, knn_num=16):
    query_tree = cKDTree(target_points)
    dists, nearest_indices = query_tree.query(query_points, k=knn_num, workers=-1)
    return dists, nearest_indices, target_points[nearest_indices]


@wp.kernel
def compute_neareast_point(grid_points: wp.array(dtype=wp.vec3),
                           gaussian_xyzs: wp.array(dtype=wp.vec3),
                           gaussian_scalings: wp.array(dtype=wp.vec3),
                           gaussian_rotations: wp.array(dtype=wp.mat33),

                           knn_num: int,
                           nearest_points_indices: wp.array(dtype=int),  # [N * K]

                           out_neareast_indices: wp.array(dtype=int),
                           out_neareast_distances: wp.array(dtype=float)
                           ):
    """
    For every interior sample point, find the surface Gaussian whose tangent ellipse contains the
    orthogonal projection of the point and whose supporting plane is closest. Each Gaussian is
    treated as a flat disk: its shortest axis is the normal n, the two remaining axes span the
    tangent plane with semi-axes (r0, r1). If no candidate ellipse contains the projection, the
    nearest Gaussian center is used instead.
    """
    tid = wp.tid()
    p = grid_points[tid]

    min_dist_sq = float(1e9)
    min_dist_index = int(-1)

    eps_axis = 1.0e-8
    eps_inside = 1.0e-3

    for i in range(knn_num):
        q = gaussian_xyzs[nearest_points_indices[tid * knn_num + i]]
        s = gaussian_scalings[nearest_points_indices[tid * knn_num + i]]
        R = gaussian_rotations[nearest_points_indices[tid * knn_num + i]]

        # Normal axis k: the axis with the smallest scale
        s0 = wp.abs(s[0])
        s1 = wp.abs(s[1])
        s2 = wp.abs(s[2])

        k = 0
        minv = s0
        if s1 < minv:
            k = 1
            minv = s1
        if s2 < minv:
            k = 2
            minv = s2

        # Tangent axes a, b
        a = 0
        b = 1
        if k == 0:
            a = 1
            b = 2
        elif k == 1:
            a = 0
            b = 2
        else:
            a = 0
            b = 1

        n = wp.vec3(R[0, k], R[1, k], R[2, k])
        t0 = wp.vec3(R[0, a], R[1, a], R[2, a])
        t1 = wp.vec3(R[0, b], R[1, b], R[2, b])

        r0 = wp.max(wp.abs(s[a]), eps_axis)
        r1 = wp.max(wp.abs(s[b]), eps_axis)

        # Signed point-to-plane distance and in-plane projection
        pq = p - q
        d = wp.dot(pq, n)
        p_proj = p - d * n

        pp = p_proj - q
        u = wp.dot(pp, t0)
        v = wp.dot(pp, t1)

        val = (u / r0) * (u / r0) + (v / r1) * (v / r1)

        if val <= 1.0 + eps_inside:
            dsq = d * d
            if dsq < min_dist_sq:
                min_dist_sq = dsq
                min_dist_index = nearest_points_indices[tid * knn_num + i]

    if min_dist_index == -1:
        for i in range(knn_num):
            q = gaussian_xyzs[nearest_points_indices[tid * knn_num + i]]
            dsq = wp.length(p - q)
            if dsq < min_dist_sq:
                min_dist_sq = dsq
                min_dist_index = nearest_points_indices[tid * knn_num + i]

    out_neareast_indices[tid] = min_dist_index
    out_neareast_distances[tid] = wp.sqrt(min_dist_sq)


def fill_internal_gaussian(gaussian: GaussianModel, resolution=128):
    """
    Fill the interior of a surface Gaussian model with isotropic Gaussians.

    1. Inside test: rays are cast from every node of a resolution^3 grid in the six axis directions
       and traced through the Gaussians. A hit whose normal points along the ray direction is a
       back-face hit; a node is labeled interior if at least 70% of its rays hit a back face.
    2. Attribute assignment: every interior node inherits appearance (SH, opacity, normal) from
       the closest surface Gaussian found by `compute_neareast_point`, and is initialized as an
       opaque isotropic Gaussian whose radius is the plane distance, clamped to 1 / (1.5 * resolution).

    :return: (surface + interior Gaussians, interior Gaussians, indices of the surface Gaussians)
    """
    gaussian.build_bvh()
    gaussian.render_mode = "raytracing"

    x = torch.linspace(0, 1, resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).contiguous().cuda()

    rays_o = grid_points
    rays_d = torch.tensor([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1]
    ]).float().cuda()
    rays_d = rays_d / torch.norm(rays_d, dim=-1)[..., None]

    # 0: no hit, 1: back-face hit (inside), 2: front-face hit (outside)
    grid_intersects = torch.zeros((len(grid_points), len(rays_d)), dtype=torch.uint8).cuda()
    grid_indexs = torch.arange(0, len(grid_intersects)).cuda()

    rays_chunk = 81920
    for b in range(len(grid_points) // rays_chunk):
        len_rays = len(rays_o[b * rays_chunk:(b + 1) * rays_chunk])
        rays_o_chunk = rays_o[b * rays_chunk:(b + 1) * rays_chunk][:, None].repeat(1, len(rays_d), 1).reshape(-1, 3)
        rays_d_chunk = rays_d[None].repeat(len(rays_o[b * rays_chunk:(b + 1) * rays_chunk]), 1, 1).reshape(-1, 3)

        chunk_index = grid_indexs[b * rays_chunk:(b + 1) * rays_chunk]

        trace_results = gaussian.trace(rays_o_chunk, rays_d_chunk)
        ray_normals = trace_results["normal"]
        ray_alpha = trace_results["alpha"]

        ray_normals = ray_normals / (torch.linalg.norm(ray_normals, dim=-1)[:, None] + 1e-6)
        inside_intersects = ((ray_normals * rays_d_chunk).sum(-1) > 0) & (ray_alpha > 0.9)
        inside_intersects = inside_intersects.reshape(len_rays, -1)
        grid_indices = torch.where(inside_intersects)
        grid_intersects[chunk_index[grid_indices[0]], grid_indices[1]] = 1

        outside_intersects = ((ray_normals * rays_d_chunk).sum(-1) < 0) & (ray_alpha > 0.9)
        outside_intersects = outside_intersects.reshape(len_rays, -1)
        grid_indices = torch.where(outside_intersects)
        grid_intersects[chunk_index[grid_indices[0]], grid_indices[1]] = 2

    gaussian.render_mode = "raster"

    inside_masks = ((grid_intersects == 1).sum(-1) >= int(len(rays_d) * 0.7)).cpu().numpy()
    filled_gaussian_points_xyz = grid_points[inside_masks]

    knn_num = 32
    dev = wp.get_device("cuda")
    _, nearest_indices, _ = find_nearest(gaussian.get_xyz.detach().cpu().numpy(),
                                         filled_gaussian_points_xyz.detach().cpu().numpy(),
                                         knn_num=knn_num)

    nearest_points_indices_wp = wp.array(nearest_indices.reshape(-1).astype(np.int32), dtype=int, device=dev)

    filled_gaussian_nearest_indices = wp.empty(filled_gaussian_points_xyz.shape[0], dtype=int)
    filled_gaussian_nearest_indices.fill_(-1)
    filled_gaussian_nearest_distances = wp.empty(filled_gaussian_points_xyz.shape[0], dtype=float)

    wp.launch(
        kernel=compute_neareast_point,
        dim=filled_gaussian_points_xyz.shape[0],
        inputs=[
            wp.from_torch(filled_gaussian_points_xyz, dtype=wp.vec3),
            wp.from_torch(gaussian.get_xyz, dtype=wp.vec3),
            wp.from_torch(gaussian.get_scaling, dtype=wp.vec3),
            wp.from_torch(gaussian.get_rotation_mat, dtype=wp.mat33),
            knn_num,
            nearest_points_indices_wp,
            filled_gaussian_nearest_indices,
            filled_gaussian_nearest_distances
        ]
    )

    filled_gaussian_nearest_indices = wp.to_torch(filled_gaussian_nearest_indices)
    filled_gaussian_nearest_distances = wp.to_torch(filled_gaussian_nearest_distances)
    filled_gaussian_nearest_distances = torch.clamp(filled_gaussian_nearest_distances, max=1.0 / (resolution * 1.5))

    filled_gaussian = GaussianModel(sh_degree=3)
    filled_gaussian.create_from_known_params(
        filled_gaussian_points_xyz.detach(),
        gaussian._features_dc[filled_gaussian_nearest_indices].detach(),
        gaussian._features_rest[filled_gaussian_nearest_indices].detach(),
        gaussian._scaling[filled_gaussian_nearest_indices].detach(),
        gaussian._rotation[filled_gaussian_nearest_indices].detach(),
        gaussian._opacity[filled_gaussian_nearest_indices].detach(),
        gaussian._normal[filled_gaussian_nearest_indices].detach()
    )
    filled_gaussian.set_scaling(filled_gaussian_nearest_distances)
    filled_gaussian.set_rotation(torch.eye(3, device="cuda")[None].repeat(len(filled_gaussian_points_xyz), 1, 1))
    filled_gaussian.set_opacity(
        torch.ones((filled_gaussian_points_xyz.shape[0], 1), dtype=torch.float, device="cuda"))

    original_gaussian_indices = torch.arange(len(gaussian.get_xyz))
    gaussian.combine_gaussians(filled_gaussian)

    print(f"Internal Filling from {len(original_gaussian_indices)} to {len(gaussian.get_xyz)} pts...")

    return gaussian, filled_gaussian, original_gaussian_indices
