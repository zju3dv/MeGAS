import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation

from tqdm import tqdm

import sys

sys.path.append("../gsplat")

from gsplat.scene.gaussian_model import GaussianModel
import torch


def create_plane_from_point_and_normal(point, normal, size=1.0, color=None):
    # 根据给定法向量生成与法向量垂直的两个基向量
    z_axis = normal / np.linalg.norm(normal)  # 法向量单位化

    # 选择任意与法向量垂直的方向作为基向量
    if abs(z_axis[0]) > 0.5:
        x_axis = np.array([0, 1, 0])
    else:
        x_axis = np.array([1, 0, 0])

    # 计算第二个基向量（与法向量和x轴垂直）
    y_axis = np.cross(z_axis, x_axis)

    # 创建平面上的四个顶点
    p0 = point - (size / 2) * x_axis - (size / 2) * y_axis
    p1 = point + (size / 2) * x_axis - (size / 2) * y_axis
    p2 = point + (size / 2) * x_axis + (size / 2) * y_axis
    p3 = point - (size / 2) * x_axis + (size / 2) * y_axis

    # 定义平面的四个顶点和两组三角形
    vertices = np.array([p0, p1, p2, p3])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])

    # 创建 TriangleMesh 对象并设置顶点和三角形
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(triangles)

    if color is not None:
        plane.paint_uniform_color(color)
    plane.compute_vertex_normals()

    return plane


# 将colmap & gaussian & mesh全部align

def align_gaussian(gaussian: GaussianModel, ground_height=0.1, gs_bbox_bottom=None):  # lego 0.4072
    gaussian._xyz = gaussian._xyz + torch.tensor([0.5, 0.5, 0.5]).cuda()

    gs_xyz = gaussian.get_xyz
    if gs_bbox_bottom is None:
        gs_bbox_bottom = gs_xyz.min(0)[0][-1]

        gs_bbox_bottom = gs_bbox_bottom - 0.002
        print(gs_bbox_bottom)
    else:
        print(f"Using provided gs_bbox_bottom: {gs_bbox_bottom}")

    translate_z = ground_height - gs_bbox_bottom
    gaussian._xyz = gaussian._xyz + torch.tensor([0.0, 0.0, translate_z]).cuda()

    return gaussian, translate_z


if __name__ == "__main__":
    # 处理3DGS
    '''
    blue_rabbit  cheese           dog        sculpture      sugar_house candy_house  chocolate_bunny  metal_owl  snow_elephant  toy_bulldozer
    '''
    for object_item in ["360_kitchen/3dgs"]:
        # ['blue_rabbit', 'cheese', 'dog', 'sculpture', 'sugar_house', 'candy_house', 'chocolate_bunny', 'metal_owl', 'snow_elephant', 'toy_bulldozer']:
        # object_item = "cheese"
        # /home/zesongyang/tmp_Projects/PhaseGaussian/gsplat/output/sketchfab/sugar_house/3dgs/point_cloud/iteration_30000/point_cloud.ply
        # /home/zesongyang/tmp_Projects/Uniform2DGS/output/sketchfab/sugar_house/prune/point_cloud/iteration_30000/point_cloud_3dgs.ply
        gs_model_path = "/home/zesongyang/tmp_Projects/Uniform2DGS/output/tnt/Ignatius/mls/point_cloud/latest/point_cloud_3dgs.ply"
        scale_ratio = 0.7

        save_dir = os.path.join("../log/gaussians", object_item, "align")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            gaussians = GaussianModel(sh_degree=3)
            gaussians.load_ply(gs_model_path)

            aligned_gaussian, translate_z = align_gaussian(gaussians, ground_height=0.1)

            aligned_gaussian.save_ply(os.path.join(save_dir, "align_gaussian.ply"))
            torch.save(translate_z, os.path.join(save_dir, "align_translate_z.pt"))

            pointcloud = o3d.io.read_point_cloud(os.path.join(save_dir, "align_gaussian_color.ply"))

            bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([0, 0, 0]), np.array([1, 1, 1]))
            bbox.color = np.array([1, 0, 0])
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])

            plane = create_plane_from_point_and_normal(np.array([0.5, 0.5, 0.1]), np.array([0, 0, 1]), size=1.0,
                                                       color=[0.6, 0.3, 0.5])

            o3d.visualization.draw_geometries([pointcloud, bbox, coordinate, plane])
