import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    # 对坐标系进行旋转
    mesh_path = "/home/zesongyang/Projects/3DGS/MPM/PhaseGaussian/assets/Dog.ply"

    rotation_degrees = np.array([0, 0, 90])
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix = Rotation.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
    rotation_matrix_4x4[:3, :3] = rotation_matrix

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.scale(1 / mesh.get_max_bound().max(), center=mesh.get_center())
    mesh.translate(-mesh.get_center())

    mesh.transform(rotation_matrix_4x4)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0., 0., 0.])

    o3d.visualization.draw_geometries([mesh, coord_frame])

    save_mesh_file = mesh_path.split(".")[0] + "_align." + mesh_path.split(".")[1]
    o3d.io.write_triangle_mesh(save_mesh_file, mesh)
