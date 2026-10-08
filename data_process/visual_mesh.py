import open3d as o3d
import numpy as np

mesh_file = "/home/zesongyang/Projects/datasets/sketchfab/renders/dog/mesh.ply"

if True:
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
else:
    mesh = o3d.io.read_point_cloud(mesh_file)

# 将mesh归一化
if False:
    mesh.scale(1 / mesh.get_max_bound().max(), center=mesh.get_center())
    # 将mesh平移到原点
    mesh.translate(-mesh.get_center())

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0., 0., 0.])

bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
bbox.color = np.array([1, 0, 0])

o3d.visualization.draw_geometries([mesh, coord_frame, bbox])
