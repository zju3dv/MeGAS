import os
import glob

import open3d as o3d

target_triangle_count = 50000

for mesh_file in glob.glob("../assets/complex/*"):
    save_mesh_file = os.path.join("../assets", os.path.basename(mesh_file))
    if os.path.exists(save_mesh_file):
        continue

    print("### Deal with mesh:", mesh_file, "###")

    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # 简化 mesh
    simplified_mesh = mesh.simplify_quadric_decimation(target_triangle_count)
    simplified_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([simplified_mesh])

    o3d.io.write_triangle_mesh(save_mesh_file, simplified_mesh)
