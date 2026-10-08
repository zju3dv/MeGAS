import numpy as np
import os
import open3d as o3d
import argparse


def align_coord(mesh, scale_ratio, init_state="drop", ground_height=0.1):
    # 让mesh对齐到[0, 1]
    mesh_vertex = np.array(mesh.vertices)
    mesh_bbox = [mesh_vertex.min(0), mesh_vertex.max(0)]
    mesh_scale = (mesh_bbox[1] - mesh_bbox[0]).max() / scale_ratio
    mesh.scale(1 / mesh_scale, center=(mesh_bbox[1] + mesh_bbox[0]) / 2)

    mesh_vertex = np.array(mesh.vertices)
    if init_state == "drop":
        mesh_bbox = [mesh_vertex.min(0), mesh_vertex.max(0)]
        mesh.translate(np.array([0.5, 0.5, 0.5] - (mesh_bbox[1] + mesh_bbox[0]) / 2))
    elif init_state == "ground":  # 贴合底部
        mesh_bbox = [mesh_vertex.min(0), mesh_vertex.max(0)]
        mesh_origin = (mesh_bbox[1] + mesh_bbox[0]) / 2
        mesh_origin[-1] = mesh_bbox[0][-1]

        mesh.translate(np.array([0.5, 0.5, ground_height] - mesh_origin))  # 底部对齐

    return mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('--mesh_file', type=str, default="../assets/Warrior.obj")
    parser.add_argument('--init_state', type=str, default="ground", help="drop,ground")
    args = parser.parse_args()

    mesh_file = args.mesh_file
    init_state = args.init_state  # ["drop","ground"]

    save_dir = os.path.join("../log/mesh", os.path.basename(mesh_file).split(".")[0], "align", init_state)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "align_mesh.ply")
    if os.path.exists(save_path):
        print(f"\033[096m### Saving into {save_path} ###\033[0m")
        # exit(0)

    scale_ratio = 0.7
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    '''
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)

    vis.add_geometry(mesh)
    vis.add_geometry(coords)

    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(800, 800,
                                                         800, 800,
                                                         400, 400)
    camera.extrinsic = np.linalg.inv(np.array([[0, 0, -1, 5],
                                               [1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, 0, 1]]))

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera, True)

    while True:
        vis.poll_events()
        vis.update_renderer()
    '''

    mesh_aligned = align_coord(mesh, scale_ratio, init_state, ground_height=0.1)

    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([0, 0, 0]), np.array([1, 1, 1]))
    bbox.color = np.array([1, 0, 0])
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
    o3d.visualization.draw_geometries([mesh_aligned, bbox, coordinate])

    o3d.io.write_triangle_mesh(save_path, mesh_aligned)
    print(f"\033[096m### Saving into {save_path} ###\033[0m")
