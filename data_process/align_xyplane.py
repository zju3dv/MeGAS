import os

import open3d as o3d
import numpy as np

ply_path = "/home/zesongyang/Projects/3DGS/MPM/PhaseGaussian/log/Glykon/align/ground/internal_filled.ply"

pcld = o3d.io.read_point_cloud(ply_path)

points = np.array(pcld.points)
pcld_bbox = [points.min(0), points.max(0)]

pcld_origin = (pcld_bbox[1] + pcld_bbox[0]) / 2
pcld_origin[-1] = pcld_bbox[0][-1]

pcld.translate(np.array([0.5, 0.5, 0] - pcld_origin))  # 底部对齐

o3d.io.write_point_cloud(os.path.join(os.path.dirname(ply_path), "align_ground.ply"), pcld)
