import numpy as np


def get_viewmatrix(look_from, look_at, up=np.array([0, 0, 1])):
    """
    Camera-to-world matrix [4, 4] of a camera at `look_from` looking at `look_at`,
    in the OpenCV convention (x right, y down, z forward).
    """
    look_dir = look_at - look_from
    look_dir = look_dir / np.linalg.norm(look_dir)
    right = np.cross(look_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, look_dir)
    up = up / np.linalg.norm(up)
    c2w = np.eye(4)
    c2w[:3] = np.stack([right, -up, look_dir, look_from], axis=1)
    return c2w
