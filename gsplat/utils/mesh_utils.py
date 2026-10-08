import torch


def normal2rotation(n, return_mat=False):
    """
    Rotation whose third column is the given normal, i.e. a frame with the normal as the local z-axis.
    Adapted from Gaussian Surfels (https://github.com/turandai/gaussian_surfels).

    :param n: normals [N, 3]
    :return: quaternions [N, 4], or rotation matrices [N, 3, 3] if return_mat
    """
    n = n / (torch.linalg.norm(n, keepdim=True, dim=-1) + 1e-5)

    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0)

    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    R = torch.stack([R0, R1, n], -1)
    if return_mat:
        return R
    return rotmat2quaternion(R)


def rotmat2quaternion(R, normalize=False):
    """Rotation matrices [N, 3, 3] -> quaternions (w, x, y, z) [N, 4]."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr.clamp(-1, 100)) / 2

    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)

    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q
