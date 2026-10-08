import torch


def skew(v):
    """Cross-product matrices [N, 3, 3] of vectors [N, 3]."""
    vx, vy, vz = v.unbind(-1)
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([O, -vz, vy], dim=-1),
        torch.stack([vz, O, -vx], dim=-1),
        torch.stack([-vy, vx, O], dim=-1),
    ], dim=-2)


def normals_rotmat_from_B_to_A(B, A, eps=1e-8):
    """
    Minimal rotations [N, 3, 3] mapping unit vectors B onto A (Rodrigues' formula),
    R = I + K + K^2 (1 - cos) / sin^2 with K = [B x A]_x.
    The unnormalized axis is used so that the formula stays stable for nearly parallel vectors.
    """
    B = torch.nn.functional.normalize(B, dim=-1)
    A = torch.nn.functional.normalize(A, dim=-1)

    v = torch.cross(B, A, dim=-1)
    s = torch.linalg.norm(v, dim=-1, keepdim=True)
    c = torch.sum(B * A, dim=-1, keepdim=True)

    I = torch.eye(3, device=B.device, dtype=B.dtype).expand(B.shape[0], 3, 3)

    K = skew(v)
    coef = (1.0 - c) / (s * s + eps)
    return I + K + K @ K * coef.unsqueeze(-1)
