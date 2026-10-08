import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from mpm_utils.warp_utils import launch_sync


@wp.kernel
def compute_rimls_kernel(points: wp.array(dtype=wp.vec3),
                         points_normals: wp.array(dtype=wp.vec3),

                         query_points: wp.array(dtype=wp.vec3),
                         nearest_indices: wp.array(dtype=int),  # [N * K]
                         h_qr: wp.array(dtype=float),  # [N] per-query kernel bandwidth

                         knn_num: int,
                         min_refit_iters: int,
                         max_refit_iters: int,
                         sigma_n: float,
                         refitting_threshold: float,

                         out_correct_points: wp.array(dtype=wp.vec3),
                         out_potential: wp.array(dtype=float),
                         out_gradient: wp.array(dtype=wp.vec3)
                         ):
    """
    Robust implicit moving least squares (RIMLS, Oztireli et al. 2009) evaluated at query points.
    For each query x and its K neighbors {p_j, n_j}, with f_j = (x - p_j) . n_j:

        w_j     = phi(||x - p_j||) * exp(-||n_j - grad f||^2 / sigma_n^2)   (normal term from iteration 1 on)
        f       = sum_j w_j f_j / sum_j w_j
        grad f  = (sum_j grad(w_j) f_j - f * sum_j grad(w_j) + sum_j w_j n_j) / sum_j w_j

    with a Gaussian spatial kernel phi(r) = exp(-r^2 / h^2). The query is projected onto the
    zero level set as x - f * grad f, and grad f is returned as the refined surface normal.
    """
    tid = wp.tid()

    xq = query_points[tid]
    h = h_qr[tid]
    hh = h * h
    inv_hh = 1.0 / hh

    grad = wp.vec3(0.0, 0.0, 0.0)
    potential = float(0.0)
    for it in range(max_refit_iters):
        prev_grad = grad

        sum_w = float(0.0)
        sum_wf = float(0.0)
        sum_grad_w = wp.vec3(0.0, 0.0, 0.0)
        sum_grad_wf = wp.vec3(0.0, 0.0, 0.0)
        sum_wn = wp.vec3(0.0, 0.0, 0.0)

        base_id = tid * knn_num
        for k in range(knn_num):
            idx = nearest_indices[base_id + k]

            pj = points[idx]
            nj = points_normals[idx]

            diff = xq - pj
            d2 = wp.dot(diff, diff)

            fx = wp.dot(diff, nj)

            w = wp.exp(-d2 * inv_hh)
            if it > 0:
                ndiff = nj - prev_grad
                w = w * wp.exp(-wp.dot(ndiff, ndiff) / (sigma_n * sigma_n))

            # The spatial gradient must use the refitted weight
            grad_w = (-2.0) * diff * inv_hh * w

            sum_w += w
            sum_grad_w = sum_grad_w + grad_w
            sum_wf += w * fx
            sum_grad_wf = sum_grad_wf + grad_w * fx
            sum_wn = sum_wn + nj * w

        denom = sum_w + 1e-8
        potential = sum_wf / denom
        grad = (sum_grad_wf - sum_grad_w * potential + sum_wn) / denom

        if it >= min_refit_iters:
            dg = grad - prev_grad
            nrm = wp.sqrt(wp.dot(dg, dg))
            if nrm <= refitting_threshold:
                break

    out_potential[tid] = potential
    out_gradient[tid] = grad
    out_correct_points[tid] = xq - potential * grad


def compute_mls_surface(refer_points: np.ndarray,
                        refer_normals: np.ndarray,
                        query_points: np.ndarray,
                        knn_num=12,
                        min_refitting_iters=-1,
                        max_refitting_iters=3,
                        sigma_n=0.8,
                        refitting_threshold=1e-3,
                        return_tensor=False
                        ):
    """
    Fit an RIMLS surface to oriented reference points and evaluate it at the query points.
    The kernel bandwidth of each query is the mean distance to its K nearest references.

    :return: projected points [N, 3], implicit potentials [N], gradients / normals [N, 3]
    """
    dev = wp.get_device("cuda")
    query_tree = cKDTree(refer_points)
    dists, nearest_indices = query_tree.query(query_points, k=knn_num, workers=-1)
    h_qr = dists.mean(-1)

    points_wp = wp.array(refer_points.astype(np.float32), dtype=wp.vec3, device=dev)
    points_normals_wp = wp.array(refer_normals.astype(np.float32), dtype=wp.vec3, device=dev)

    query_points_wp = wp.array(query_points.astype(np.float32), dtype=wp.vec3, device=dev)
    nearest_indices_wp = wp.array(nearest_indices.reshape(-1).astype(np.int32), dtype=int, device=dev)
    h_qr_wp = wp.array(h_qr.astype(np.float32), dtype=float, device=dev)

    out_correct_points_wp = wp.zeros(len(query_points), dtype=wp.vec3, device=dev)
    out_potential_wp = wp.zeros(len(query_points), dtype=float, device=dev)
    out_gradient_wp = wp.zeros(len(query_points), dtype=wp.vec3, device=dev)

    launch_sync(
        kernel=compute_rimls_kernel,
        dim=len(query_points),
        inputs=[
            points_wp,
            points_normals_wp,
            query_points_wp,
            nearest_indices_wp,
            h_qr_wp,
            knn_num,
            min_refitting_iters,
            max_refitting_iters,
            sigma_n,
            refitting_threshold,
            out_correct_points_wp,
            out_potential_wp,
            out_gradient_wp,
        ],
        device=dev
    )

    if return_tensor:
        potentials = wp.to_torch(out_potential_wp)
        gradients = wp.to_torch(out_gradient_wp)
        correct_points = wp.to_torch(out_correct_points_wp)
    else:
        potentials = out_potential_wp.numpy().astype(np.float32)
        gradients = out_gradient_wp.numpy().astype(np.float32)
        correct_points = out_correct_points_wp.numpy().astype(np.float32)

    return correct_points, potentials, gradients
