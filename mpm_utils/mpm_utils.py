import torch
import warp as wp

from mpm_modules.MPMCollider import *
from mpm_modules.MPMStruct import MPMParticle_State, MPMGrid_State
from mpm_utils.kirchhoff_stress_utils import compute_stress
from mpm_utils.return_mapping_utils import foam_herschel_return_mapping, foam_stvk_return_mapping
from mpm_utils.warp_utils import *


@wp.kernel
def zero_grid(mpm_grid_state: MPMGrid_State):
    i, j, k = wp.tid()
    mpm_grid_state.grid_v[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    mpm_grid_state.grid_m[i, j, k] = 0.0
    mpm_grid_state.grid_thermal[i, j, k] = 0.0
    mpm_grid_state.grid_rhs[i, j, k] = 0.0
    mpm_grid_state.cg_r[i, j, k] = 0.0
    mpm_grid_state.cg_p[i, j, k] = 0.0
    mpm_grid_state.cg_Ap[i, j, k] = 0.0
    mpm_grid_state.cg_y[i, j, k] = 0.0


@wp.kernel
def p2g_apic(mpm_particle_state: MPMParticle_State, mpm_grid_state: MPMGrid_State, dt: float, use_damping: bool):
    """
    APIC particle-to-grid transfer of momentum and temperature with quadratic B-spline weights w_ip:

        m_i         = sum_p w_ip * m_p
        (m v)_i     = sum_p w_ip * m_p * (v_p + C_p (x_i - x_p)) - dt * V_p * tau_p * grad(w_ip)
        (m T)_i     = sum_p w_ip * m_p * (T_p + grad(T_p) . (x_i - x_p))
    """
    p = wp.tid()

    particle_stress = compute_stress(mpm_particle_state, p, dt)

    grid_pos = mpm_particle_state.particle_x[p] * mpm_grid_state.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)

    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(wp.cw_mul(wa, wa) * 0.5,
                 wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                 wp.cw_mul(wc, wc) * 0.5, )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * mpm_grid_state.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                weight = w[0, i] * w[1, j] * w[2, k]
                dweight = wp.vec3(dw[0, i] * w[1, j] * w[2, k],
                                  w[0, i] * dw[1, j] * w[2, k],
                                  w[0, i] * w[1, j] * dw[2, k]) * mpm_grid_state.inv_dx
                C = mpm_particle_state.particle_C[p]

                # RPIC damping: attenuate the rotational part of the affine matrix
                if use_damping:
                    rpic_damping = 0.5
                    C = (1.0 - rpic_damping) * C + rpic_damping / 2.0 * (C - wp.transpose(C))

                elastic_force = -mpm_particle_state.particle_vol[p] * particle_stress * dweight
                v_in_add = weight * mpm_particle_state.particle_mass[p] * (mpm_particle_state.particle_v[p] + C * dpos) \
                           + dt * elastic_force

                thermal_in_add = weight * mpm_particle_state.particle_mass[p] * (
                        mpm_particle_state.particle_thermal[p] + \
                        wp.dot(dpos, mpm_particle_state.delta_particle_thermal[p])
                )

                wp.atomic_add(mpm_grid_state.grid_v, ix, iy, iz, v_in_add)
                wp.atomic_add(mpm_grid_state.grid_thermal, ix, iy, iz, thermal_in_add)
                wp.atomic_add(mpm_grid_state.grid_m, ix, iy, iz, weight * mpm_particle_state.particle_mass[p])


@wp.kernel
def compute_correct_gradient(mpm_particle_state: MPMParticle_State, dt: float):
    """Plastic return mapping for visco-plastic (foam) particles."""
    p = wp.tid()

    if mpm_particle_state.phase_state[p] == 0:
        material_state = mpm_particle_state.solid_material_state
    elif mpm_particle_state.phase_state[p] == 1:
        material_state = mpm_particle_state.liquid_material_state

    if material_state.material_type == 5:
        if material_state.elastic_method_type[p] == 1:
            mpm_particle_state.particle_F[p] = \
                foam_stvk_return_mapping(mpm_particle_state.particle_F[p],
                                         material_state.mu[p],
                                         material_state.viscosity,
                                         material_state.shear_ratio,
                                         material_state.yield_threshold,
                                         dt)

        elif material_state.elastic_method_type[p] == 3:
            mpm_particle_state.particle_be_hat[p] = \
                foam_herschel_return_mapping(mpm_particle_state.particle_be_hat[p],
                                             material_state.mu[p],
                                             material_state.viscosity,
                                             material_state.shear_ratio,
                                             material_state.yield_threshold,
                                             dt)


@wp.kernel
def grid_update(mpm_grid_state: MPMGrid_State,
                gravity: wp.vec3, dt: float,
                colliders: wp.array(dtype=Colliders),
                use_damping: bool,
                ):
    """Normalize nodal momentum and temperature by mass, apply gravity, damping and collisions."""
    grid_x, grid_y, grid_z = wp.tid()

    if mpm_grid_state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        mpm_grid_state.grid_v[grid_x, grid_y, grid_z] = mpm_grid_state.grid_v[grid_x, grid_y, grid_z] * (
                1.0 / mpm_grid_state.grid_m[grid_x, grid_y, grid_z])

        mpm_grid_state.grid_v[grid_x, grid_y, grid_z] = mpm_grid_state.grid_v[
                                                            grid_x, grid_y, grid_z] + dt * gravity
        if use_damping:
            mpm_grid_state.grid_v[grid_x, grid_y, grid_z] = 0.9995 * mpm_grid_state.grid_v[grid_x, grid_y, grid_z]

        mpm_grid_state.grid_thermal[grid_x, grid_y, grid_z] = mpm_grid_state.grid_thermal[grid_x, grid_y, grid_z] * (
                1.0 / mpm_grid_state.grid_m[grid_x, grid_y, grid_z])

        for collider_index in range(colliders.shape[0]):
            collider = colliders[collider_index]
            offset = wp.vec3(
                float(grid_x) * mpm_grid_state.dx - collider.point[0],
                float(grid_y) * mpm_grid_state.dx - collider.point[1],
                float(grid_z) * mpm_grid_state.dx - collider.point[2]
            )
            dotproduct = wp.dot(offset, collider.normal)
            if dotproduct <= 0.0:
                if collider.collider_type == 0:  # sticky
                    mpm_grid_state.grid_v[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def grid_build_rhs_advect(mpm_grid_state: MPMGrid_State, dt: float):
    """
    Right-hand side of the implicit heat equation with explicit first-order upwind advection:

        b_i = T_i - dt * (v_i . grad T_i)

    Only neighbors carrying mass contribute to the finite differences; otherwise the center
    value is used, which corresponds to a zero-flux (Neumann) condition at the free surface.
    """
    i, j, k = wp.tid()

    N = mpm_grid_state.n_grid
    dx = mpm_grid_state.dx
    inv_dx = 1.0 / dx

    Tc = mpm_grid_state.grid_thermal[i, j, k]
    m = mpm_grid_state.grid_m[i, j, k]
    if m <= 1.0e-15:
        mpm_grid_state.grid_rhs[i, j, k] = Tc
        return

    T_ip1 = Tc
    if i + 1 < N and mpm_grid_state.grid_m[i + 1, j, k] > 1.0e-15:
        T_ip1 = mpm_grid_state.grid_thermal[i + 1, j, k]
    T_im1 = Tc
    if i - 1 >= 0 and mpm_grid_state.grid_m[i - 1, j, k] > 1.0e-15:
        T_im1 = mpm_grid_state.grid_thermal[i - 1, j, k]
    T_jp1 = Tc
    if j + 1 < N and mpm_grid_state.grid_m[i, j + 1, k] > 1.0e-15:
        T_jp1 = mpm_grid_state.grid_thermal[i, j + 1, k]
    T_jm1 = Tc
    if j - 1 >= 0 and mpm_grid_state.grid_m[i, j - 1, k] > 1.0e-15:
        T_jm1 = mpm_grid_state.grid_thermal[i, j - 1, k]
    T_kp1 = Tc
    if k + 1 < N and mpm_grid_state.grid_m[i, j, k + 1] > 1.0e-15:
        T_kp1 = mpm_grid_state.grid_thermal[i, j, k + 1]
    T_km1 = Tc
    if k - 1 >= 0 and mpm_grid_state.grid_m[i, j, k - 1] > 1.0e-15:
        T_km1 = mpm_grid_state.grid_thermal[i, j, k - 1]

    v = mpm_grid_state.grid_v[i, j, k]

    if v[0] > 0.0:
        dTx = (Tc - T_im1) * inv_dx
    else:
        dTx = (T_ip1 - Tc) * inv_dx
    if v[1] > 0.0:
        dTy = (Tc - T_jm1) * inv_dx
    else:
        dTy = (T_jp1 - Tc) * inv_dx
    if v[2] > 0.0:
        dTz = (Tc - T_km1) * inv_dx
    else:
        dTz = (T_kp1 - Tc) * inv_dx

    adv = v[0] * dTx + v[1] * dTy + v[2] * dTz
    mpm_grid_state.grid_rhs[i, j, k] = Tc - dt * adv


@wp.kernel
def grid_update_thermal_bc(mpm_grid_state: MPMGrid_State,
                           thermal_bound: Plane_Thermal_Boundary
                           ):
    grid_x, grid_y, grid_z = wp.tid()

    offset = wp.vec3(
        float(grid_x) * mpm_grid_state.dx - thermal_bound.point[0],
        float(grid_y) * mpm_grid_state.dx - thermal_bound.point[1],
        float(grid_z) * mpm_grid_state.dx - thermal_bound.point[2]
    )
    dotproduct = wp.dot(offset, thermal_bound.normal)
    if dotproduct <= 0.0:
        mpm_grid_state.grid_thermal[grid_x, grid_y, grid_z] = thermal_bound.thermal


@wp.kernel
def g2p(mpm_particle_state: MPMParticle_State,
        mpm_grid_state: MPMGrid_State,
        dt: float):
    """
    APIC grid-to-particle transfer and deformation update:

        v_p      = sum_i w_ip * v_i
        C_p      = 4 / dx^2 * sum_i w_ip * v_i (x_i - x_p)^T
        T_p      = sum_i w_ip * T_i,      grad(T_p) = sum_i T_i * grad(w_ip)
        x_p     += dt * v_p
        F_p     <- (I + dt * grad(v_p)) F_p
        b_e_hat <- f_hat b_e_hat f_hat^T, with f_hat = det(f)^(-1/3) f the isochoric increment
    """
    p = wp.tid()

    grid_pos = mpm_particle_state.particle_x[p] * mpm_grid_state.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(wp.cw_mul(wa, wa) * 0.5,
                 wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                 wp.cw_mul(wc, wc) * 0.5)
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

    new_particle_v = wp.vec3(0.0, 0.0, 0.0)
    new_particle_thermal = 0.0
    new_particle_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    new_delta_particle_v = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    new_delta_particle_thermal = wp.vec3(0.0, 0.0, 0.0)

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * mpm_grid_state.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k

                weight = w[0, i] * w[1, j] * w[2, k]
                dweight = wp.vec3(dw[0, i] * w[1, j] * w[2, k],
                                  w[0, i] * dw[1, j] * w[2, k],
                                  w[0, i] * w[1, j] * dw[2, k]) * mpm_grid_state.inv_dx

                new_particle_v = new_particle_v + mpm_grid_state.grid_v[ix, iy, iz] * weight
                new_particle_thermal = new_particle_thermal + mpm_grid_state.grid_thermal[ix, iy, iz] * weight
                new_particle_C = new_particle_C + wp.outer(mpm_grid_state.grid_v[ix, iy, iz], dpos) * weight * \
                                 (mpm_grid_state.inv_dx * mpm_grid_state.inv_dx) * 4.0
                new_delta_particle_v = new_delta_particle_v + wp.outer(mpm_grid_state.grid_v[ix, iy, iz],
                                                                       dweight)
                new_delta_particle_thermal = new_delta_particle_thermal + mpm_grid_state.grid_thermal[
                    ix, iy, iz] * dweight

    mpm_particle_state.delta_particle_thermal[p] = new_delta_particle_thermal
    mpm_particle_state.particle_thermal[p] = new_particle_thermal

    mpm_particle_state.particle_v[p] = new_particle_v
    mpm_particle_state.particle_x[p] = mpm_particle_state.particle_x[p] + dt * new_particle_v
    mpm_particle_state.particle_C[p] = new_particle_C

    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    f_update = I33 + new_delta_particle_v * dt
    f_update_hat = wp.pow(wp.determinant(f_update), -1.0 / 3.0) * f_update

    mpm_particle_state.particle_F[p] = f_update * mpm_particle_state.particle_F[p]
    mpm_particle_state.particle_be_hat[p] = f_update_hat * mpm_particle_state.particle_be_hat[p] * \
                                            wp.transpose(f_update_hat)


@wp.kernel
def particle_phase_update(mpm_particle_state: MPMParticle_State):
    """Solid-liquid phase transition driven by the melting temperature."""
    p = wp.tid()
    if mpm_particle_state.particle_thermal[p] > mpm_particle_state.thermal_phase_threshold:
        mpm_particle_state.phase_state[p] = 1
    else:
        mpm_particle_state.phase_state[p] = 0


@wp.kernel
def particle_update_thermal_bc(mpm_particle_state: MPMParticle_State,
                               sphere_bound: Particle_Thermal_Boundary):
    p = wp.tid()
    if sphere_bound.particle_mask[p] == 1:
        mpm_particle_state.particle_thermal[p] = sphere_bound.thermal


@wp.kernel
def apply_A(
        mpm_grid_state: MPMGrid_State,
        dt: float,
        x: wp.array3d(dtype=float),
        y: wp.array3d(dtype=float)
):
    """
    Matrix-free product y = A x for the implicit diffusion operator A = I - alpha * dt * Laplacian.
    With a = alpha * dt / dx^2 and N(i) the set of neighbors carrying mass:

        y_i = (1 + a * |N(i)|) * x_i - a * sum_{j in N(i)} x_j

    Empty nodes use A = I so that the system stays well-conditioned.
    """
    i, j, k = wp.tid()

    N = mpm_grid_state.n_grid
    dx = mpm_grid_state.dx
    alpha = mpm_grid_state.heat_diff_ratio * dt / (dx * dx)

    if mpm_grid_state.grid_m[i, j, k] <= 1.0e-15:
        y[i, j, k] = x[i, j, k]
        return

    xc = x[i, j, k]
    sumN = float(0.0)
    cnt = float(0.0)

    if i + 1 < N and (mpm_grid_state.grid_m[i + 1, j, k] > 1.0e-15):
        sumN += x[i + 1, j, k]
        cnt += 1.0
    if i - 1 >= 0 and (mpm_grid_state.grid_m[i - 1, j, k] > 1.0e-15):
        sumN += x[i - 1, j, k]
        cnt += 1.0
    if j + 1 < N and (mpm_grid_state.grid_m[i, j + 1, k] > 1.0e-15):
        sumN += x[i, j + 1, k]
        cnt += 1.0
    if j - 1 >= 0 and (mpm_grid_state.grid_m[i, j - 1, k] > 1.0e-15):
        sumN += x[i, j - 1, k]
        cnt += 1.0
    if k + 1 < N and (mpm_grid_state.grid_m[i, j, k + 1] > 1.0e-15):
        sumN += x[i, j, k + 1]
        cnt += 1.0
    if k - 1 >= 0 and (mpm_grid_state.grid_m[i, j, k - 1] > 1.0e-15):
        sumN += x[i, j, k - 1]
        cnt += 1.0

    y[i, j, k] = (1.0 + alpha * cnt) * xc - alpha * sumN


@wp.kernel
def vec_copy(dst: wp.array3d(dtype=float), src: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    dst[i, j, k] = src[i, j, k]


@wp.kernel
def vec_axpy(y: wp.array3d(dtype=float), a: float, x: wp.array3d(dtype=float)):
    # y <- y + a * x
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + a * x[i, j, k]


@wp.kernel
def vec_scale(y: wp.array3d(dtype=float), a: float):
    i, j, k = wp.tid()
    y[i, j, k] = a * y[i, j, k]


@wp.kernel
def vec_sub(dst: wp.array3d(dtype=float), a: wp.array3d(dtype=float), b: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    dst[i, j, k] = a[i, j, k] - b[i, j, k]


@wp.kernel
def vec_dot(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    # out[0] += sum(x * y)
    i, j, k = wp.tid()
    wp.atomic_add(out, 0, x[i, j, k] * y[i, j, k])


@wp.kernel
def clear_array(arr: wp.array(dtype=float)):
    i = wp.tid()
    arr[i] = 0.0


def vec_to_mat33(vec_tensor):
    """Upper-triangular covariance [N, 6] -> symmetric matrix [N, 3, 3]."""
    a, b, c, d, e, f = [vec_tensor[:, i] for i in range(6)]

    mats = torch.stack([
        torch.stack([a, b, c], dim=-1),
        torch.stack([b, d, e], dim=-1),
        torch.stack([c, e, f], dim=-1),
    ], dim=1)
    return mats


def mat33_to_vec(mat33_tensor):
    """Symmetric matrix [N, 3, 3] -> upper-triangular covariance [N, 6]."""
    a = mat33_tensor[:, 0, 0]
    b = mat33_tensor[:, 0, 1]
    c = mat33_tensor[:, 0, 2]
    d = mat33_tensor[:, 1, 1]
    e = mat33_tensor[:, 1, 2]
    f = mat33_tensor[:, 2, 2]
    return torch.stack([a, b, c, d, e, f], dim=-1)


@wp.kernel
def update_cov3D(mpm_particle_state: MPMParticle_State):
    """
    Deform the Gaussian kernel with the deformation gradient, Sigma = F Sigma_0 F^T, and
    eigen-decompose it as Sigma = R diag(s^2) R^T to recover per-axis rotation and scale.
    The eigenvector order is not aligned with the original Gaussian axes; this is resolved
    on the PyTorch side by axis matching.
    """
    p = wp.tid()

    init_cov3D_mat = mpm_particle_state.particle_init_cov3D[p]
    F = mpm_particle_state.particle_F[p]

    update_cov3D_mat = F @ init_cov3D_mat @ wp.transpose(F)
    update_cov3D_mat = 0.5 * (update_cov3D_mat + wp.transpose(update_cov3D_mat))
    mpm_particle_state.particle_cov3D[p] = update_cov3D_mat

    s = wp.vec3()
    R = wp.mat33()
    wp.eig3(update_cov3D_mat, R, s)

    scale = wp.vec3(wp.sqrt(wp.max(s[0], 0.0)),
                    wp.sqrt(wp.max(s[1], 0.0)),
                    wp.sqrt(wp.max(s[2], 0.0)))

    mpm_particle_state.particle_R[p] = R
    mpm_particle_state.particle_Scale[p] = scale
