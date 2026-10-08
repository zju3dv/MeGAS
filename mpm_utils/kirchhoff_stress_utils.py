import warp as wp

from mpm_modules.MPMStruct import MPMParticle_State


@wp.func
def compute_stress(mpm_particle_state: MPMParticle_State,
                   p: int,
                   dt: float):
    """
    Kirchhoff stress tau of particle p. The constitutive model is selected by the particle's
    phase (solid / liquid) and by the elastic method of the corresponding material.
    """
    if mpm_particle_state.phase_state[p] == 0:
        material_state = mpm_particle_state.solid_material_state
    elif mpm_particle_state.phase_state[p] == 1:
        material_state = mpm_particle_state.liquid_material_state

    if material_state.material_type == 0:  # jelly
        if material_state.elastic_method_type[p] == 0:
            stress = kirchoff_stress_FCR(mpm_particle_state.particle_F[p],
                                         material_state.mu[p],
                                         material_state.lam[p])
        elif material_state.elastic_method_type[p] == 1:
            stress = kirchoff_stress_StVK(mpm_particle_state.particle_F[p],
                                          material_state.mu[p],
                                          material_state.lam[p])
        elif material_state.elastic_method_type[p] == 2:
            stress = kirchoff_stress_neo_hookean(mpm_particle_state.particle_F[p],
                                                 material_state.mu[p])
        elif material_state.elastic_method_type[p] == 3:
            stress = kirchoff_stress_herschel_bulkley(mpm_particle_state.particle_F[p],
                                                      mpm_particle_state.particle_be_hat[p],
                                                      material_state.mu[p],
                                                      material_state.bulk[p])

    elif material_state.material_type == 5:  # foam
        if material_state.elastic_method_type[p] == 1:
            stress = kirchoff_stress_StVK(mpm_particle_state.particle_F[p],
                                          material_state.mu[p],
                                          material_state.lam[p])
        elif material_state.elastic_method_type[p] == 3:
            stress = kirchoff_stress_herschel_bulkley(mpm_particle_state.particle_F[p],
                                                      mpm_particle_state.particle_be_hat[p],
                                                      material_state.mu[p],
                                                      material_state.bulk[p])

    stress = (stress + wp.transpose(stress)) / 2.0  # enforce symmetry
    return stress


@wp.func
def kirchoff_stress_FCR(F: wp.mat33,
                        mu: float,
                        lam: float):
    """Fixed corotated model: tau = 2 * mu * (F - R) F^T + lam * J * (J - 1) * I."""
    J = wp.determinant(F)

    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)


@wp.func
def kirchoff_stress_StVK(F: wp.mat33,
                         mu: float, lam: float):
    """St. Venant-Kirchhoff model with Hencky strain: tau = U (2 * mu * eps + lam * tr(eps) * I) V^T F^T."""
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    # Clamp singular values to avoid NaNs under extreme compression
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )
    epsilon = wp.mat33(wp.log(sig[0]), 0.0, 0.0,
                       0.0, wp.log(sig[1]), 0.0,
                       0.0, 0.0, wp.log(sig[2]))

    tmp = 2.0 * mu * epsilon + lam * (epsilon[0, 0] + epsilon[1, 1] + epsilon[2, 2]
                                      ) * wp.mat33(1.0, 0.0, 0.0,
                                                   0.0, 1.0, 0.0,
                                                   0.0, 0.0, 1.0)
    return U * tmp * wp.transpose(V) * wp.transpose(F)


@wp.func
def kirchoff_stress_herschel_bulkley(F: wp.mat33,
                                     b_elastic_hat: wp.mat33,
                                     mu: float,
                                     bulk: float):
    """Herschel-Bulkley elasticity: tau = kappa / 2 * (J^2 - 1) * I + mu * dev(b_e_hat)."""
    J = wp.determinant(F)
    trace_b_elastic_hat = b_elastic_hat[0, 0] + b_elastic_hat[1, 1] + b_elastic_hat[2, 2]
    dev_b_elastic_hat = b_elastic_hat - wp.mat33(trace_b_elastic_hat / 3.0, 0.0, 0.0,
                                                 0.0, trace_b_elastic_hat / 3.0, 0.0,
                                                 0.0, 0.0, trace_b_elastic_hat / 3.0)

    stress_volume = bulk / 2.0 * (J ** 2.0 - 1.0)
    stress_det = mu * dev_b_elastic_hat

    return wp.diag(wp.vec3(stress_volume, stress_volume, stress_volume)) + stress_det


@wp.func
def kirchoff_stress_neo_hookean(F: wp.mat33, mu: float):
    J = wp.determinant(F)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return mu * (F * wp.transpose(F) - id) + wp.log(J) * id
