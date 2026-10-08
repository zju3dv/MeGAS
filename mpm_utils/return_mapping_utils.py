import warp as wp


@wp.func
def frobenius_norm(A: wp.mat33) -> float:
    return wp.sqrt(
        A[0, 0] * A[0, 0] + A[0, 1] * A[0, 1] + A[0, 2] * A[0, 2] +
        A[1, 0] * A[1, 0] + A[1, 1] * A[1, 1] + A[1, 2] * A[1, 2] +
        A[2, 0] * A[2, 0] + A[2, 1] * A[2, 1] + A[2, 2] * A[2, 2]
    )


@wp.func
def foam_stvk_return_mapping(particle_F: wp.mat33,
                             mu: float,
                             viscosity: float,
                             shear_ratio: float,
                             yield_stress: float,
                             dt: float):
    """
    Herschel-Bulkley return mapping in the principal Hencky-strain space (StVK elasticity).

        epsilon   = log(sigma),  epsilon_hat = dev(epsilon)
        s_trial   = 2 * mu * epsilon_hat
        mu_hat    = mu / 3 * tr(sigma^2)
        |s_new| solves  eta^(1/n) * (|s_new| - |s_trial|) + 2 * mu_hat * dt * (|s_new| - sqrt(2/3) * tau_Y)^(1/n) = 0
        F_new     = U * exp(s_new / (2 * mu) + tr(epsilon) / 3) * V^T
    """
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(particle_F, U, sig, V)
    sig = wp.vec3(wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01))

    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )

    s_trial = 2.0 * mu * epsilon_hat
    s_trial_norm = wp.length(s_trial)

    yield_cond = s_trial_norm - wp.sqrt(2.0 / 3.0) * yield_stress
    if yield_cond <= 0.0:
        return particle_F
    else:
        mu_hat = mu / 3.0 * (sig[0] * sig[0] + sig[1] * sig[1] + sig[2] * sig[2])
        if shear_ratio == 1.0:
            # Closed-form Bingham solution
            s_trial_update_norm = s_trial_norm - yield_cond / (1.0 + viscosity / (2.0 * mu_hat * dt))
        else:
            # Bisection on the monotone scalar residual
            s_lo = wp.sqrt(2.0 / 3.0) * yield_stress
            s_hi = s_trial_norm
            for _ in range(30):
                s_mid = 0.5 * (s_lo + s_hi)
                delta = wp.max(s_mid - wp.sqrt(2.0 / 3.0) * yield_stress, 0.0)
                g = (wp.pow(viscosity, 1.0 / shear_ratio)) * (s_mid - s_trial_norm) + \
                    2.0 * mu_hat * dt * wp.pow(delta, 1.0 / shear_ratio)
                if g > 0.0:
                    s_hi = s_mid
                else:
                    s_lo = s_mid
                if wp.abs(g / s_trial_norm) < 1e-10:
                    break
            s_trial_update_norm = 0.5 * (s_lo + s_hi)

        s_trial_update = s_trial_update_norm * (s_trial / s_trial_norm)

        epsilon_update_tmp = s_trial_update / (2.0 * mu) + wp.vec3(trace_epsilon / 3.0, trace_epsilon / 3.0,
                                                                   trace_epsilon / 3.0)
        epsilon_update = wp.vec3(wp.exp(epsilon_update_tmp[0]), wp.exp(epsilon_update_tmp[1]),
                                 wp.exp(epsilon_update_tmp[2]))
        return U * (wp.diag(epsilon_update)) * wp.transpose(V)


@wp.func
def foam_herschel_return_mapping(b_elastic_hat: wp.mat33,
                                 mu: float,
                                 viscosity: float,
                                 shear_ratio: float,
                                 yield_threshold: float,
                                 dt: float):
    """
    Herschel-Bulkley return mapping on the isochoric elastic left Cauchy-Green tensor b_e_hat.

        s_trial = mu * dev(b_e_hat),  mu_hat = mu * tr(b_e_hat) / 3
        |s_new| solves  eta^(1/n) * (|s_new| - |s_trial|) + 2 * mu_hat * dt * (|s_new| - sqrt(2/3) * tau_Y)^(1/n) = 0
        b_e_hat <- s_new / mu + tr(b_e_hat) / 3 * I, renormalized to det(b_e_hat) = 1
    """
    trace_b_elastic_hat = b_elastic_hat[0, 0] + b_elastic_hat[1, 1] + b_elastic_hat[2, 2]
    dev_b_elastic_hat = b_elastic_hat - wp.mat33(trace_b_elastic_hat / 3.0, 0.0, 0.0,
                                                 0.0, trace_b_elastic_hat / 3.0, 0.0,
                                                 0.0, 0.0, trace_b_elastic_hat / 3.0)
    s_trial = mu * dev_b_elastic_hat
    s_trial_norm = frobenius_norm(s_trial)

    yield_condition = s_trial_norm - wp.sqrt(2.0 / 3.0) * yield_threshold
    if yield_condition > 0:
        mu_hat = mu * trace_b_elastic_hat / 3.0

        if shear_ratio == 1.0:
            s_new_norm = s_trial_norm - yield_condition / (1.0 + viscosity / (2.0 * mu_hat * dt))
        else:
            s_lo = wp.sqrt(2.0 / 3.0) * yield_threshold
            s_hi = s_trial_norm
            for _ in range(30):
                s_mid = 0.5 * (s_lo + s_hi)
                delta = wp.max(s_mid - wp.sqrt(2.0 / 3.0) * yield_threshold, 0.0)
                g = (wp.pow(viscosity, 1.0 / shear_ratio)) * (s_mid - s_trial_norm) + \
                    2.0 * mu_hat * dt * wp.pow(delta, 1.0 / shear_ratio)
                if g > 0.0:
                    s_hi = s_mid
                else:
                    s_lo = s_mid
                if wp.abs(g / s_trial_norm) < 1e-6:
                    break
            s_new_norm = 0.5 * (s_lo + s_hi)

        # Radial return along the trial flow direction
        s_new = (s_trial / s_trial_norm) * s_new_norm
        b_elastic_hat_update = s_new / mu + wp.mat33(1.0 / 3.0 * trace_b_elastic_hat, 0.0, 0.0,
                                                     0.0, 1.0 / 3.0 * trace_b_elastic_hat, 0.0,
                                                     0.0, 0.0, 1.0 / 3.0 * trace_b_elastic_hat)
        b_elastic_hat_update = wp.pow(wp.determinant(b_elastic_hat_update), -1.0 / 3.0) * b_elastic_hat_update

        return b_elastic_hat_update
    else:
        return b_elastic_hat
