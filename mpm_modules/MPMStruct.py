import warp as wp


@wp.struct
class MPMGrid_State:
    n_grid: int
    dx: float
    inv_dx: float

    grid_m: wp.array(dtype=float, ndim=3)  # nodal mass
    grid_v: wp.array(dtype=wp.vec3, ndim=3)  # nodal momentum (after P2G) / velocity (after normalization)

    # Heat advection-diffusion
    grid_thermal: wp.array(dtype=float, ndim=3)  # nodal temperature T
    heat_diff_ratio: float  # thermal diffusivity alpha = k / (rho * c)

    # Conjugate-gradient buffers for the implicit diffusion solve (I - alpha * dt * Laplacian) T = b
    grid_rhs: wp.array(dtype=float, ndim=3)  # b = T^n - dt * (v . grad T^n)
    cg_r: wp.array(dtype=float, ndim=3)
    cg_p: wp.array(dtype=float, ndim=3)
    cg_Ap: wp.array(dtype=float, ndim=3)
    cg_y: wp.array(dtype=float, ndim=3)
    cg_dot_buf: wp.array(dtype=float)  # scalar reduction buffer


@wp.struct
class MPMMaterial_State:
    material_type: int  # 0: jelly (hyperelastic), 5: foam (visco-plastic)

    E: wp.array(dtype=float)  # Young's modulus
    nu: wp.array(dtype=float)  # Poisson's ratio
    mu: wp.array(dtype=float)  # shear modulus (Lame's second parameter)
    lam: wp.array(dtype=float)  # Lame's first parameter
    bulk: wp.array(dtype=float)  # bulk modulus

    elastic_method_type: wp.array(dtype=int)  # 0: fixed corotated, 1: StVK, 2: Neo-Hookean, 3: Herschel-Bulkley

    # Herschel-Bulkley visco-plasticity
    shear_ratio: float  # flow index n (n = 1 recovers Bingham plasticity)
    viscosity: float
    yield_threshold: float


@wp.struct
class MPMParticle_State:
    n_particles: int

    particle_x: wp.array(dtype=wp.vec3)  # position
    particle_v: wp.array(dtype=wp.vec3)  # velocity
    particle_F: wp.array(dtype=wp.mat33)  # deformation gradient
    particle_C: wp.array(dtype=wp.mat33)  # APIC affine velocity matrix
    particle_be_hat: wp.array(dtype=wp.mat33)  # volume-preserving elastic left Cauchy-Green tensor

    # Gaussian kernel deformation
    particle_init_cov3D: wp.array(dtype=wp.mat33)  # rest-state covariance Sigma_0
    particle_cov3D: wp.array(dtype=wp.mat33)  # deformed covariance F Sigma_0 F^T
    particle_R: wp.array(dtype=wp.mat33)  # eigenvectors of the deformed covariance
    particle_Scale: wp.array(dtype=wp.vec3)  # square roots of the eigenvalues

    # Thermal state
    thermal_phase_threshold: float  # melting temperature for the solid-to-liquid transition
    particle_thermal: wp.array(dtype=float)  # temperature T_p
    delta_particle_thermal: wp.array(dtype=wp.vec3)  # temperature gradient grad T_p (affine thermal transfer)

    particle_density: wp.array(dtype=float)
    particle_vol: wp.array(dtype=float)
    particle_mass: wp.array(dtype=float)

    phase_state: wp.array(dtype=int)  # 0: solid, 1: liquid

    solid_material_state: MPMMaterial_State
    liquid_material_state: MPMMaterial_State
