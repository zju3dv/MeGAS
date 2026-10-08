import warp as wp


@wp.struct
class Colliders:
    point: wp.vec3
    normal: wp.vec3
    collider_type: int  # 0: sticky


@wp.struct
class Plane_Thermal_Boundary:
    """Dirichlet boundary: grid nodes behind the plane are held at a fixed temperature."""
    boundary_type: int
    point: wp.vec3
    normal: wp.vec3
    thermal: float


@wp.struct
class Particle_Thermal_Boundary:
    """Heat source: particles inside the mask are held at a fixed temperature."""
    boundary_type: int
    particle_mask: wp.array(dtype=int)
    thermal: float
