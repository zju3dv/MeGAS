import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(viewpoint_camera,
           xyz, features, opacity, scales, rotations,
           bg_color: torch.Tensor,
           cov3D_precomp=None,
           colors_precomp=None,
           extra_attributes=None,
           only_rgb=True):
    """
    Rasterize activated Gaussian parameters.

    Covariances are given either as (scales, rotations) or as `cov3D_precomp`. Colors are given either
    as precomputed RGB `colors_precomp` or as SH `features`, of which only the DC term is rasterized.
    `extra_attributes` [N, C] are alpha-blended alongside the color.

    :return: dict with
        render       [3, H, W]  RGB image
        extra_attri  [C, H, W]  blended extra attributes
        point_id     [M]        Gaussians that are the maximum-weight contributor of at least one pixel
        render_depth [1, H, W]  camera-space depth (only if only_rgb=False)
        render_alpha [1, H, W]  accumulated opacity (only if only_rgb=False)
    """
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, device="cuda")

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    shs = features if colors_precomp is None else None

    rendered_image, _, point_id_pixel, _, _, render_extra_attri = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        extra_attrs=extra_attributes, )

    # point_id_pixel stores, for each pixel, the index of its maximum-weight Gaussian (-1 for none)
    point_id = torch.unique(point_id_pixel, sorted=True)
    if point_id[0] == -1:
        point_id = point_id[1:]

    output = {"render": rendered_image,
              "point_id": point_id,
              "extra_attri": render_extra_attri}

    if not only_rgb:
        # Depth and alpha are obtained by rasterizing per-Gaussian constants as colors
        W2C_R = torch.from_numpy(viewpoint_camera.R).float().cuda().T
        W2C_T = torch.from_numpy(viewpoint_camera.T).float().cuda()
        means3D_depth = (W2C_R @ xyz.T + W2C_T[:, None]).T[:, 2:]

        rendered_depth, _, _, _, _, _ = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=means3D_depth.repeat(1, 3),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            extra_attrs=extra_attributes)
        output["render_depth"] = rendered_depth.mean(dim=0)[None]

        # Composited over the background color, so this is exact only for a black background
        rendered_alpha, _, _, _, _, _ = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=torch.ones_like(xyz),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            extra_attrs=extra_attributes)
        output["render_alpha"] = rendered_alpha[:1]

    return output
