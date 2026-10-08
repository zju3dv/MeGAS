#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal


class Camera:
    """
    Pinhole camera following the 3DGS convention.

    :param R: camera-to-world rotation [3, 3]
    :param T: world-to-camera translation [3]
    """

    def __init__(self, R, T, FoVx, FoVy, image_width, image_height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.0001

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = None

    @property
    def get_intrinsic(self):
        if self.intrinsic is None:
            intrinsic = np.eye(4)
            intrinsic[0, 0] = fov2focal(self.FoVx, self.image_width)
            intrinsic[1, 1] = fov2focal(self.FoVy, self.image_height)
            intrinsic[0, 2] = self.image_width / 2
            intrinsic[1, 2] = self.image_height / 2
            self.intrinsic = torch.tensor(intrinsic).cuda().float()
        return self.intrinsic

    @property
    def get_c2w(self):
        return torch.inverse(self.world_view_transform.T)
