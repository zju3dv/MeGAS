import matplotlib
import torch


def thermal_vis(thermal_input, min_value=0.0, max_value=100.0, self_normal=False, cmap="turbo",
                flag_input_array=False):
    """
    Map temperatures to pseudo-colors.

    :param thermal_input: tensor of shape [1, H, W] (or [N]), or a numpy array if flag_input_array
    :param self_normal: normalize by the input's own range instead of [min_value, max_value]
    :return: colors in [0, 1]; a tensor [3, H, W] on the input device, or a numpy array [..., 3]
    """
    if self_normal:
        min_value = thermal_input.min()
        max_value = thermal_input.max()

    if not flag_input_array:
        thermal_input_vis = thermal_input.squeeze().detach().cpu().numpy()
    else:
        thermal_input_vis = thermal_input
    thermal_input_vis = (thermal_input_vis - min_value) / (max_value - min_value + 1e-7)
    thermal_input_vis = thermal_input_vis.clip(0, 1)

    cm = matplotlib.colormaps[cmap]
    thermal_input_vis = cm(thermal_input_vis, bytes=False)[..., 0:3]

    if not flag_input_array:
        return torch.from_numpy(thermal_input_vis).float().to(thermal_input.device).permute(2, 0, 1)
    else:
        return thermal_input_vis
