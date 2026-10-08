import warp as wp


def launch_sync(kernel, dim, inputs, device="cuda"):
    """
    Launch a Warp kernel with device-wide synchronization before and after. Warp and PyTorch
    run on different CUDA streams and share buffers through zero-copy views (wp.to_torch /
    wp.from_torch), so both sides must be flushed around kernels that read or write them.
    """
    wp.synchronize()
    wp.launch(kernel=kernel, dim=dim, inputs=inputs, device=device)
    wp.synchronize()


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0)


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value


def torch_to_wp_mat33(input_tensor, device="cuda"):
    input_tensor_flatten = input_tensor.reshape(-1, 9).cpu().numpy()
    return wp.array(input_tensor_flatten, dtype=wp.mat33f, device=device)
