"""Check manual bilinear backprojection, nine gradients and input contiguity."""

import argparse
import math

import torch


NAMES = (
    "projections", "angles", "sid", "sod", "det_u0", "det_v0",
    "src_x", "src_y", "src_z",
)
PIXEL_SIZE = 0.9
VOXEL_SIZE = 1.0
VOLUME_SHAPE = (3, 4, 5)
VOLUME_OFFSET = (0.27, -0.31, 0.23)


def reference_backproject(parameters):
    """Evaluate the CUDA formula on CPU with explicit detector-index sampling."""
    projections, angles, sid, sod, u_center, v_center, src_x, src_y, src_z = parameters
    views, detector_u, detector_v = projections.shape
    depth, height, width = VOLUME_SHAPE
    z, y, x = torch.meshgrid(
        torch.arange(depth, dtype=projections.dtype),
        torch.arange(height, dtype=projections.dtype),
        torch.arange(width, dtype=projections.dtype),
        indexing="ij",
    )
    vx = (x - width * 0.5) * VOXEL_SIZE + VOLUME_OFFSET[0]
    vy = (y - height * 0.5) * VOXEL_SIZE + VOLUME_OFFSET[1]
    vz = (z - depth * 0.5) * VOXEL_SIZE + VOLUME_OFFSET[2]
    volume = torch.zeros_like(vx)
    for a in range(views):
        cosine, sine = angles[a].cos(), angles[a].sin()
        px, py = vx * cosine + vy * sine, -vx * sine + vy * cosine
        ds_u = src_x[a] * cosine + src_y[a] * sine
        ds_n = -src_x[a] * sine + src_y[a] * cosine
        denominator = sod[a] + ds_n - py
        detector_distance = sid[a] + ds_n
        geometry_valid = (denominator > 1e-6) & (detector_distance > 1e-6)
        # Prevent invalid rays from creating NaN gradients before masking.
        safe_denominator = torch.where(geometry_valid, denominator, 1.0)
        magnification = detector_distance / safe_denominator
        u_idx = (ds_u + magnification * (px - ds_u)) / PIXEL_SIZE + u_center[a]
        v_idx = (src_z[a] + magnification * (vz - src_z[a])) / PIXEL_SIZE + v_center[a]
        weight = ((sod[a] + ds_n) / safe_denominator).square()
        valid = (
            geometry_valid & (u_idx >= 0) & (u_idx < detector_u - 1)
            & (v_idx >= 0) & (v_idx < detector_v - 1)
        )
        # Clamp only integer gather addresses; the CUDA mask defines support.
        u0 = u_idx.floor().to(torch.long).clamp(0, detector_u - 2)
        v0 = v_idx.floor().to(torch.long).clamp(0, detector_v - 2)
        fu, fv = u_idx - u0, v_idx - v0
        image = projections[a]
        interpolation = (
            (1 - fu) * (1 - fv) * image[u0, v0]
            + fu * (1 - fv) * image[u0 + 1, v0]
            + (1 - fu) * fv * image[u0, v0 + 1]
            + fu * fv * image[u0 + 1, v0 + 1]
        )
        volume = volume + torch.where(valid, weight * interpolation, 0.0)
    return volume


def make_case(dtype):
    """Create an asymmetric case with source shifts, volume shifts and edges."""
    views, detector_u, detector_v = 3, 9, 7
    a = torch.arange(views, dtype=dtype)[:, None, None]
    v = torch.arange(detector_v, dtype=dtype)[None, :, None]
    u = torch.arange(detector_u, dtype=dtype)[None, None, :]
    # A transpose intentionally supplies non-contiguous [A, U, V] projections.
    projections = (0.4 + 0.07 * a + 0.08 * u + 0.05 * v + 0.013 * u * v)
    projections = projections.transpose(1, 2)
    geometry_values = (
        [0.17, 1.23, 2.51],
        [18.0, 18.7, 17.8],
        [12.0, 12.4, 11.8],
        [4.17, 3.83, 4.11],
        [3.13, 2.91, 3.19],
        [0.13, -0.21, 0.16],
        [-0.17, 0.14, 0.09],
        [0.11, -0.08, 0.17],
    )
    parameters = [projections.detach().requires_grad_()]
    parameters.extend(torch.tensor(value, dtype=dtype, requires_grad=True) for value in geometry_values)
    depth, height, width = VOLUME_SHAPE
    # A transposed upstream gradient also exercises backward contiguous handling.
    upstream = torch.linspace(0.2, 1.1, depth * height * width, dtype=dtype)
    upstream = upstream.reshape(depth, width, height).transpose(1, 2)
    assert not parameters[0].is_contiguous()
    assert not upstream.is_contiguous()
    return parameters, upstream


def scalar_loss(parameters, upstream):
    """Independently evaluate the weighted loss using Python double arithmetic."""
    projections, angles, sid, sod, u_center, v_center, src_x, src_y, src_z = (
        value.detach().tolist() for value in parameters
    )
    weights = upstream.tolist()
    detector_u, detector_v = len(projections[0]), len(projections[0][0])
    depth, height, width = VOLUME_SHAPE
    terms = []
    for z in range(depth):
        vz = (z - depth * 0.5) * VOXEL_SIZE + VOLUME_OFFSET[2]
        for y in range(height):
            vy = (y - height * 0.5) * VOXEL_SIZE + VOLUME_OFFSET[1]
            for x in range(width):
                vx = (x - width * 0.5) * VOXEL_SIZE + VOLUME_OFFSET[0]
                contributions = []
                for a, angle in enumerate(angles):
                    cosine, sine = math.cos(angle), math.sin(angle)
                    px, py = vx * cosine + vy * sine, -vx * sine + vy * cosine
                    ds_u = src_x[a] * cosine + src_y[a] * sine
                    ds_n = -src_x[a] * sine + src_y[a] * cosine
                    denominator = sod[a] + ds_n - py
                    detector_distance = sid[a] + ds_n
                    if denominator <= 1e-6 or detector_distance <= 1e-6:
                        continue
                    magnification = detector_distance / denominator
                    u_idx = (ds_u + magnification * (px - ds_u)) / PIXEL_SIZE + u_center[a]
                    v_idx = (src_z[a] + magnification * (vz - src_z[a])) / PIXEL_SIZE + v_center[a]
                    if not (0 <= u_idx < detector_u - 1 and 0 <= v_idx < detector_v - 1):
                        continue
                    u0, v0 = math.floor(u_idx), math.floor(v_idx)
                    fu, fv = u_idx - u0, v_idx - v0
                    image = projections[a]
                    interpolation = (
                        (1 - fu) * (1 - fv) * image[u0][v0]
                        + fu * (1 - fv) * image[u0 + 1][v0]
                        + (1 - fu) * fv * image[u0][v0 + 1]
                        + fu * fv * image[u0 + 1][v0 + 1]
                    )
                    contributions.append(((sod[a] + ds_n) / denominator) ** 2 * interpolation)
                terms.append(weights[z][y][x] * math.fsum(contributions))
    return math.fsum(terms)


def unravel_index(flat_index, shape):
    """Convert a flat logical tensor index without assuming contiguous storage."""
    indices = []
    for size in reversed(shape):
        indices.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(indices))


def check_cpu():
    """Check every gradient family against two independent finite differences."""
    parameters, upstream = make_case(torch.float64)
    loss = (reference_backproject(parameters) * upstream).sum()
    scalar_value = scalar_loss(parameters, upstream)
    if not math.isclose(loss.item(), scalar_value, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("Vectorized and scalar CPU forward references disagree.")
    gradients = torch.autograd.grad(loss, parameters)
    epsilon = 1e-6
    for name, parameter, gradient in zip(NAMES, parameters, gradients):
        # Select meaningful entries; all nine input families receive two samples.
        samples = gradient.detach().abs().reshape(-1).topk(min(2, parameter.numel())).indices.tolist()
        maximum_error = 0.0
        for flat_index in samples:
            index = unravel_index(flat_index, parameter.shape)
            plus = [value.detach().clone() for value in parameters]
            minus = [value.detach().clone() for value in parameters]
            position = NAMES.index(name)
            plus[position][index] += epsilon
            minus[position][index] -= epsilon
            finite_difference = (scalar_loss(plus, upstream) - scalar_loss(minus, upstream)) / (2 * epsilon)
            analytic = gradient[index].item()
            maximum_error = max(maximum_error, abs(analytic - finite_difference))
            if not math.isclose(analytic, finite_difference, rel_tol=2e-6, abs_tol=1e-7):
                raise AssertionError(
                    f"{name}{index}: autograd={analytic:.9g}, finite difference={finite_difference:.9g}"
                )
        print(f"CPU {name:12s} finite differences: PASS (max abs error {maximum_error:.3e})")


def check_cuda():
    """Compare CUDA forward and all nine backward outputs with CPU autograd."""
    from cbct_backprojector import backproject

    parameters, upstream = make_case(torch.float32)
    cpu_parameters = [value.detach().to(torch.float64).requires_grad_() for value in parameters]
    expected = reference_backproject(cpu_parameters)
    cpu_gradients = torch.autograd.grad((expected * upstream.to(torch.float64)).sum(), cpu_parameters)
    gpu_parameters = [value.detach().to("cuda").requires_grad_() for value in parameters]
    gpu_upstream = upstream.to("cuda")
    assert not gpu_parameters[0].is_contiguous()
    assert not gpu_upstream.is_contiguous()
    actual = backproject(
        *gpu_parameters, PIXEL_SIZE, VOLUME_SHAPE, VOXEL_SIZE,
        volume_offset=VOLUME_OFFSET,
    )
    gpu_gradients = torch.autograd.grad(actual, gpu_parameters, grad_outputs=gpu_upstream)
    torch.cuda.synchronize()
    actual_cpu = actual.detach().cpu().to(torch.float64)
    torch.testing.assert_close(actual_cpu, expected.detach(), rtol=2e-5, atol=2e-5)
    print(f"CUDA forward: PASS (max abs error {(actual_cpu - expected).abs().max().item():.3e})")
    for name, gpu_gradient, cpu_gradient in zip(NAMES, gpu_gradients, cpu_gradients):
        actual_gradient = gpu_gradient.detach().cpu().to(torch.float64)
        torch.testing.assert_close(actual_gradient, cpu_gradient, rtol=3e-4, atol=3e-5)
        error = (actual_gradient - cpu_gradient).abs().max().item()
        print(f"CUDA {name:12s} gradient: PASS (max abs error {error:.3e})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Validate CPU reference and finite differences; do not test CUDA",
    )
    args = parser.parse_args()
    if not args.cpu_only and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --cpu-only to validate only the CPU reference.")
    check_cpu()
    if args.cpu_only:
        print("CPU reference checks passed. CUDA forward/backward were NOT tested.")
    else:
        check_cuda()
        print("CPU reference and CUDA forward/backward checks passed.")


if __name__ == "__main__":
    main()
