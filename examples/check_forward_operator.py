"""Check fixed-step ray integration, linearity and local first-order gradients."""

import argparse
import math

import torch


NAMES = (
    "volume", "angles", "sid", "sod", "det_u0", "det_v0",
    "src_x", "src_y", "src_z",
)
DETECTOR_U, DETECTOR_V = 5, 4
PIXEL_SIZE, VOXEL_SIZE, STEP_SIZE = 0.75, 0.8, 0.37
VOLUME_OFFSET = (0.17, -0.12, 0.09)
SELECTED_RAY = (0, 2, 1)
SELECTED_WEIGHT = 1.3


def intersect_box(origin, direction, lower, upper):
    """Mirror the CUDA slab rule, retaining the selected entry derivative."""
    entry = torch.zeros((), dtype=origin.dtype)
    exit_point = torch.full((), float("inf"), dtype=origin.dtype)
    entry_axis = -1
    for axis in range(3):
        if abs(direction[axis].item()) < 1e-12:
            if not lower[axis] <= origin[axis].item() <= upper[axis]:
                return None
            continue
        first = (lower[axis] - origin[axis]) / direction[axis]
        last = (upper[axis] - origin[axis]) / direction[axis]
        if first.item() > last.item():
            first, last = last, first
        if first.item() > entry.item():
            entry, entry_axis = first, axis
        exit_point = torch.minimum(exit_point, last)
        if exit_point.item() < entry.item():
            return None
    return entry, exit_point, entry_axis


def sample_trilinear(volume, points):
    """Interpolate eight clamped neighbors using raw floor-based fractions."""
    depth, height, width = volume.shape
    raw = points.floor().to(torch.long)
    fractions = points - raw
    value = torch.zeros(points.shape[0], dtype=volume.dtype)
    for dx in (0, 1):
        x = (raw[:, 0] + dx).clamp(0, width - 1)
        wx = fractions[:, 0] if dx else 1 - fractions[:, 0]
        for dy in (0, 1):
            y = (raw[:, 1] + dy).clamp(0, height - 1)
            wy = fractions[:, 1] if dy else 1 - fractions[:, 1]
            for dz in (0, 1):
                z = (raw[:, 2] + dz).clamp(0, depth - 1)
                wz = fractions[:, 2] if dz else 1 - fractions[:, 2]
                value = value + wx * wy * wz * volume[z, y, x]
    return value, tuple(raw.reshape(-1).tolist())


def reference_project(parameters):
    """Evaluate CPU rays and return sampling traces for local derivative checks.

    Every sample has a full STEP_SIZE weight, including the last sample. The
    sample count is a detached integer: its changes are outside the derivative
    implemented by the CUDA backward kernel. No grid_sample convention is used.
    """
    volume, angles, sid, sod, u_center, v_center, src_x, src_y, src_z = parameters
    depth, height, width = volume.shape
    sizes_xyz = (width, height, depth)
    lower = [(-size * 0.5 - 0.5) * VOXEL_SIZE + shift for size, shift in zip(sizes_xyz, VOLUME_OFFSET)]
    upper = [(size * 0.5 - 0.5) * VOXEL_SIZE + shift for size, shift in zip(sizes_xyz, VOLUME_OFFSET)]
    offset = torch.tensor(VOLUME_OFFSET, dtype=volume.dtype)
    half_sizes = torch.tensor(sizes_xyz, dtype=volume.dtype) * 0.5
    outputs, traces = [], []
    for a in range(angles.numel()):
        cosine, sine = angles[a].cos(), angles[a].sin()
        origin = torch.stack((-sod[a] * sine + src_x[a], sod[a] * cosine + src_y[a], src_z[a]))
        detector_n = sod[a] - sid[a]
        for u in range(DETECTOR_U):
            u_phys = (u - u_center[a]) * PIXEL_SIZE
            for v in range(DETECTOR_V):
                v_phys = (v - v_center[a]) * PIXEL_SIZE
                target = torch.stack((
                    u_phys * cosine - detector_n * sine,
                    u_phys * sine + detector_n * cosine,
                    v_phys,
                ))
                ray = target - origin
                length = ray.norm()
                intersection = None if length.item() <= 1e-8 else intersect_box(origin, ray / length, lower, upper)
                if intersection is None:
                    outputs.append(volume.sum() * 0)
                    traces.append((-1, 0, ()))
                    continue
                entry, exit_point, entry_axis = intersection
                count = math.floor((exit_point - entry).detach().item() / STEP_SIZE) + 1
                # Arithmetic is double on CPU; CUDA repeatedly adds a float step.
                t = entry + torch.arange(count, dtype=volume.dtype) * STEP_SIZE
                points = origin[None] + t[:, None] * (ray / length)[None]
                voxel_points = (points - offset) / VOXEL_SIZE + half_sizes
                sampled, cells = sample_trilinear(volume, voxel_points)
                outputs.append(sampled.sum() * STEP_SIZE)
                traces.append((entry_axis, count, cells))
    return torch.stack(outputs).reshape(angles.numel(), DETECTOR_U, DETECTOR_V), traces


def make_case(dtype):
    """Create a non-cubic, non-contiguous volume with zero boundary samples."""
    depth, height, width = 5, 7, 6
    z, x, y = torch.meshgrid(
        torch.arange(depth, dtype=dtype),
        torch.arange(width, dtype=dtype),
        torch.arange(height, dtype=dtype),
        indexing="ij",
    )
    envelope = (
        x * (width - 1 - x) / (width - 1) ** 2
        * y * (height - 1 - y) / (height - 1) ** 2
        * z * (depth - 1 - z) / (depth - 1) ** 2
    )
    volume = (64 * envelope * (1 + 0.06 * x + 0.04 * y + 0.08 * z)).transpose(1, 2)
    geometry = (
        [0.13, 1.28], [40.0, 41.0], [24.0, 24.7],
        [2.11, 1.87], [1.42, 1.69], [0.13, -0.18],
        [-0.11, 0.15], [0.19, -0.07],
    )
    parameters = [volume.detach().requires_grad_()]
    parameters.extend(torch.tensor(values, dtype=dtype, requires_grad=True) for values in geometry)
    upstream = torch.linspace(0.2, 1.1, 2 * DETECTOR_U * DETECTOR_V, dtype=dtype)
    upstream = upstream.reshape(2, DETECTOR_V, DETECTOR_U).transpose(1, 2)
    assert not parameters[0].is_contiguous() and not upstream.is_contiguous()
    return parameters, upstream


def unravel_index(flat_index, shape):
    """Convert an index using logical tensor order, including transposed inputs."""
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(result))


def selected_trace(traces):
    a, u, v = SELECTED_RAY
    return traces[(a * DETECTOR_U + u) * DETECTOR_V + v]


def finite_difference_cases(parameters, gradients):
    """Choose a volume entry and geometry changes that stay within one region."""
    volume_flat = gradients[0].detach().abs().reshape(-1).argmax().item()
    yield 0, unravel_index(volume_flat, parameters[0].shape)
    for name in ("src_z", "det_u0", "det_v0"):
        yield NAMES.index(name), (SELECTED_RAY[0],)


def check_cpu():
    """Check constant/zero volumes, linearity and local representative gradients."""
    parameters, upstream = make_case(torch.float64)
    actual, traces = reference_project(parameters)
    assert tuple(actual.shape) == (2, DETECTOR_U, DETECTOR_V)
    assert torch.isfinite(actual).all().item() and all(trace[1] > 0 for trace in traces)
    constant = [torch.ones_like(parameters[0]), *parameters[1:]]
    constant_projection, _ = reference_project(constant)
    expected_constant = torch.tensor([trace[1] * STEP_SIZE for trace in traces], dtype=torch.float64).reshape_as(actual)
    torch.testing.assert_close(constant_projection, expected_constant, rtol=1e-12, atol=1e-12)
    zero_projection, _ = reference_project([torch.zeros_like(parameters[0]), *parameters[1:]])
    torch.testing.assert_close(zero_projection, torch.zeros_like(actual), rtol=0, atol=0)
    other_volume = parameters[0].detach().flip(-1)
    other_projection, _ = reference_project([other_volume, *parameters[1:]])
    combined, _ = reference_project([1.3 * parameters[0] - 0.4 * other_volume, *parameters[1:]])
    torch.testing.assert_close(combined, 1.3 * actual - 0.4 * other_projection, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad((actual * upstream).sum(), parameters, retain_graph=True)
    for name, gradient in zip(NAMES, gradients):
        if not torch.isfinite(gradient).all().item():
            raise AssertionError(f"CPU reference {name} gradient is non-finite")
    selected_gradients = torch.autograd.grad(SELECTED_WEIGHT * actual[SELECTED_RAY], parameters)
    baseline_trace = selected_trace(traces)
    epsilon = 1e-5
    for position, index in finite_difference_cases(parameters, selected_gradients):
        plus, minus = ([value.detach().clone() for value in parameters] for _ in range(2))
        plus[position][index] += epsilon
        minus[position][index] -= epsilon
        plus_projection, plus_traces = reference_project(plus)
        minus_projection, minus_traces = reference_project(minus)
        if selected_trace(plus_traces) != baseline_trace or selected_trace(minus_traces) != baseline_trace:
            raise AssertionError("CPU finite difference crossed a slab, sample-count or interpolation-cell boundary")
        difference = SELECTED_WEIGHT * (plus_projection[SELECTED_RAY] - minus_projection[SELECTED_RAY]).item() / (2 * epsilon)
        analytic = selected_gradients[position][index].item()
        if not math.isclose(analytic, difference, rel_tol=2e-6, abs_tol=1e-7):
            raise AssertionError(f"CPU {NAMES[position]}: analytic={analytic:.9g}, finite difference={difference:.9g}")
        print(f"CPU {NAMES[position]:12s} local finite difference: PASS")
    print("CPU forward sanity: PASS (constant volume, zero volume, linearity, nine finite gradients)")


def check_cuda():
    """Compare CUDA to CPU sampling and test local volume/source/detector changes."""
    from cbct_backprojector import project

    parameters, upstream = make_case(torch.float32)
    cpu_parameters = [value.detach().to(torch.float64).requires_grad_() for value in parameters]
    expected, traces = reference_project(cpu_parameters)
    expected_gradients = torch.autograd.grad((expected * upstream.to(torch.float64)).sum(), cpu_parameters)
    gpu_parameters = [value.detach().to("cuda").requires_grad_() for value in parameters]
    gpu_upstream = upstream.to("cuda")
    assert not gpu_parameters[0].is_contiguous() and not gpu_upstream.is_contiguous()

    def run(values):
        return project(
            *values, PIXEL_SIZE, DETECTOR_U, DETECTOR_V,
            step_size=STEP_SIZE, voxel_size=VOXEL_SIZE,
            off_x=VOLUME_OFFSET[0], off_y=VOLUME_OFFSET[1], off_z=VOLUME_OFFSET[2],
        )

    actual = run(gpu_parameters)
    gpu_gradients = torch.autograd.grad(actual, gpu_parameters, grad_outputs=gpu_upstream)
    torch.cuda.synchronize()
    actual_cpu = actual.detach().cpu().to(torch.float64)
    torch.testing.assert_close(actual_cpu, expected.detach(), rtol=5e-5, atol=5e-5)
    print(f"CUDA forward: PASS (max abs error {(actual_cpu - expected).abs().max().item():.3e})")
    for name, actual_gradient, expected_gradient in zip(NAMES, gpu_gradients, expected_gradients):
        actual_gradient = actual_gradient.detach().cpu().to(torch.float64)
        if not torch.isfinite(actual_gradient).all().item():
            raise AssertionError(f"CUDA {name} gradient is non-finite")
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-3, atol=1e-4)
        error = (actual_gradient - expected_gradient).abs().max().item()
        print(f"CUDA {name:12s} gradient: PASS (max abs error {error:.3e})")

    with torch.no_grad():
        other_volume = gpu_parameters[0].flip(-1)
        other = run([other_volume, *gpu_parameters[1:]])
        combined = run([1.3 * gpu_parameters[0] - 0.4 * other_volume, *gpu_parameters[1:]])
        torch.testing.assert_close(combined, 1.3 * actual.detach() - 0.4 * other, rtol=5e-5, atol=5e-5)
    print("CUDA volume linearity: PASS")

    selected = run(gpu_parameters)
    selected_gradients = torch.autograd.grad(SELECTED_WEIGHT * selected[SELECTED_RAY], gpu_parameters)
    baseline_trace = selected_trace(traces)
    epsilon = 1e-3
    for position, index in finite_difference_cases(gpu_parameters, selected_gradients):
        plus, minus = ([value.detach().clone() for value in gpu_parameters] for _ in range(2))
        plus[position][index] += epsilon
        minus[position][index] -= epsilon
        # Check the actual float32 perturbations with the double CPU geometry.
        plus_cpu = [value.cpu().to(torch.float64) for value in plus]
        minus_cpu = [value.cpu().to(torch.float64) for value in minus]
        _, plus_traces = reference_project(plus_cpu)
        _, minus_traces = reference_project(minus_cpu)
        if selected_trace(plus_traces) != baseline_trace or selected_trace(minus_traces) != baseline_trace:
            raise AssertionError("CUDA finite difference crossed a slab, sample-count or interpolation-cell boundary")
        with torch.no_grad():
            positive, negative = run(plus)[SELECTED_RAY].item(), run(minus)[SELECTED_RAY].item()
        denominator = (plus[position][index] - minus[position][index]).item()
        difference = SELECTED_WEIGHT * (positive - negative) / denominator
        analytic = selected_gradients[position][index].item()
        if not math.isclose(analytic, difference, rel_tol=1e-2, abs_tol=2e-3):
            raise AssertionError(f"CUDA {NAMES[position]}: analytic={analytic:.9g}, finite difference={difference:.9g}")
        print(f"CUDA {NAMES[position]:12s} local finite difference: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Run CPU reference sanity checks; do not test the CUDA operator",
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
    print("Geometry checks apply locally with fixed entry slab, sample count and interpolation cells.")


if __name__ == "__main__":
    main()
