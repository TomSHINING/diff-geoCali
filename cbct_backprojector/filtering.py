"""The project's chunked FDK initialization for a uniformly sampled circular scan.

These helpers preserve the filter and ``pi / number_of_views`` normalization
used in the simulation calibration pipeline. The cosine preweight uses centered
detector coordinates without detector/source offsets. This is the project's FDK
approximation; it is not a general reconstruction formula for arbitrary orbits,
short scans, nonuniform angles, or perturbed geometry.
"""

import math
import operator

import torch

from .operator import backproject


def _check_projections(projections):
    if not isinstance(projections, torch.Tensor):
        raise TypeError("projections must be a torch.Tensor")
    if projections.ndim != 3 or any(size < 1 for size in projections.shape):
        raise ValueError("projections must have a nonempty [A, U, V] shape")
    if projections.dtype != torch.float32:
        raise TypeError("projections must have dtype torch.float32")


def _positive_scalar(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _check_alpha(alpha):
    alpha = _positive_scalar("alpha", alpha)
    if alpha > 1:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    return alpha


def _filter_distance(name, value, projections):
    distance = torch.as_tensor(
        value, dtype=projections.dtype, device=projections.device
    ).reshape(-1)
    if distance.numel() == 1:
        distance = distance.expand(projections.shape[0])
    elif distance.numel() != projections.shape[0]:
        raise ValueError(f"{name} must be a scalar or have one value per view")
    if not bool(torch.isfinite(distance).all()) or bool((distance <= 0).any()):
        raise ValueError(f"{name} must contain finite positive distances")
    return distance.view(-1, 1, 1)


def fdk_filter(projections, pixel_size, sod, sid, alpha=0.8):
    """Apply centered cosine weighting and a Hann-windowed discrete ramp filter.

    Args:
        projections: Float32 line-integral tensor with layout ``[A, U, V]``.
            CPU and CUDA tensors are supported by the filter alone.
        pixel_size: Square detector pixel spacing in the geometry's length unit.
        sod: Source-to-origin distance, scalar or one value per view.
        sid: Source-to-detector distance, scalar or one value per view.
        alpha: Hann cutoff as a fraction of the Nyquist frequency, in ``(0, 1]``.

    Returns:
        A contiguous float32 tensor with the same shape and device. Filtering
        runs along U, with zero padding to the next power of two at least 2U,
        followed by the pipeline's ``pixel_size * sid / sod`` scale factor.

    The cosine weights intentionally ignore detector and source offsets, as in
    the original pipeline. Inputs should be line integrals, not raw intensities.
    """
    _check_projections(projections)
    pixel_size = _positive_scalar("pixel_size", pixel_size)
    alpha = _check_alpha(alpha)
    sod = _filter_distance("sod", sod, projections)
    sid = _filter_distance("sid", sid, projections)
    _, num_u, num_v = projections.shape
    device = projections.device
    dtype = projections.dtype

    u = (torch.arange(num_u, device=device, dtype=dtype) - (num_u - 1) / 2) * pixel_size
    v = (torch.arange(num_v, device=device, dtype=dtype) - (num_v - 1) / 2) * pixel_size
    distance = torch.sqrt(sid.square() + u.view(1, -1, 1).square() + v.view(1, 1, -1).square())
    weighted = projections * (sid / distance)

    pad_u = 1 << (2 * num_u - 1).bit_length()
    projection_fft = torch.fft.fft(weighted, n=pad_u, dim=1)
    indices = torch.arange(pad_u, device=device)
    n = torch.where(indices > pad_u // 2, indices - pad_u, indices)
    ramp_spatial = torch.zeros(pad_u, device=device, dtype=dtype)
    ramp_spatial[0] = 1.0 / (4.0 * pixel_size**2)
    odd = n % 2 != 0
    ramp_spatial[odd] = -1.0 / (math.pi * n[odd].to(dtype) * pixel_size).square()
    ramp_kernel = torch.fft.fft(ramp_spatial).real

    frequency = torch.fft.fftfreq(pad_u, device=device, dtype=dtype)
    hann_window = 0.5 + 0.5 * torch.cos(2.0 * math.pi * frequency / alpha)
    hann_window = torch.where(
        frequency.abs() <= alpha * 0.5, hann_window, torch.zeros_like(hann_window)
    )
    projection_fft.mul_((ramp_kernel * hann_window).view(1, -1, 1))
    filtered = torch.fft.ifft(projection_fft, dim=1).real[:, :num_u, :]
    return (filtered * pixel_size * (sid / sod)).contiguous()


@torch.no_grad()
def reconstruct_fdk(
    projections,
    angles,
    sid,
    sod,
    det_u0,
    det_v0,
    src_x,
    src_y,
    src_z,
    pixel_size,
    volume_shape,
    voxel_size,
    volume_offset=(0.0, 0.0, 0.0),
    alpha=0.8,
    angle_chunk_size=32,
):
    """Filter and backproject in view chunks, then apply ``pi / A`` and clamp.

    ``projections`` is a float32 CUDA tensor ``[A, U, V]``. Each geometry argument
    from ``angles`` through ``src_z`` must be a float32 tensor ``[A]`` on the same
    CUDA device; angles are radians and detector centers are pixel indices.
    ``volume_shape`` is ``(D, H, W)``; ``volume_offset`` follows ``backproject``.

    This inference helper disables autograd, accumulates the raw backprojection
    for every chunk, multiplies by ``pi / A``, and clips negative values to zero,
    matching the original FDK initializer. Chunking reduces FFT working memory.
    Uniform angular weights assume a complete, uniformly sampled circular scan.
    Passing calibrated geometry preserves the project's approximation, including
    its centered cosine preweight; it does not make this arbitrary-orbit FDK.
    """
    _check_projections(projections)
    if not projections.is_cuda:
        raise ValueError("reconstruct_fdk requires CUDA projections")
    if min(projections.shape[1:]) < 2:
        raise ValueError("reconstruct_fdk requires detector dimensions U and V >= 2")
    num_angles = projections.shape[0]
    geometry = (angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z)
    names = ("angles", "sid", "sod", "det_u0", "det_v0", "src_x", "src_y", "src_z")
    for name, values in zip(names, geometry):
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor of shape [A]")
        if values.ndim != 1 or values.numel() != num_angles:
            raise ValueError(f"{name} must have shape [{num_angles}]")
        if values.dtype != projections.dtype or values.device != projections.device:
            raise ValueError(f"{name} must match the projections' dtype and device")
    if not bool(torch.isfinite(torch.stack(geometry)).all()):
        raise ValueError("geometry must contain only finite values")
    if bool((sid <= 0).any()) or bool((sod <= 0).any()):
        raise ValueError("sid and sod must contain positive distances")

    pixel_size = _positive_scalar("pixel_size", pixel_size)
    voxel_size = _positive_scalar("voxel_size", voxel_size)
    alpha = _check_alpha(alpha)
    try:
        angle_chunk_size = operator.index(angle_chunk_size)
        volume_shape = tuple(operator.index(size) for size in volume_shape)
    except TypeError as error:
        raise TypeError("angle_chunk_size and volume dimensions must be integers") from error
    if angle_chunk_size < 1:
        raise ValueError("angle_chunk_size must be at least 1")
    if len(volume_shape) != 3 or any(size < 1 for size in volume_shape):
        raise ValueError("volume_shape must contain three positive dimensions (D, H, W)")
    if math.prod(volume_shape) > 2**31 - 1:
        raise ValueError("volume exceeds the backprojector's 32-bit indexing limit")
    if max(volume_shape[:2]) > 8 * 65535:
        raise ValueError("D,H exceed the backprojector's 3D CUDA grid limits")
    if min(num_angles, angle_chunk_size) * math.prod(projections.shape[1:]) > 2**31 - 1:
        raise ValueError("Reduce angle_chunk_size to fit the backprojector's 32-bit indexing limit")
    volume_offset = tuple(float(value) for value in volume_offset)
    if len(volume_offset) != 3 or not all(math.isfinite(value) for value in volume_offset):
        raise ValueError("volume_offset must contain three finite values")

    volume = torch.zeros(volume_shape, device=projections.device, dtype=projections.dtype)
    for start in range(0, num_angles, angle_chunk_size):
        end = min(start + angle_chunk_size, num_angles)
        filtered = fdk_filter(
            projections[start:end], pixel_size, sod[start:end], sid[start:end], alpha
        )
        chunk_geometry = tuple(values[start:end].contiguous() for values in geometry)
        volume_chunk = backproject(
            filtered,
            *chunk_geometry,
            pixel_size,
            volume_shape,
            voxel_size,
            volume_offset=volume_offset,
        )
        volume.add_(volume_chunk)
        del filtered, volume_chunk
    return volume.mul_(math.pi / num_angles).clamp_min_(0.0)
