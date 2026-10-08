"""First-order autograd wrapper for the source-only CUDA backprojector."""

import importlib
import math
from numbers import Integral, Real

import torch
from torch.autograd.function import once_differentiable


def _positive_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive Python number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _native():
    try:
        return importlib.import_module("src_cbct_backprojector_cuda")
    except ModuleNotFoundError as error:
        if error.name != "src_cbct_backprojector_cuda":
            raise
        raise RuntimeError(
            "CUDA extension is not built. From the repository root run: "
            "python -m pip install --no-build-isolation --no-deps -e ."
        ) from error


def detector_centers(delta_u, delta_v, detector_u, detector_v, pixel_size):
    """Convert physical detector offsets to zero-ray pixel indices.

    Positive physical offsets decrease the corresponding center pixel index.
    Tensor inputs preserve their autograd history.
    """
    pixel_size = _positive_scalar(pixel_size, "pixel_size")
    for name, size in (("detector_u", detector_u), ("detector_v", detector_v)):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    return (
        (detector_u - 1.0) / 2.0 - delta_u / pixel_size,
        (detector_v - 1.0) / 2.0 - delta_v / pixel_size,
    )


class _Backproject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj, angles, sid, sod, u0, v0, sx, sy, sz,
                pixel_size, volume_shape, voxel_size, volume_offset):
        tensors = tuple(t.contiguous() for t in (proj, angles, sid, sod, u0, v0, sx, sy, sz))
        ctx.save_for_backward(*tensors)
        ctx.scalars = (pixel_size, voxel_size, volume_offset)
        return _native().forward(
            *tensors, pixel_size, *volume_shape, voxel_size, *volume_offset
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_volume):
        pixel_size, voxel_size, volume_offset = ctx.scalars
        gradients = _native().backward(
            grad_volume.contiguous(), *ctx.saved_tensors,
            pixel_size, voxel_size, *volume_offset,
        )
        return tuple(
            grad if needed else None
            for grad, needed in zip(gradients, ctx.needs_input_grad[:9])
        ) + (None,) * 4


def backproject(projections, angles, sid, sod, det_u0, det_v0,
                src_x, src_y, src_z, pixel_size, volume_shape, voxel_size,
                volume_offset=(0.0, 0.0, 0.0)):
    """Backproject [A,U,V] projections to a [D,H,W] = [Z,Y,X] volume.

    All nine tensors must be CUDA float32 on the same device. Each of the
    eight geometry tensors has shape [A]. ``sid`` means total SDD; ``sod``
    means SOD. Angles are radians; detector centers are pixel indices;
    all distances share a physical unit. Source offsets use world axes.

    This operation sums weighted bilinear samples without filtering or
    angular normalization. Only first derivatives of tensor inputs exist.
    The scalar volume offset is ordered (x, y, z).
    """
    names = ("projections", "angles", "sid", "sod", "det_u0", "det_v0",
             "src_x", "src_y", "src_z")
    tensors = (projections, angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z)
    for name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float32 or tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA float32 tensor")
        if tensor.device != projections.device:
            raise ValueError("All tensors must be on the same CUDA device")
    if projections.ndim != 3:
        raise ValueError("projections must have shape [A,U,V]")
    num_angles, detector_u, detector_v = projections.shape
    if num_angles < 1 or min(detector_u, detector_v) < 2:
        raise ValueError("projections requires A >= 1 and U,V >= 2")
    for name, tensor in zip(names[1:], tensors[1:]):
        if tensor.shape != (num_angles,):
            raise ValueError(f"{name} must have shape [{num_angles}]")
    pixel_size = _positive_scalar(pixel_size, "pixel_size")
    voxel_size = _positive_scalar(voxel_size, "voxel_size")
    volume_shape = tuple(volume_shape)
    if len(volume_shape) != 3 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size < 1
        for size in volume_shape
    ):
        raise ValueError("volume_shape must contain three positive integers (D,H,W)")
    volume_shape = tuple(int(size) for size in volume_shape)
    if projections.numel() > 2**31 - 1 or math.prod(volume_shape) > 2**31 - 1:
        raise ValueError("Input or volume exceeds the kernel's 32-bit indexing limit")
    if max(volume_shape[:2]) > 8 * 65535:
        raise ValueError("D,H exceed the kernel's 3D CUDA grid limits")
    volume_offset = tuple(volume_offset)
    if len(volume_offset) != 3 or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in volume_offset
    ):
        raise ValueError("volume_offset must contain three finite numbers (x,y,z)")
    volume_offset = tuple(float(value) for value in volume_offset)
    return _Backproject.apply(
        *tensors, pixel_size, volume_shape, voxel_size, volume_offset
    )
