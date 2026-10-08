"""First-order autograd wrapper for the source-only cone-beam ray integrator.

The public arguments match the original ``source_only_forward_projector.py``.
The native extension is imported only when a projection is evaluated.
"""

import importlib
import math
from numbers import Integral, Real

import torch
from torch.autograd import Function

from .operator import _positive_scalar


def _native():
    try:
        return importlib.import_module("src_cbct_forwardprojector_cuda")
    except ModuleNotFoundError as error:
        if error.name != "src_cbct_forwardprojector_cuda":
            raise
        raise RuntimeError(
            "The forward-projection CUDA extension is not built. "
            "From the repository root run: "
            "python -m pip install --no-build-isolation --no-deps -e ."
        ) from error


def _validate_inputs(volume, angles, sid, sod, det_u0_arr, det_v0_arr,
                     src_x_arr, src_y_arr, src_z_arr, pixel_size,
                     detector_u, detector_v, step_size, voxel_size,
                     off_x, off_y, off_z):
    names = ("volume", "angles", "sid", "sod", "det_u0_arr", "det_v0_arr",
             "src_x_arr", "src_y_arr", "src_z_arr")
    tensors = (volume, angles, sid, sod, det_u0_arr, det_v0_arr,
               src_x_arr, src_y_arr, src_z_arr)
    for name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float32 or tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA float32 tensor")
        if tensor.device != volume.device:
            raise ValueError("All tensors must be on the same CUDA device")
    if volume.ndim != 3 or any(size < 1 for size in volume.shape):
        raise ValueError("volume must have a nonempty [D,H,W] shape")
    if angles.ndim != 1 or angles.numel() < 1:
        raise ValueError("angles must have a nonempty shape [A]")
    num_angles = angles.numel()
    for name, tensor in zip(names[1:], tensors[1:]):
        if tensor.shape != (num_angles,):
            raise ValueError(f"{name} must have shape [{num_angles}]")
    for name, size in (("detector_u", detector_u), ("detector_v", detector_v)):
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise TypeError(f"{name} must be a positive integer")
        if size < 1:
            raise ValueError(f"{name} must be a positive integer")
    offsets = (off_x, off_y, off_z)
    for name, offset in zip(("off_x", "off_y", "off_z"), offsets):
        if isinstance(offset, bool) or not isinstance(offset, Real):
            raise TypeError(f"{name} must be a finite Python number")
        if not math.isfinite(offset):
            raise ValueError(f"{name} must be finite")
    return (
        _positive_scalar(pixel_size, "pixel_size"),
        int(detector_u),
        int(detector_v),
        _positive_scalar(step_size, "step_size"),
        _positive_scalar(voxel_size, "voxel_size"),
        *(float(offset) for offset in offsets),
    )


class DifferentiableSourceOnlyForwardProjector(Function):
    @staticmethod
    def forward(ctx, volume, angles, sid, sod, det_u0_arr, det_v0_arr,
                src_x_arr, src_y_arr, src_z_arr, pixel_size,
                detector_u, detector_v, step_size, voxel_size,
                off_x=0.0, off_y=0.0, off_z=0.0):
        (pixel_size, detector_u, detector_v, step_size, voxel_size,
         off_x, off_y, off_z) = _validate_inputs(
            volume, angles, sid, sod, det_u0_arr, det_v0_arr,
            src_x_arr, src_y_arr, src_z_arr, pixel_size,
            detector_u, detector_v, step_size, voxel_size,
            off_x, off_y, off_z,
        )
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                volume, angles, sid, sod, det_u0_arr, det_v0_arr,
                src_x_arr, src_y_arr, src_z_arr,
            )
        )
        ctx.save_for_backward(*tensors)
        ctx.scalars = (pixel_size, step_size, voxel_size, off_x, off_y, off_z)
        # Direct class.apply(...) callers may omit optional ROI arguments.
        ctx.num_inputs = len(ctx.needs_input_grad)
        return _native().forward(
            *tensors, pixel_size, detector_u, detector_v,
            step_size, voxel_size, off_x, off_y, off_z,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_projection):
        pixel_size, step_size, voxel_size, off_x, off_y, off_z = ctx.scalars
        gradients = _native().backward(
            grad_projection.contiguous(), *ctx.saved_tensors,
            pixel_size, step_size, voxel_size,
            bool(ctx.needs_input_grad[0]), off_x, off_y, off_z,
        )
        result = tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad[:9])
        ) + (None,) * 8
        return result[:ctx.num_inputs]


def project(volume, angles, sid, sod, det_u0_arr, det_v0_arr,
            src_x_arr, src_y_arr, src_z_arr, pixel_size,
            detector_u, detector_v, step_size=0.2, voxel_size=1.0,
            off_x=0.0, off_y=0.0, off_z=0.0):
    """Return physical ray-integral projections with layout [A,U,V].

    All nine tensors must be CUDA float32 on the same device. ``volume`` has
    shape [D,H,W], and each of the eight geometry tensors has shape [A].
    ``sid`` is the total SDD; source offsets use world coordinates and move
    only the source. The detector stays at local y = SOD - SDD. Angles are
    radians; detector center indices are in pixels. Distances, ``step_size``,
    ``voxel_size`` and ROI offsets share one physical length unit.

    The ROI uses x = (index_x - W/2) * voxel_size + off_x, and similarly for
    y and z. The integrator samples the trilinear volume at fixed physical
    steps. Its first-order gradients are piecewise derivatives: sample-count,
    interpolation-cell and box-intersection changes are nonsmooth. Scalar
    settings and ROI offsets have no gradients; higher derivatives are not
    supported. This ray integrator and the weighted FDK backprojector share
    geometry but are not a strict adjoint pair.
    """
    return DifferentiableSourceOnlyForwardProjector.apply(
        volume, angles, sid, sod, det_u0_arr, det_v0_arr,
        src_x_arr, src_y_arr, src_z_arr, pixel_size,
        detector_u, detector_v, step_size, voxel_size,
        off_x, off_y, off_z,
    )
