"""Project a synthetic volume, reconstruct it, and inspect nine input gradients.

Run after installing the package: ``python examples/forward.py --help``.
The volume and both reconstruction calls use identical world-coordinate ROI
offsets. Saved tensors use volume [D,H,W] and projection [A,U,V] layouts.
"""

import argparse
import math
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--detector-u", type=int, default=48)
    parser.add_argument("--detector-v", type=int, default=40)
    parser.add_argument("--volume-shape", type=int, nargs=3, default=(16, 20, 24), metavar=("D", "H", "W"))
    parser.add_argument("--voxel-size", type=float, default=0.7)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--volume-offset", type=float, nargs=3, default=(0.2, -0.1, 0.15), metavar=("X", "Y", "Z"))
    parser.add_argument("--sid", type=float, default=90.0, help="total source-to-detector distance")
    parser.add_argument("--sod", type=float, default=60.0, help="nominal source-to-origin distance")
    parser.add_argument("--src-offset", type=float, nargs=3, default=(0.1, -0.08, 0.12), metavar=("X", "Y", "Z"))
    parser.add_argument("--delta-u", type=float, default=0.15, help="physical detector U offset")
    parser.add_argument("--delta-v", type=float, default=-0.10, help="physical detector V offset")
    parser.add_argument("--angle-chunk-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("forward_demo.pt"))
    return parser, parser.parse_args()


def synthetic_volume(shape, voxel_size, offset, device):
    """Smooth asymmetric object defined on the CUDA kernel's voxel coordinates."""
    depth, height, width = shape
    x = (torch.arange(width, device=device, dtype=torch.float32) - width * 0.5) * voxel_size + offset[0]
    y = (torch.arange(height, device=device, dtype=torch.float32) - height * 0.5) * voxel_size + offset[1]
    z = (torch.arange(depth, device=device, dtype=torch.float32) - depth * 0.5) * voxel_size + offset[2]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    # Positions are specified in world coordinates relative to the same ROI.
    radius = min(shape) * voxel_size / 5
    main = torch.exp(-(
        ((xx - offset[0] - radius * 0.30) / radius).square()
        + ((yy - offset[1] + radius * 0.20) / (radius * 0.85)).square()
        + ((zz - offset[2] - radius * 0.15) / (radius * 0.70)).square()
    ))
    satellite = 0.35 * torch.exp(-(
        ((xx - offset[0] + radius * 0.9) / (radius * 0.4)).square()
        + ((yy - offset[1] - radius * 0.6) / (radius * 0.5)).square()
        + ((zz - offset[2] + radius * 0.2) / (radius * 0.4)).square()
    ))
    return (main + satellite).contiguous().requires_grad_()


def main():
    parser, args = parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and a CUDA-enabled PyTorch installation are required.")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device must select a CUDA device")
    if args.angles < 1 or min(args.detector_u, args.detector_v) < 2:
        parser.error("--angles must be positive and detector dimensions must be >= 2")
    if min(args.volume_shape) < 1 or args.angle_chunk_size < 1:
        parser.error("volume dimensions and --angle-chunk-size must be positive")
    if not all(math.isfinite(value) and value > 0 for value in (
        args.pixel_size, args.voxel_size, args.step_size, args.sid, args.sod,
    )):
        parser.error("pixel/voxel/step sizes and distances must be finite and positive")
    if not args.sid > args.sod:
        parser.error("--sid must be greater than --sod")
    if not all(math.isfinite(value) for value in (
        *args.volume_offset, *args.src_offset, args.delta_u, args.delta_v,
    )):
        parser.error("ROI, source and detector offsets must be finite")

    from cbct_backprojector import backproject, detector_centers
    from cbct_backprojector.filtering import reconstruct_fdk
    from cbct_backprojector.source_only_forward_projector import project

    def constant(value):
        return torch.full((args.angles,), value, dtype=torch.float32, device=device)

    volume = synthetic_volume(args.volume_shape, args.voxel_size, args.volume_offset, device)
    angles = torch.arange(args.angles, dtype=torch.float32, device=device)
    angles = (angles * (2 * math.pi / args.angles) + 0.13).requires_grad_()
    sid, sod = constant(args.sid).requires_grad_(), constant(args.sod).requires_grad_()
    det_u0, det_v0 = detector_centers(
        constant(args.delta_u), constant(args.delta_v),
        args.detector_u, args.detector_v, args.pixel_size,
    )
    det_u0, det_v0 = det_u0.requires_grad_(), det_v0.requires_grad_()
    src_x, src_y, src_z = (constant(value).requires_grad_() for value in args.src_offset)
    geometry = (angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z)
    projections = project(
        volume, *geometry, args.pixel_size,
        args.detector_u, args.detector_v, args.step_size, args.voxel_size,
        *args.volume_offset,
    )
    loss = projections.square().mean()
    loss.backward()
    print(f"projections: {tuple(projections.shape)}; loss = {loss.item():.6g}")
    for name, tensor in zip(
        ("volume", "angles", "sid", "sod", "det_u0", "det_v0", "src_x", "src_y", "src_z"),
        (volume, *geometry),
    ):
        gradient = tensor.grad
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"Missing or non-finite gradient for {name}")
        print(f"{name:10s} grad norm = {gradient.norm().item():.6e}")

    # Inference outputs are separate from the differentiable ray-integral loss.
    with torch.no_grad():
        detached_geometry = tuple(value.detach() for value in geometry)
        raw_backprojection = backproject(
            projections.detach(), *detached_geometry, args.pixel_size,
            tuple(args.volume_shape), args.voxel_size,
            volume_offset=tuple(args.volume_offset),
        )
        reconstruction = reconstruct_fdk(
            projections.detach(), *detached_geometry, args.pixel_size,
            tuple(args.volume_shape), args.voxel_size,
            volume_offset=tuple(args.volume_offset),
            angle_chunk_size=args.angle_chunk_size,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "volume": volume.detach().cpu(),
        "projections": projections.detach().cpu(),
        "backprojection": raw_backprojection.cpu(),
        "fdk": reconstruction.cpu(),
        "geometry": {name: value.detach().cpu() for name, value in zip(
            ("angles", "sid", "sod", "det_u0", "det_v0", "src_x", "src_y", "src_z"),
            geometry,
        )},
        "pixel_size": args.pixel_size,
        "voxel_size": args.voxel_size,
        "step_size": args.step_size,
        "volume_offset": tuple(args.volume_offset),
    }, args.output)
    print(f"Saved original volume, projections, backprojection and FDK to {args.output}")
    print("FDK uses the project's centered cosine-weight approximation; the two operators are not strict adjoints.")


if __name__ == "__main__":
    main()
