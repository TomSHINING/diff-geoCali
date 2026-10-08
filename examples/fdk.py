"""Reconstruct a tensor of line integrals, or a synthetic sphere, using FDK.

After installing the package, run ``python examples/fdk.py --help``. The input
file, when supplied, must contain a torch-saved tensor with layout [A, U, V].
The example uses a complete circular scan with uniform angular spacing.
"""

import argparse
import math
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="torch.save tensor [A, U, V], already converted to line integrals")
    parser.add_argument("--output", type=Path, default=Path("volume.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--angles", type=int, default=90, help="view count for synthetic data")
    parser.add_argument("--detector-u", type=int, default=128, help="U size for synthetic data")
    parser.add_argument("--detector-v", type=int, default=128, help="V size for synthetic data")
    parser.add_argument("--sid", type=float, default=1000.0, help="source-to-detector distance")
    parser.add_argument("--sod", type=float, default=500.0, help="source-to-origin distance")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--volume-shape", type=int, nargs=3, default=(64, 64, 64), metavar=("D", "H", "W"))
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--volume-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--delta-u", type=float, default=0.0, help="detector U offset in physical length units")
    parser.add_argument("--delta-v", type=float, default=0.0, help="detector V offset in physical length units")
    parser.add_argument("--sphere-radius", type=float, default=10.0, help="radius for synthetic unit-density sphere")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--angle-chunk-size", type=int, default=32)
    return parser.parse_args()


def sphere_projections(args, device):
    """Compute analytic line integrals for a sphere at the rotation origin."""
    if args.angles < 1 or min(args.detector_u, args.detector_v) < 2:
        raise ValueError("angles must be positive; detector-u and detector-v must be >= 2")
    if not 0.0 < args.sphere_radius < args.sod:
        raise ValueError("sphere-radius must be positive and smaller than sod")
    if args.sid <= args.sod + args.sphere_radius:
        raise ValueError("sid must place the detector beyond the synthetic sphere")
    u = (torch.arange(args.detector_u, dtype=torch.float32, device=device) - (args.detector_u - 1) / 2) * args.pixel_size + args.delta_u
    v = (torch.arange(args.detector_v, dtype=torch.float32, device=device) - (args.detector_v - 1) / 2) * args.pixel_size + args.delta_v
    radial_squared = u[:, None].square() + v[None, :].square()
    ray_distance_squared = args.sod**2 * radial_squared / (args.sid**2 + radial_squared)
    one_view = 2.0 * torch.sqrt((args.sphere_radius**2 - ray_distance_squared).clamp_min(0.0))
    # A centered sphere has the same ideal projection at every rotation angle.
    return one_view.unsqueeze(0).expand(args.angles, -1, -1).contiguous()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU and CUDA-enabled PyTorch are required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    if not math.isfinite(args.pixel_size) or args.pixel_size <= 0:
        raise ValueError("pixel-size must be finite and positive")

    from cbct_backprojector.filtering import reconstruct_fdk
    from cbct_backprojector.operator import detector_centers

    if args.input is None:
        projections = sphere_projections(args, device)
    else:
        projections = torch.load(args.input, map_location="cpu", weights_only=True)
        if not isinstance(projections, torch.Tensor) or projections.ndim != 3:
            raise ValueError("--input must contain a torch.Tensor with shape [A, U, V]")
        projections = projections.to(device=device, dtype=torch.float32).contiguous()
    num_angles, num_u, num_v = projections.shape
    if num_angles < 1 or min(num_u, num_v) < 2:
        raise ValueError("projection dimension A must be positive, and U and V must be >= 2")

    def constant(value):
        return torch.full((num_angles,), value, dtype=torch.float32, device=device)

    angles = torch.arange(num_angles, dtype=torch.float32, device=device) * (2.0 * math.pi / num_angles)
    sid, sod = constant(args.sid), constant(args.sod)
    det_u0, det_v0 = detector_centers(
        constant(args.delta_u), constant(args.delta_v), num_u, num_v, args.pixel_size
    )
    zeros = constant(0.0)
    print(f"Reconstructing {num_angles} views, detector ({num_u}, {num_v}), volume {tuple(args.volume_shape)}")
    volume = reconstruct_fdk(
        projections, angles, sid, sod, det_u0, det_v0, zeros, zeros, zeros,
        args.pixel_size, tuple(args.volume_shape), args.voxel_size,
        volume_offset=tuple(args.volume_offset), alpha=args.alpha,
        angle_chunk_size=args.angle_chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(volume.cpu(), args.output)
    print(f"Saved {args.output} with shape {tuple(volume.shape)} and dtype {volume.dtype}")
    print(f"Value range: [{volume.min().item():.6g}, {volume.max().item():.6g}]")


if __name__ == "__main__":
    main()
