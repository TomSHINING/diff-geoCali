"""Backproject small synthetic projections and optionally save a CPU tensor."""

import argparse
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output .pt file")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and a CUDA-enabled PyTorch installation are required.")

    from cbct_backprojector import backproject, detector_centers

    device = torch.device("cuda")
    views, detector_u, detector_v = 16, 48, 40
    pixel_size, voxel_size = 1.0, 0.8
    volume_shape = (16, 20, 24)  # (D, H, W), corresponding to (z, y, x).
    angles = torch.arange(views, device=device, dtype=torch.float32)
    angles = angles * (2.0 * math.pi / views)  # Radians; omit the duplicate endpoint.

    u = torch.arange(detector_u, device=device, dtype=torch.float32)
    v = torch.arange(detector_v, device=device, dtype=torch.float32)
    u = (u - (detector_u - 1) / 2) / 10.0
    v = (v - (detector_v - 1) / 2) / 8.0
    profile = torch.exp(-0.5 * (u[:, None].square() + v[None, :].square()))
    projections = (1.0 + 0.15 * angles.cos())[:, None, None] * profile[None]
    # These smooth values demonstrate the API; they are not measured CT data.

    sid = torch.full_like(angles, 180.0)
    sod = torch.full_like(angles, 120.0)
    zero = torch.zeros_like(angles)
    det_u0, det_v0 = detector_centers(
        zero, zero, detector_u, detector_v, pixel_size
    )
    volume = backproject(
        projections, angles, sid, sod, det_u0, det_v0,
        zero, zero, zero, pixel_size, volume_shape, voxel_size,
        volume_offset=(0.0, 0.0, 0.0),
    )
    print(f"Projection shape [A, U, V]: {tuple(projections.shape)}")
    print(f"Volume shape [D, H, W]: {tuple(volume.shape)}")
    print(f"Volume range: [{volume.min().item():.6f}, {volume.max().item():.6f}]")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(volume.detach().cpu(), args.output)
        print(f"Saved CPU volume tensor to {args.output}")


if __name__ == "__main__":
    main()
