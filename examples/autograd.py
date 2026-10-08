"""Compute gradients for projections and all eight per-view geometry tensors."""

import argparse
import math

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and a CUDA-enabled PyTorch installation are required.")

    from cbct_backprojector import backproject, detector_centers

    device = torch.device("cuda")
    views, detector_u, detector_v = 8, 24, 20
    pixel_size, voxel_size = 1.0, 0.7
    angles = torch.arange(views, dtype=torch.float32, device=device)
    angles = (angles * (2.0 * math.pi / views) + 0.13).requires_grad_()

    u = torch.arange(detector_u, device=device, dtype=torch.float32)[:, None]
    v = torch.arange(detector_v, device=device, dtype=torch.float32)[None, :]
    profile = torch.exp(-((u - 10.7).square() + (v - 8.4).square()) / 45.0)
    projections = (
        (1.0 + 0.2 * angles.detach().cos())[:, None, None] * profile[None]
    ).requires_grad_()
    sid = torch.full_like(angles, 90.0, requires_grad=True)
    sod = torch.full_like(angles, 60.0, requires_grad=True)
    det_u0, det_v0 = detector_centers(
        torch.zeros_like(angles), torch.zeros_like(angles),
        detector_u, detector_v, pixel_size,
    )
    det_u0 = det_u0.detach().requires_grad_()
    det_v0 = det_v0.detach().requires_grad_()
    src_x = torch.full_like(angles, 0.10, requires_grad=True)
    src_y = torch.full_like(angles, -0.08, requires_grad=True)
    src_z = torch.full_like(angles, 0.12, requires_grad=True)

    volume = backproject(
        projections, angles, sid, sod, det_u0, det_v0,
        src_x, src_y, src_z, pixel_size, (6, 8, 10), voxel_size,
        volume_offset=(0.2, -0.1, 0.15),
    )
    loss = volume.square().mean()
    loss.backward()
    print(f"loss = {loss.item():.6f}")
    names = (
        "projections", "angles", "sid", "sod", "det_u0", "det_v0",
        "src_x", "src_y", "src_z",
    )
    parameters = (
        projections, angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z,
    )
    for name, parameter in zip(names, parameters):
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"Missing or non-finite gradient for {name}")
        print(f"{name:12s} grad norm = {gradient.norm().item():.6e}")

    # Use an optimizer on selected geometry leaves for a real calibration loss.
    # pixel_size, voxel_size, volume_shape and volume_offset are fixed settings.


if __name__ == "__main__":
    main()
