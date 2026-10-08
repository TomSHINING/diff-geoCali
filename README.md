# diff-geoCali: Differentiable CBCT Operators

[中文说明](docs/README.zh-CN.md)

CUDA forward and backprojection for cone-beam CT (CBCT), with a simple PyTorch interface.
Use these operators to simulate X-ray projections, reconstruct a volume, or calculate gradients for geometry optimization.

This release contains the projection operators and an FDK reconstruction helper. The full two-stage entropy/SNSF geometry calibration pipeline is not included.

## What is included?

- **Forward projection**: turn a 3D volume into simulated projection images.
- **Backprojection**: map projection images back into a 3D volume.
- **PyTorch autograd**: calculate first-order gradients for the input data and per-view geometry.
- **FDK helper**: filter and backproject data from a complete, uniformly sampled circular scan.
- **Small examples**: try the code with synthetic data; no external dataset is needed.

The geometry can vary for each view: rotation angle, source-to-detector distance, source-to-origin distance, detector center, and source position offset.

## Installation

The documented build workflow uses Linux. You need:

- Python 3.10 or newer and CUDA-enabled PyTorch 2.0 or newer.
- An NVIDIA GPU and a working driver to run the operators.
- The full CUDA Toolkit, including `nvcc`, and a compatible C++ compiler to build them.

Install PyTorch first, using a CUDA version that matches your local Toolkit. Then run:

```bash
git clone https://github.com/TomSHINING/diff-geoCali.git
cd diff-geoCali
python -m pip install -r requirements-build.txt
MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps -e .
```

This builds both CUDA extensions. PyTorch's bundled CUDA runtime alone does not include `nvcc`.
Check your environment with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

`torch.cuda.is_available()` must be `True` to run the GPU examples.

## Try an example

Run these commands from the repository root after installation:

```bash
python -m examples.forward       # Simulate projections, reconstruct, and calculate gradients
python -m examples.basic         # Backprojection
python -m examples.autograd      # Geometry gradients
python -m examples.fdk           # FDK reconstruction
```

Use `--help` to see each example's options. Examples save small `.pt` files where applicable.
For example, `examples.fdk` accepts your own preprocessed projection tensor with `--input projections.pt`.
Set the distances and pixel size to match your scan; the defaults are for synthetic data.
The FDK example uses a complete circle with uniform angles. Use the Python interface for per-view geometry.

## Minimal Python example

```python
import math
import torch
from cbct_backprojector import project, detector_centers

A, U, V = 24, 48, 40              # Views, detector width, detector height
device = "cuda"
volume = torch.rand(16, 20, 24, device=device, requires_grad=True)
angles = torch.arange(A, device=device, dtype=torch.float32) * (2 * math.pi / A)
sid = torch.full((A,), 180.0, device=device)  # Source-to-detector distance
sod = torch.full((A,), 120.0, device=device)  # Source-to-origin distance
zero = torch.zeros(A, device=device)
u0, v0 = detector_centers(zero, zero, U, V, pixel_size=1.0)
src_x = torch.zeros(A, device=device, requires_grad=True)

projections = project(
    volume, angles, sid, sod, u0, v0, src_x, zero, zero,
    pixel_size=1.0, detector_u=U, detector_v=V,
    step_size=0.5, voxel_size=1.0,
)
projections.square().mean().backward()  # Example loss
print(projections.shape)              # [24, 48, 40]
print(volume.grad.norm(), src_x.grad.norm())
```

Replace the example loss with the objective required by your reconstruction or calibration method.

## Input conventions

| Item | Convention |
| --- | --- |
| Volume | `[D, H, W]`, corresponding to `[Z, Y, X]` |
| Projections | `[A, U, V]`; transpose data stored as `[A, V, U]` |
| Geometry tensors | One value per view, shape `[A]` |
| Tensor type | `torch.float32`, all on the same CUDA device |
| Angles | Radians |
| Distances | One consistent unit, such as millimeters |
| `sid` / `sod` | Total source-to-detector / source-to-origin distance |
| `det_u0`, `det_v0` | Detector center indices in pixels |
| Source offsets | World-coordinate X/Y/Z offsets; move the source only |

`backproject` returns a weighted sum without filtering or angle normalization.
Use `reconstruct_fdk` for the supplied filter-and-backproject workflow; this helper disables autograd.
Real projections must already have dark/flat-field correction and negative-log conversion to line integrals.

## Important limits

- Only first-order derivatives are supported. Scalar settings such as voxel size and ROI offsets have no gradients.
- Geometry gradients are piecewise derivatives and may change at sampling or field-of-view boundaries.
- Forward projection and weighted backprojection are not a strict adjoint pair. For the forward operator's transpose, use its autograd volume gradient.
- The FDK helper assumes a complete circular scan with uniform angles and uses centered detector preweights. It is an approximate initializer when geometry is perturbed.
- Pixels must be square and voxels isotropic. Arbitrary detector tilt and curved detectors are not supported.

## Checks and further details

```bash
python -m examples.check_operator
python -m examples.check_forward_operator
```

Both checks require a GPU. Add `--cpu-only` to check the CPU reference calculations only.
The existing validation record reports successful compilation and CPU checks; CUDA kernel runtime checks remain pending.

- [Detailed usage, coordinates, and troubleshooting](docs/USAGE.md)
- [Code origin and packaging notes](docs/SOURCE.md)
- [Validation record](docs/VALIDATION.md)

No license file is currently provided.
