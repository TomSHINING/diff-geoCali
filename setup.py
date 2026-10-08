"""Build the standalone source-aware cone-beam projection extensions."""

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="cbct-backprojector-cuda",
    version="0.1.0",
    description="Differentiable source-aware cone-beam projection operators for PyTorch",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["source_only_forward_projector"],
    install_requires=["torch>=2.0"],
    ext_modules=[
        CUDAExtension(
            name="src_cbct_backprojector_cuda",
            sources=["csrc/src_cbct_backprojector_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            name="src_cbct_forwardprojector_cuda",
            sources=["csrc/src_cbct_forwardprojector_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
