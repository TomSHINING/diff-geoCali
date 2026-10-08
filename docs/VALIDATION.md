# 验证记录

日期：2026-10-08。本页区分实际执行结果与仍需 GPU 执行的检查。

## 本机环境

| 项目 | 版本 / 状态 |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA 运行库 | 12.8 |
| 本机 CUDA Toolkit / nvcc | 12.1 / V12.1.105 |
| C++ 编译器 | g++ 11.4.0 |
| 构建目标 | `TORCH_CUDA_ARCH_LIST=8.6` |
| GPU 运行能力 | `torch.cuda.is_available() == False`；NVIDIA 驱动不可访问 |

当前默认 Python 没有 PyTorch，验证使用项目已有的 Python 3.10 / PyTorch 环境。CUDA Toolkit 12.1 与 PyTorch CUDA 12.8 存在次版本差异，构建工具发出警告；这里记录实际结果，不把该组合列为推荐部署环境。部署时仍建议使用匹配的 Toolkit 和 PyTorch。

## 已执行的检查

- 正投影和反投影扩展分别完成 `nvcc` 编译及链接；打包构建入口会同时构建两个扩展。
- `import torch` 后，两个原生扩展及 Python 包均可导入；原 `source_only_forward_projector` 兼容入口与包内 `project` 指向同一函数。
- 反投影 CPU 手工双线性参考与独立 Python 双精度标量计算一致；9 组 Tensor 输入各选 2 个样本做中心有限差分，全部通过，最大绝对误差约 `5.06e-9`。
- 正投影 CPU 手工射线积分参考通过常量体积、零体积、线性性和 9 组梯度有限值检查；体积、源 Z 偏移与探测器 U/V 中心的局部有限差分通过。几何差分检查确认入射面、积分步数和插值分段保持不变。
- FDK 滤波与主线 Python 公式在多组非方形检测器上比较，最大绝对误差不超过 `1.2e-7`；分块滤波与整批滤波一致。
- Python 文件语法、全部 6 个示例 CLI 帮助和源码来源校验通过。

- 源码发行包（sdist）与 wheel 均构建成功；wheel 包含两个原生扩展和原导入路径兼容模块，解压到独立临时目录后导入通过。
- sdist 包含构建文件、两个 CUDA 源文件、文档和示例，未包含历史二进制或真实数据。交付的 ZIP / tar.gz 另逐文件校验与工作源码一致，并提供压缩包 SHA-256。

## 尚未执行的检查

本机无法运行 CUDA 内核，因此未执行 GPU 前向数值对照、9 组真实 CUDA 梯度对照、正投影局部几何有限差分、GPU FDK 重建示例或跨 CUDA stream 的运行检查。编译及 CPU 检查通过不代表这些 GPU 检查已经通过。

有匹配环境和 NVIDIA GPU 的机器上，可运行：

```bash
python -m examples.check_operator
python -m examples.check_forward_operator
python -m examples.basic
python -m examples.autograd
python -m examples.forward
python -m examples.fdk
```

两个检查程序在没有 GPU 时会明确失败；使用 `--cpu-only` 只检查 CPU 参考，不会把跳过 CUDA 当作通过。
