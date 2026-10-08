# diff-geoCali 可微 CBCT 投影算子

[English](../README.md)

这个项目为 PyTorch 提供 CUDA 加速的锥束 CT（CBCT）正投影和反投影。你可以用它把三维体数据转换成投影图像、把投影反投影到三维空间，以及计算几何参数的梯度。

当前代码包包含投影算子、FDK 重建辅助函数和运行示例。论文中的完整熵最小化、SNSF 和两阶段几何自校准流程尚未包含在这个代码包中。

## 可以做什么

- **正投影 `project`**：从三维体数据生成模拟投影。
- **反投影 `backproject`**：将投影映射到三维体数据。
- **自动求导**：计算输入数据及逐视角几何参数的一阶梯度，供几何优化使用。
- **FDK 重建 `reconstruct_fdk`**：提供滤波和反投影流程。
- **合成示例**：无需下载真实数据，即可了解调用方式。

每个视角可以设置自己的角度、源到探测器距离、源到原点距离、探测器中心和射线源位置偏移。

## 安装

文档中的编译流程使用 Linux。需要 Python 3.10 或以上版本、支持 CUDA 的 PyTorch 2.0 或以上版本、NVIDIA GPU 和驱动，以及包含 `nvcc` 的完整 CUDA Toolkit 和兼容的 C++ 编译器。

先安装与你本机 CUDA Toolkit 匹配的 PyTorch，再运行：

```bash
git clone https://github.com/TomSHINING/diff-geoCali.git
cd diff-geoCali
python -m pip install -r requirements-build.txt
MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps -e .
```

这会一次编译两个 CUDA 扩展。PyTorch 自带的 CUDA 运行库不包含 `nvcc`，因此仅安装 PyTorch 还不够。

可以先检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

运行 GPU 示例时，`torch.cuda.is_available()` 应为 `True`。

## 先运行一个示例

安装完成后，在仓库根目录运行：

```bash
python -m examples.forward       # 正投影、重建和求导
python -m examples.basic         # 反投影
python -m examples.autograd      # 几何参数梯度
python -m examples.fdk           # FDK 重建
```

每个示例都可以加 `--help` 查看参数。部分示例可以保存 `.pt` 文件；合成数据用于演示调用方式。

使用自己的预处理投影时，可运行 `python -m examples.fdk --input projections.pt`，并将距离、像素尺寸等参数设置为你的扫描参数。此示例使用完整圆轨迹和均匀角度；需要逐视角几何时，请调用 Python 接口。

## 输入数据怎么放

| 输入 | 要求 |
| --- | --- |
| 三维体数据 | `[D, H, W]`，对应 `[Z, Y, X]` |
| 投影 | `[A, U, V]`，分别为视角数、探测器宽度、探测器高度 |
| 几何参数 | 每个参数都是 `[A]` 张量，每个视角一个值 |
| 张量类型 | `torch.float32`，全部位于同一块 CUDA GPU |
| 角度 | 使用弧度 |
| 距离 | 全部使用相同单位，例如毫米 |
| `sid` / `sod` | 源到探测器距离 / 源到原点距离 |
| `det_u0`、`det_v0` | 探测器中心的像素索引 |
| 射线源偏移 | 世界坐标系下的 X/Y/Z 偏移，只移动射线源 |

如果你的投影存成 `[A, V, U]`，需要交换后两个维度。真实投影应提前完成暗场、平场校正和负对数转换，得到线积分数据。

`backproject` 本身只做加权反投影，不包含滤波或角度归一化。需要这些步骤时使用 `reconstruct_fdk`；这个辅助函数不启用自动求导。

## 使用时需要了解

- 支持一阶梯度；体素大小、ROI 平移等标量设置不参与求导。
- 几何梯度是分段导数，在采样或视野边界处可能发生变化。
- 正投影和加权反投影不是严格的伴随算子；正投影的转置运算可通过其体数据梯度获得。
- FDK 辅助函数假设完整圆轨迹、均匀角度采样，预权重使用居中的探测器坐标。几何有偏移时，它用于近似初始化。
- 探测器像素应为正方形，体素应为各向同性；不支持任意探测器倾斜和曲面探测器。

## 检查与详细说明

```bash
python -m examples.check_operator
python -m examples.check_forward_operator
```

以上检查需要 GPU。加 `--cpu-only` 仅检查 CPU 参考计算。

已有验证记录包括编译和 CPU 参考检查，CUDA 内核运行检查仍待完成。当前未提供许可证文件。

- [完整使用说明、坐标约定和常见问题](USAGE.md)
- [代码来源与打包说明](SOURCE.md)
- [验证记录](VALIDATION.md)
