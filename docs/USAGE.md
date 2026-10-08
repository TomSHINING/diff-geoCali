# CBCT Projectors CUDA

面向 PyTorch 的可微锥束 CT 正投影与反投影算子，提供统一 CUDA 编译入口、Python 自动求导封装、源偏移几何和 FDK 使用示例。

代码提取自 CoMoCo 项目当前重建与几何标定脚本使用的 `Differenential_forward/src_cbct_backprojector_cuda.cu`、`src_cbct_forwardprojector_cuda.cu` 和 `source_only_forward_projector.py`。支持逐投影角度、SOD/SDD、探测器中心偏移、世界坐标下的源位置偏移，以及重建 ROI 平移。具体来源和打包改动见 [SOURCE.md](SOURCE.md)。

## 目录结构

```text
cbct-backprojector/
├── csrc/
│   ├── src_cbct_backprojector_cuda.cu   # 反投影 forward / backward
│   └── src_cbct_forwardprojector_cuda.cu # 正投影 forward / backward
├── cbct_backprojector/
│   ├── operator.py                     # 自动求导和几何接口
│   ├── source_only_forward_projector.py # 正投影自动求导，保留原 project API
│   └── filtering.py                    # 项目主线 FDK 滤波与分块重建
├── source_only_forward_projector.py    # 原项目导入路径的兼容入口
├── examples/
│   ├── basic.py                        # 合成投影 → 反投影
│   ├── autograd.py                     # 投影与几何参数梯度
│   ├── fdk.py                          # FDK 重建，可读取自己的投影
│   ├── forward.py                      # 体积 → 正投影 → 重建与求导
│   ├── check_operator.py               # 反投影 CPU 参考、梯度与 CUDA 对照
│   └── check_forward_operator.py       # 正投影射线积分与梯度检查
├── docs/
│   ├── SOURCE.md                       # 来源、提取范围和包装改动
│   ├── source_manifest.json            # 原源码及打包源码 SHA-256
│   └── VALIDATION.md                   # 本次实际验证结果
├── setup.py
├── pyproject.toml
├── tools/package_source.py             # 重新生成干净源码压缩包
└── requirements-build.txt
```

包只依赖 PyTorch 和 CUDA 构建工具。示例使用合成数据，不需要原项目的数据集、配置、个人路径或已编译的 `.so`。

## 编译与安装

需要 Linux、Python 3.10 或更高版本、带 CUDA 的 PyTorch、CUDA Toolkit（包含 `nvcc`），以及该 Toolkit 支持的 C++ 编译器。运行算子还需要可用的 NVIDIA GPU 和驱动。`pip` 安装的 PyTorch CUDA 运行库不包含完整编译工具链。

先在独立环境中安装与本机 CUDA Toolkit 匹配的 PyTorch，再执行以下步骤。确认三项检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
nvidia-smi
```

用于运行示例的 `torch.cuda.is_available()` 应为 `True`。建议 PyTorch 的 `torch.version.cuda` 与 `nvcc` 的 CUDA 主次版本一致；CUDA 主版本不一致通常会导致编译失败。

在仓库根目录编译并安装：

```bash
cd diff-geoCali
python -m pip install -r requirements-build.txt
MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps -e .
```

这里关闭构建隔离，直接使用当前环境已安装的 PyTorch 编译扩展；`--no-deps` 避免构建时替换该 PyTorch。一次构建会生成 `src_cbct_backprojector_cuda` 与 `src_cbct_forwardprojector_cuda` 两个扩展，以及 `cbct_backprojector` Python 包。若需要普通安装，可把 `-e .` 改为 `.`。

也可以只在当前目录编译，用于本地调试：

```bash
MAX_JOBS=2 python setup.py build_ext --inplace
python -c "import torch; import src_cbct_backprojector_cuda; import src_cbct_forwardprojector_cuda; import cbct_backprojector; print('import OK')"
```

这种方式请从仓库根目录以 `python -m examples.basic` 运行示例。`examples` 是 Python 的命名空间目录，无须额外安装。

如果 Toolkit 不在默认路径，设置 `CUDA_HOME`，例如：

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
```

没有 GPU 的构建机器上还需要显式指定目标 GPU 架构。例如目标 GPU 的计算能力为 8.6 时：

```bash
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=2 python setup.py build_ext --inplace
```

将 `8.6` 替换为实际部署 GPU 的计算能力；可以使用 `"8.0;8.6"` 构建多个目标。编译成功不代表无 GPU 的机器能运行算子。

## 快速运行

```bash
python -m examples.basic --output backprojection.pt
python -m examples.autograd
python -m examples.fdk --output fdk_volume.pt
python -m examples.forward
python -m examples.check_operator
python -m examples.check_forward_operator
```

以上示例和完整 CUDA 检查需要 GPU。合成数据用于演示调用方式；示例输出不用于判断真实扫描的重建质量。无 GPU 时可以执行 CPU 参考实现检查：

```bash
python -m examples.check_operator --cpu-only
python -m examples.check_forward_operator --cpu-only
```

这项 CPU 检查不代表 CUDA 内核已通过运行验证。本次实际编译与检查结果见 [VALIDATION.md](VALIDATION.md)。

## Python 使用示例

```python
import math
import torch
from cbct_backprojector import backproject, detector_centers

device = torch.device("cuda:0")
A, U, V = 60, 128, 96
pixel_size = 0.8                  # mm
voxel_size = 0.5                  # mm
volume_shape = (64, 64, 64)       # D,H,W = Z,Y,X

# 替换为你的投影：形状必须是 [A,U,V]。
projections = torch.rand(A, U, V, device=device, dtype=torch.float32)
angles = torch.arange(A, device=device, dtype=torch.float32) * (2 * math.pi / A)
sod = torch.full((A,), 500.0, device=device)
sid = torch.full((A,), 1000.0, device=device)  # 总 SDD，单位 mm
zero = torch.zeros(A, device=device)
det_u0, det_v0 = detector_centers(zero, zero, U, V, pixel_size)

volume = backproject(
    projections, angles, sid, sod,
    det_u0, det_v0, zero, zero, zero,
    pixel_size, volume_shape, voxel_size,
    volume_offset=(0.0, 0.0, 0.0),  # 世界坐标 x,y,z，单位 mm
)
print(volume.shape)                # torch.Size([64, 64, 64])
torch.save(volume.detach().cpu(), "volume.pt")
```

`backproject` 返回带几何权重的投影累加结果。完整 FDK 还需要滤波和角度归一化，可使用下面的接口：

```python
from cbct_backprojector import reconstruct_fdk

with torch.no_grad():
    volume = reconstruct_fdk(
        projections, angles, sid, sod,
        det_u0, det_v0, zero, zero, zero,
        pixel_size, volume_shape, voxel_size,
        alpha=0.8, angle_chunk_size=16,
    )
```

## 正投影：`source_only_forward_projector.project`

正投影 CUDA 为 `csrc/src_cbct_forwardprojector_cuda.cu`，Python 自动求导封装为 `cbct_backprojector/source_only_forward_projector.py`。两种导入方式均可使用：

```python
from cbct_backprojector import project
# 原项目代码的导入方式也兼容：
from source_only_forward_projector import project
# 原脚本中的函数别名也可以保持：
from source_only_forward_projector import project as source_only_forward_project
```

API 保留原项目参数顺序和默认值：

```python
projections = project(
    volume, angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z,
    pixel_size, detector_u, detector_v,
    step_size=0.2, voxel_size=1.0,
    off_x=0.0, off_y=0.0, off_z=0.0,
)
# 输入 volume: CUDA float32 [D,H,W]；输出: [A,detector_u,detector_v]
```

`step_size` 是射线积分的**物理步长**，与体素尺寸使用相同距离单位；例如 `voxel_size=0.5 mm` 时可从 `step_size=0.25 mm` 开始选择。减小步长会增加采样数和耗时。正投影使用与反投影一致的角度、SDD/SOD、中心像素索引、世界坐标源偏移和体素坐标约定。其 ROI 使用三个独立参数 `off_x/off_y/off_z`；传给反投影时对应 `volume_offset=(off_x,off_y,off_z)`。

下面生成球形体积，计算投影，并对体积和源偏移求导：

```python
import math
import torch
from cbct_backprojector import project, detector_centers

device = torch.device("cuda:0")
A, U, V = 24, 48, 40
voxel_size, pixel_size = 1.0, 1.0
D, H, W = 16, 20, 24
z, y, x = torch.meshgrid(
    torch.arange(D, device=device, dtype=torch.float32),
    torch.arange(H, device=device, dtype=torch.float32),
    torch.arange(W, device=device, dtype=torch.float32), indexing="ij",
)
radius2 = ((x-W/2)*voxel_size)**2 + ((y-H/2)*voxel_size)**2 + ((z-D/2)*voxel_size)**2
volume = (radius2 <= 5.0**2).float().requires_grad_()
angles = torch.arange(A, device=device, dtype=torch.float32) * (2*math.pi/A)
sod = torch.full((A,), 120.0, device=device)
sid = torch.full((A,), 180.0, device=device)
zero = torch.zeros(A, device=device)
det_u0, det_v0 = detector_centers(zero, zero, U, V, pixel_size)
src_x = torch.zeros(A, device=device, requires_grad=True)

projections = project(
    volume, angles, sid, sod, det_u0, det_v0, src_x, zero, zero,
    pixel_size, U, V, step_size=0.5, voxel_size=voxel_size,
)
loss = projections.square().mean()  # 示例损失；实际标定请使用观测投影误差等目标
loss.backward()
print(projections.shape, volume.grad.norm(), src_x.grad.norm())
torch.save(projections.detach().cpu(), "projections.pt")
```

正投影自动求导支持体积和 8 个逐角度几何 Tensor 的一阶导数，共 9 组梯度。尺寸、像素/体素大小、积分步长与 ROI 标量不求导。体积不需要梯度时，封装会跳过体积梯度缓冲区及其归约。直接调用原生扩展 `src_cbct_forwardprojector_cuda.forward(...)` 不会注册自动求导；推荐调用 `project`。

正投影的离散积分具体为：每个探测器像素发出一条源到像素方向的射线，与体积的轴对齐包围盒求交，从入射位置开始，每隔 `step_size` 三线性采样并累加 `value * step_size`。包围盒 X 范围为 `[(-W/2-0.5)*voxel_size+off_x, (W/2-0.5)*voxel_size+off_x]`，Y/Z 同理；边缘三线性邻点索引会截到边缘体素。积分沿射线到体积出射点，未额外截到探测器终点，因此扫描几何应使探测器位于体积之外。

该离散积分包含入射采样点，末采样点也乘完整步长，没有末段长度修正。几何反向传播包含射线归一化、当前入射面交点和三线性空间梯度，但不对出射点决定的离散步数变化、入射面切换或命中/未命中切换求导。因此只在采样分段保持不变的局部区域使用几何梯度；步数和边界变化处可能不连续。体积梯度则对应当前离散采样的三线性权重归约。

两种算子共享扫描几何。`project` 计算近似物理线积分，`backproject` 包含 FDK 几何权重，**二者不是严格伴随算子**。需要正投影的精确离散转置时，使用它自己的自动求导体积梯度，而非把 `backproject` 当成转置。

可运行完整示例及正投影检查：

```bash
python -m examples.forward --help
python -m examples.forward
python -m examples.check_forward_operator
```

## 参数与坐标约定

所有 9 个 Tensor 输入须为同一 GPU 上的 `torch.float32`。Python 包会检查形状，自动连续化输入并保留求导链。输入数值须有限，几何应具有物理意义。距离单位可使用 mm，所有距离参数必须一致。

| 参数 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `projections` | `[A,U,V]` Tensor | A 个角度，U 为探测器水平方向，V 为竖直方向 |
| `angles` | `[A]` Tensor | 实际投影角度，单位 rad |
| `sid` | `[A]` Tensor | 总源探距离 SDD；原 CUDA 的参数名称保留为 sid |
| `sod` | `[A]` Tensor | 源到旋转中心距离 SOD |
| `det_u0`, `det_v0` | 各 `[A]` Tensor | 零射线的探测器中心像素索引 |
| `src_x`, `src_y`, `src_z` | 各 `[A]` Tensor | 世界坐标源偏移，距离单位；只移动源 |
| `pixel_size` | 正标量 | U/V 共用的探测器像素尺寸 |
| `volume_shape` | `(D,H,W)` | 输出尺寸，对应 `(Z,Y,X)` |
| `voxel_size` | 正标量 | X/Y/Z 共用的体素尺寸 |
| `volume_offset` | `(off_x,off_y,off_z)` | ROI 世界坐标平移，默认 `(0,0,0)` |

反投影及共用的 `detector_centers` 工具需要 `U,V >= 2`。不支持非等距体素、U/V 不同像素尺寸、探测器任意倾斜或曲面探测器。反投影内核使用 32 位索引，单次调用的投影张量或体积不能超过 `2^31-1` 个元素，D/H 还受三维 CUDA grid 上限限制；通常 CT 体积尺寸远小于该限制。

常见存储布局为 `[A,V,U]` 时，先转置：

```python
projections = projections_avu.permute(0, 2, 1).contiguous()
```

无探测器偏移时，中心索引为 `(U-1)/2` 和 `(V-1)/2`。与当前项目 CSV 约定一致，探测器毫米偏移 `delta_u / delta_v` 转换为：

```python
det_u0 = (U - 1.0) / 2.0 - delta_u / pixel_size
det_v0 = (V - 1.0) / 2.0 - delta_v / pixel_size
angles = ideal_angles + delta_beta  # delta_beta 使用弧度
```

体积元素 `[z,y,x]` 的世界坐标严格按原内核计算：

```text
vx = (x - W/2) * voxel_size + off_x
vy = (y - H/2) * voxel_size + off_y
vz = (z - D/2) * voxel_size + off_z
```

注意这里使用 `W/2,H/2,D/2`。偶数尺寸的中心体素索引是 `W/2`；不要在外部额外加减半个体素。

角度 β 下旋转坐标为 `px=vx*cosβ+vy*sinβ`、`py=-vx*sinβ+vy*cosβ`、`pz=vz`。名义源位于局部 `(0,SOD,0)`，探测器平面位于 `y=SOD-SDD`。源世界坐标偏移转入该局部坐标系后：

```text
ds_u = src_x*cosβ + src_y*sinβ
ds_n = -src_x*sinβ + src_y*cosβ
ds_v = src_z
q = SOD + ds_n - py
t = (SDD + ds_n) / q
u = (ds_u + t*(px-ds_u)) / pixel_size + det_u0
v = (ds_v + t*(pz-ds_v)) / pixel_size + det_v0
weight = ((SOD + ds_n) / q)^2
```

反投影对每个角度做双线性插值并累加 `weight * projection(u,v)`。当 `q` 或 `SDD+ds_n` 不大于 `1e-6`，或采样点不在 `[0,U-1) × [0,V-1)` 内时，该角度对该体素贡献为零。

## 自动求导

`backproject` 内部将原扩展的 `backward` 接入 PyTorch 自动求导。以下 9 个输入支持一阶梯度：投影、角度、SDD、SOD、探测器 U/V 中心、源 X/Y/Z 偏移。像素尺寸、体素尺寸、体积形状和 ROI 标量不求导；不支持二阶导数。

例如优化源的世界坐标 X 偏移：

```python
src_x = torch.zeros(A, device=device, requires_grad=True)
volume = backproject(
    projections, angles, sid, sod,
    det_u0, det_v0, src_x, zero, zero,
    pixel_size, volume_shape, voxel_size,
)
loss = volume.square().mean()
loss.backward()
print(src_x.grad)
```

`detector_centers` 会保留 `delta_u / delta_v` 的梯度，所以也可以直接优化物理探测器偏移。实际优化应选择与你的重建或标定目标一致的损失。

双线性采样及有效视野边界是分段可微的，边界附近不适合用普通有限差分判断梯度。反向归约采用 `atomicAdd`，浮点舍入可能随线程执行顺序变化，不能保证逐位一致。

原生扩展也保留以下低层 API：

```python
import torch
import src_cbct_backprojector_cuda as native

volume = native.forward(
    projections, angles, sid, sod, det_u0, det_v0, src_x, zero, zero,
    pixel_size, *volume_shape, voxel_size, 0.0, 0.0, 0.0,
)
grads = native.backward(
    torch.ones_like(volume), projections, angles, sid, sod,
    det_u0, det_v0, src_x, zero, zero,
    pixel_size, voxel_size, 0.0, 0.0, 0.0,
)
# grads 顺序：proj, angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z
```

直接调用 `native.forward` 不会自动注册反向传播；需要 PyTorch 自动求导时调用 `backproject`。

## FDK 与自己的投影数据

`fdk_filter` 提供项目主线的余弦预加权、U 方向离散 Ram-Lak 滤波、Hann 窗（默认截止系数 `alpha=0.8`）以及 `pixel_size * SDD/SOD` 缩放。`reconstruct_fdk` 按角度分块滤波、反投影，最后乘 `π/A` 并截断负值；该函数作为重建辅助接口在 `no_grad` 下运行。

这一 FDK 辅助实现沿用当前项目约定：均匀、完整圆轨道扫描；预权重采用名义居中的探测器坐标。源/探测器偏移进入反投影，但滤波预权重未针对所有偏移重新推导。因此它用于复现项目中的近似 FDK 初始化；短扫描、非均匀角度或明显几何扰动需要自行选择角度积分权重、Parker 权重及相应预处理。

投影应为已完成暗场/平场校正及负对数处理的线积分数据。算子不负责这些预处理。可将 `[A,U,V]` 的 float32 Tensor 保存为 `.pt`，再使用示例 CLI（完整选项见 `--help`）：

```bash
python -m examples.fdk --help
python -m examples.fdk --input projections.pt --output volume.pt
```

这条命令使用示例的默认圆轨道几何；接入真实数据时通过 CLI 指定扫描参数，或在 Python 中传入逐角度的几何 Tensor。CSV 对应关系为 `SAD_Rs → sod`、`DD → sid`、`delta_U/V → detector_centers`、`delta_src_x/y/z → src_x/y/z`、`delta_Beta(rad) → angles`；`ODD = SDD-SOD` 不直接传给 `sid`。

FDK 各角度块的反投影输出可线性相加。降低 `angle_chunk_size` 能减少滤波 FFT 的峰值显存；若通过 `backproject` 分块后统一反向传播，求导图仍会保留各块所需投影数据。

## 常见问题

- **找不到 `nvcc` / `CUDA_HOME`**：安装完整 CUDA Toolkit，并指向实际 Toolkit 路径。
- **CUDA 版本不一致**：比较 `torch.version.cuda` 与 `nvcc --version`，在同一环境中使用匹配的 PyTorch 和 Toolkit。
- **编译进程被系统终止**：减小 `MAX_JOBS`，例如设为 `1`；PyTorch C++ 头文件编译需要较多内存。
- **找不到扩展或出现未定义符号**：确认安装、编译和运行使用同一个 Python 环境；更换 Python/PyTorch 后重新编译。不要复用原项目的历史 `.so`。
- **`no kernel image is available`**：用部署 GPU 对应的 `TORCH_CUDA_ARCH_LIST` 重新编译。
- **布局或中心不匹配**：检查 `[A,U,V]`、毫米到像素偏移的负号，以及体素坐标 `index-size/2` 的约定。

打包副本的原生包装会选择输入 GPU 并使用其当前 PyTorch CUDA stream，支持在正确同步的自定义流中调用；跨流数据依赖仍由调用者按 PyTorch 规则同步。

## GitHub 发布

可以把这个目录直接作为仓库根目录上传，或上传本次生成的源码压缩包解压后的目录。`.gitignore` 已排除本机二进制、编译缓存、数据与结果文件。代码来源未附独立许可证，本次未擅自指定授权条款；仓库所有者可按自己的授权意愿添加 `LICENSE`。

重新生成源码 ZIP、tar.gz 和压缩包校验和：

```bash
python tools/package_source.py
```

压缩包默认输出到仓库目录的上一级，只包含源码、说明和示例。修改源码后，可同步更新 `docs/source_manifest.json` 的版本记录再打包。
