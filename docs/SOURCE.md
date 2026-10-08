# 代码来源与打包范围

整理日期：2026-10-08。提取范围根据当前目录内重建与标定程序的实际导入和调用关系确定。

## 当前使用的实现

| 原项目相对路径 | 打包用途 |
| --- | --- |
| `Differenential_forward/src_cbct_backprojector_cuda.cu` | 带源世界坐标偏移及 ROI 平移的反投影、9 组梯度 |
| `Differenential_forward/src_cbct_forwardprojector_cuda.cu` | 与之共享几何的射线积分正投影、9 组梯度 |
| `Differenential_forward/source_only_forward_projector.py` | `project` 和 `DifferentiableSourceOnlyForwardProjector` 自动求导 API |
| `Differenential_forward/selfsupervised_calibration_entropy_2stage.py` | 反投影 `torch.autograd.Function` 的提取依据 |
| `Differenential_forward/selfsupervised_calibration_PBGD_Free_Simulation.py` | 当前主线 `detector_centers`、FDK 滤波与分块重建的提取依据 |

`selfsupervised_calibration_PBGD_Free_Simulation_FDK.py` 直接导入反投影扩展及 `source_only_forward_projector.project`。`rerun_h5_fdk_plateau.py` 复用 `selfsupervised_calibration_PBGD_Free_Simulation.py` 中的重建函数；MicroCT、SuperO 和 WBCBCT 的后台同样使用这一反投影实现。因此本包选择这些源码作为当前使用版本。

原目录没有可读的 Git 提交历史。本次用调用关系、文件时间和 SHA-256 标记提取结果，未把修改时间当作提交版本号。原反投影 `.so` 的文件时间早于其 `.cu`，不能据此断定现有二进制对应最新源码；本包通过独立构建脚本重新编译，不分发历史二进制。

原项目 `Differenential_forward/setup.py` 当时仅构建正投影扩展。本包的 `setup.py` 同时构建两个扩展，不要求先编译其它项目模块。

## 对打包副本的改动

正投影 CUDA 源码按原文件复制，数学内核与原生绑定保持一致。Python 正投影封装保留 `project` 参数顺序、默认值及 `DifferentiableSourceOnlyForwardProjector` 类，增加输入校验和延迟加载扩展；根目录兼容模块保留 `from source_only_forward_projector import project` 的用法。

反投影 CUDA 的两个数学内核保持原样。原生包装补充以下检查与运行处理：投影/梯度的维度、每组几何 `[A]` 形状、同 GPU、尺寸正值、标量有限值、32 位索引上限、输入设备的 `CUDAGuard`、PyTorch 当前 CUDA stream，以及内核启动错误检测。原参数顺序和返回的 9 组梯度顺序保持一致。

Python 反投影包装从实验脚本提取为独立模块，以 `volume_shape=(D,H,W)` 和 `volume_offset=(x,y,z)` 提供便于调用的 API。FDK 辅助函数移除 NumPy、实验配置、日志与个人路径等依赖，保留主线滤波、`π/A` 缩放和非负截断；仅采用 PyTorch 与标准库。

主线 FDK 预权重采用名义居中的探测器坐标，未对所有源/探测器偏移重新推导。本包保留该行为，并在 README 中说明其近似初始化用途。

源码包包含新的小规模合成示例与检查程序，不携带原项目的真实投影、体积、CSV、训练结果、缓存或 `.so`。原工作目录中的文件未被替换。

## 复核方式

`source_manifest.json` 记录原始文件路径、修改时间、SHA-256，以及打包源码的 SHA-256。可用 `sha256sum` 或 Python `hashlib` 与原目录或下载后的源码对照。验证状态另见 [VALIDATION.md](VALIDATION.md)。

原选定文件没有附独立许可证。本次整理不额外授予开源许可；仓库所有者可按自己的授权意愿添加许可证。
