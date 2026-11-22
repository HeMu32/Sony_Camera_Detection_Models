# Sony 7RM5 / 1M2 通用格式导出

## 模型与范围

| 使用场景 | 7RM5 槽 | 1M2 槽 | 模型文件 | 输入 RGB `uint8`（CHW） | 输出数 |
|---|---:|---:|---|---|---:|
| 人体姿态 | 1 | 1 | [ONNX](../runs/portable-release/slot01/sony-slot01.onnx) / [TorchScript](../runs/portable-release/slot01/sony-slot01.pt) | `3×336×512` | 2 |
| 鸟检测 | 8 | 8 | [ONNX](../runs/portable-release/slot08/sony-slot08.onnx) / [TorchScript](../runs/portable-release/slot08/sony-slot08.pt) | `3×320×320` | 21 |
| 飞机检测 | 10 | 10 | [ONNX](../runs/portable-release/slot10/sony-slot10.onnx) / [TorchScript](../runs/portable-release/slot10/sony-slot10.pt) | `3×320×320` | 21 |
| 猫姿态（7RM5） | 12 | — | [ONNX](../runs/portable-release/slot12/sony-slot12.onnx) / [TorchScript](../runs/portable-release/slot12/sony-slot12.pt) | `3×288×416` | 2 |
| 猫姿态（1M2） | — | 18 | [ONNX](../runs/portable-release/slot18/sony-slot18.onnx) / [TorchScript](../runs/portable-release/slot18/sony-slot18.pt) | `3×288×416` | 12 |

原图按 SHA-256 校验。槽 1、8、10 在两台机型中分别逐字节相同，
所以各只生成一份。猫两个槽不同。此交付不含 7M4，也不把阶段交付
未运行的其他检测或细化槽纳入验证结论。

## 运行与复现

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
& $py -B tools/export_sony_portable.py runs/portable-new
& $py -B tools/verify_sony_portable.py runs/portable-new --extra-sdk runs/portable-release/detector-sdk
& $py -B tools/export_sony_slot18.py export runs/portable-new
& $py -B tools/export_sony_slot18.py verify runs/portable-new
& $py -B tools/export_sony_legacy_pose.py export runs/portable-new
& $py -B tools/export_sony_legacy_pose.py verify runs/portable-new
```

上表的模型与同目录的接口 JSON 随仓库提供，位于 `runs/portable-release/`。
可选的 SDK 参考转储和其他中间文件不随仓库提供。运行导出模型只需要
相应的模型文件和 ONNX Runtime 或 PyTorch；不需要代码库、CUDA、CuPy、
原始系数包或 VeriSilicon SDK。导出脚本和验证脚本复用现有 CUDA
工程依赖。ONNX 使用 opset 17 标准算子，不含自定义域和外部模型数据。

槽 8 和 10 都接收单张 `uint8[3,320,320]` RGB，未内置读图、缩放和画框。
样本预处理做 EXIF 方向校正、双线性 letterbox、填充值 128。输出是
按原接口编号排序的 21 个 `uint8` 张量，其中鸟模型有三个未写入的接口，
导出时初始化为零。各输出的形状、存储顺序、量化尺度和零点见
`slot08/sony-slot08.json`、`slot10/sony-slot10.json`。

```python
import numpy as np
import onnxruntime as ort
import torch

rgb = np.fromfile("input-320x320-chw.raw", dtype=np.uint8).reshape(3, 320, 320)
prefix = "runs/portable-release/slot08/sony-slot08"
onnx_result = ort.InferenceSession(prefix + ".onnx", providers=["CPUExecutionProvider"]).run(
    None, {"rgb": rgb}
)
with torch.inference_mode():
    torch_result = torch.jit.load(prefix + ".pt", map_location="cpu")(torch.from_numpy(rgb))
```

导出保留原始整数卷积权重和偏置。卷积累加的整数范围允许 FP32 精确表示；
NN 重量化使用 FP64 做精确乘积，避免先舍入到 FP32 时的半整数误差。
TP 重量化使用由原始零点、尾数和移位生成的 256 项表，不从样本拟合。
这是便于移植和核对的 CPU 参考，不是原 CUDA 路径的性能替代。

## 实测

ONNX Runtime 1.24.4 CPU 和 PyTorch 2.8.0+cu129 CPU 重新加载模型后，
五个模型均使用 `assets_aux` 的五张图片，与固定版 x86 SDK 运行原始
NBG 的输出逐张对照。每个格式共比较 **5,166,750** 个输出字节、290 个
张量，差异均为 0，最大量化码误差为 0。鸟和飞机还对照了独立 CUDA 路径。
五个 ONNX 和五个 TorchScript 都在仓库外独立重载并完成 CPU 推理，
文件哈希和总计见 [验证汇总](../runs/portable-release/verification-summary.json)。
`detector-sdk/` 保存鸟、飞机各自的三张交叉样本原始输出；
不传 `--extra-sdk` 则仅对照此前随上游交付的匹配正样本和天际线 SDK 输出，
但仍在全部五张图上比较 CUDA。

| 输入 | 槽 8 鸟框 | 槽 10 飞机框 |
|---|---:|---:|
| 人物 | 0 | 0 |
| 鸟 | 2 | 0 |
| 飞机 | 0 | 1 |
| 猫 | 0 | 0 |
| 天际线 | 0 | 0 |

两格式的类别、分数及框坐标与同一解码器下的 CUDA 基准一致。ONNX
叠图已目视核对：鸟只检出两只；飞机框覆盖机身与尾翼，略切机鼻；
天际线无框。保留上游实验解码器的限制，这不是 Sony 机内后处理或
召回率验证。原始逐张结果和误差在 `verification.json`。

## 1M2 猫槽 18

输入为 `uint8[3,288,416]` 平面 RGB。两种独立模型各输出 12 个 CHW
量化张量，量化尺度及零点在 `slot18/sony-slot18.json`。原执行的 4 条 TP
把每个输入通道分为 4 个空间相位，并在右边和底边各补 1 个码值 127。
在猫和天际线图上，直接和 SRAM 比较的 **727,320** 个字节完全一致。

原命令的绝对地址指出哪三路特征被拼接、哪些分支写出最终头；这套连线
在猫和天际线图上逐条核对了 69 个 NN 命令，共 **22,917,024** 个中间
字节，差异为 0。对五张样本，每种导出格式另与原始 SDK 的最终输出对照
**1,132,560** 字节，差异为 0。ONNX 猫图生成 1 个与 SDK 相同的实验框；
它没有覆盖耳尖与后躯。天际线为 0 框。

`slot18/slot18-verification.json` 保存逐图结果。验证默认读取
`runs/portable-slot18-stage1/` 下的 SDK 和中间追踪，也可用 `--sdk`
及 `--trace` 指定位置。采集可选用 `NBG_POSE_STEM_SRAM` 取得 TP 后的
SRAM；用 `NBG_NN_WRITES` 与 `NBG_NN_WRITES_OPS=69` 记录 NN 写入。

## 人和猫的 CUDA 对照

CUDA 运行使用原系数重建的 PyTorch eager 图，不是把 CPU 导出的
TorchScript 或 ONNX 直接放到 GPU。卷积用 FP32 并关闭 TF32；
重量化复用 `nbg_cuda_quantization.py`，两种模式均不使用 GPU FP64。
`simulator` 只在 FP32 乘积落在半整数时用 FP32 FMA 残差纠正舍入；
`simple` 不纠正。下表比较五张 `assets_aux` 样图的所有最终输出字节：

| 原模型 | 对照字节 | `simulator` 差异 | `simple` 差异 | `simple` 最大码差 |
|---|---:|---:|---:|---:|
| 两机共享人体槽 1 | 792,960 | 0 | 13,538 | 15 |
| A7R V 猫槽 12 | 552,240 | 0 | 24,603 | 85 |
| A1 II 猫槽 18 | 1,132,560 | 0 | 435,945 | 46 |

人体图的六个实验框各选中一人。A7R V 的猫框覆盖身躯，但切掉耳尖
和尾巴；A1 II 的框还漏掉相当一部分后躯。三个模型对天际线都没有
画出实验框。两种舍入的框数相同，但人体和 A7R V 的猫框坐标不同；
A1 II 这张猫图的框坐标相同。这些框仍来自实验后处理，不是对原厂
JNP 峰提取、关联或检测框发布的验证。

```powershell
& 'E:/anaconda3/envs/test311/python.exe' -B tools/verify_pose_cuda.py runs/pose-cuda-new
```

输出包括每张图每个张量的码差、CUDA 事件耗时、实验框和叠图。
逐字节一致只针对这些样图及固定版 x86 SDK，不代表芯片实测。

## 人体与 7RM5 猫

人体槽 1 有两路 `64×42` 输出，7RM5 猫槽 12 有两路 `52×36` 输出。
每张图依次输出 40 和 19 通道，两机人体合用一套权重与导出文件。
五张样本逐图对照固定版原始 NBG SDK，人体每种格式核对 **792,960**
个输出字节，7RM5 猫核对 **552,240** 个，均为零差异。导出版本画框与
SDK 相同：人物图 6 框、7RM5 猫图 1 框，两路天空线均为 0 框。

深度卷积使用 SDK 记录的通道映射；两级下采样还原为步长 2 的深度卷积。
尾部按原命令读写地址拼接通道，残差融合保留原始无符号系数，反馈分支从
每条命令读取边界填充值。7RM5 猫尾部 48 条 TP 搬运合并为 12 次全图
重量化，结果在下一条 NN 的输出上验证。正样本逐条对照如下：

| 原始图 | 对照 NN 命令 | 逻辑张量输出字节 | 差异 |
|---|---:|---:|---:|
| 人体 | 415 | 53,165,952 | 0 |
| 7RM5 猫 | 218 | 37,026,288 | 0 |

追踪器在末级分块另外记录了索引 128–134 的钩子调用；这 7 路不在
逻辑 128 通道张量中，不计入比较。`slot01/verification.json` 和
`slot12/verification.json` 记录逐图结果，包含这些调用的计数。
验证脚本默认使用 `runs/portable-legacy-fullnn/` 的完整追踪，及
`extra/` 中用于三张交叉样本的 SDK 输出。

## 姿态权重检查点

`pose-parameters/sony-*-weights.pt` 可用 `torch.load(..., weights_only=True)`
加载，是按 NN 命令编号组织的参数字典。`weight_kzyx` 保留模拟器写入
的原始系数值，使用 `uint8` 或 `int16`；它不是已确定语义的训练权重。
偏置保留低高字，量化记录保留原字。重复切片还没有归并为逻辑层。

用固定版 x86 SDK 运行原始图，从正样本提取系数、偏置和后处理配置，
再用空天际线独立提取并逐项对照。下表的系数数包括硬件切片中的重复项。
正负样本的系数和偏置均一致，带追踪运行的 32 个输出张量与未追踪
参考逐字节一致。正样本重画的框是人体 6 个、猫各 1 个，
其中 1M2 的猫框未覆盖耳尖和后躯。天际线均为 0 框。
这些图在 `pose-sdk-overlays/`，是 SDK 基线，不是检查点的推理结果。

| 原始图 | NN 命令及系数块 | TP 命令 | 核对的系数值 |
|---|---:|---:|---:|
| 人体 | 415 | 4 | 2,696,096 |
| 7RM5 猫 | 218 | 52 | 1,941,696 |
| 1M2 猫 | 69 | 4 | 1,824,256 |

导出和验证脚本：

```powershell
& $py -B tools/export_sony_pose_weights.py runs/portable-release/pose-evidence runs/pose-new
& $py -B tools/export_sony_pose_weights.py runs/portable-release/pose-evidence runs/pose-new --verify-empty
```

采集时在 `run_sony_nbg_cpu.sh` 前设置 `NBG_COEFF_TRACE`、`NBG_COEFF_TRACE_OPS=1000`
和 `NBG_COEFF_PARAMS_ONLY=1`。天际线组另设 `NBG_NN_COMMANDS`、`NBG_TP_TRACE`
及 `NBG_TP_COMMANDS_ONLY=1`；用 `gzip -1k` 压缩 `.nbc`。原模型、输入和
SDK 版本沿用主机交付。

参数文件本身不包含可执行计算图；这组正负样本检查验证的是追踪和
系数提取。独立计算图的结果以对应槽的导出验证报告为准。
`pose-evidence/*-empty.nnc` 记录每次 NN 命令的执行序号、SDK 索引和
32 个命令字；`.nbt` 记录 TP 命令及其与 NN 的执行顺序。

两张旧版姿态图的重排输入在右、下边增加的码值为
0。普通逐点卷积可直接运行，深度卷积则需要用 SDK 真实记录的通道映射
重排系数。已分别提取 7RM5 猫图 100 张、共用人体图 213 张命令映射表；
映射不只一种，不能把首层的规律直接套到后面。第一处下采样匹配步长
为 2 的深度卷积，不是最大池化。

| 主干样本 | 输入重排字节 | 前 25 条 NN 命令输出字节 | 差异 |
|---|---:|---:|---:|
| 7RM5 猫图 | 363,660 | 2,582,528 | 0 |
| 共用人体的正样本 | 521,196 | 2,195,456 | 0 |

```powershell
& $py -B tools/verify_legacy_pose_stem.py runs/portable-legacy-stem runs/portable-release/pose-parameters
```

这份额外的首段对照和通道映射在 `runs/portable-legacy-stem/stem-verification.json`。
