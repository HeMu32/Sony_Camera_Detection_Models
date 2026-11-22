# Sony 原始模型的主机侧推理实测

2026-11-18。未使用相机机身、VIP9000 硬件或相机直通。这是阶段结果，不是三机型全部检测器的验收结论。

## 已验证与待验证

| 源固件／槽 | 输入 | 当前结果 |
|---|---|---|
| A7R V 和 A1 II 槽 1 `sony_net_opt_NHWC` | 人像和天际线 | 两机固件各自提取的原始图分别执行人像推理，两份输出逐字节一致。人体关键点聚类生成 6 框，目视分别对应 6 人；A7R V 同流程天际线 0 框。框由主机后处理生成，不宣称复制机身的框算法。 |
| A7R V／A1 II 槽 8（同字节） | 鸟和天际线 | 实验解码在鸟图分别框出两只鸟；目视确认两个框各自落在主体上，但有漏检，边界不完全贴合。天际线 0 框。先验、分数阈值和 NMS 不是固件确认值。 |
| A7R V 槽 12、A1 II 槽 18 | 猫和天际线 | A7R V：基于热图与局部成对向量场，主机框覆盖猫头、躯干和前爪；镜像与平移仍定位到猫，边界变化。A1 II：独立姿态热图构框，目视只覆盖头部主体、胸和前爪，切掉耳尖、后躯及尾巴，不能称全身框。两机天际线均 0 框。 |
| A7R V 和 A1 II 槽 **10** | 飞机与天际线 | 两机槽 10 源字节一致，分别跑飞机图得到相同的 21 个张量。实验分类峰 `0.9999999`，天际线同一输出 `0.000072` 并且 0 框。飞机框覆盖机身与尾翼但略切机鼻。槽 9、11 没有同等证据。 |
| A1 II 槽 8 + 10 | 鸟与飞机合成同帧 | 同一输入分别执行两个图，按类别合并后有鸟、飞机各一个实验框，已目视核对。这是主机并行流程，不证明机身调度策略。 |
| A7 IV FDN 正脸／侧脸字典 | 离线格式与固件调用链 | 两份字典的字节布局已完整分段。`debug/deepseek_a7iv/REPORT.md` 记录 CP 阈值比较路径与实测负结果；准确的前向、数值阈值和脸框仍未验收。未用另两机的 NBG 冒充 A7 IV。 |

原始输出在 `debug/cpu_raw/`，输入字节和几何记录在 `debug/cpu_inputs/`。视觉核对图在 `debug/visual_checks/`；带 `peaks` 的图是激活诊断，不是检测结果。人体正式的本阶段验证产物是 `human_pose_boxes.{png,json}` 和 `empty_pose_boxes.{png,json}`。COCO Keypoints 兼容性另见 `docs/COCO-KEYPOINTS-ASSESSMENT.md`；`human_pose_coco.json` 是形状可验证的实验适配结果，不是 COCO 官方解码器输出。

2026-11-19 重新目视检查：[人](../debug/visual_checks/human_pose_boxes.png)、
[鸟](../debug/visual_checks/gpu_slot08_bird.png)、
[飞机](../debug/visual_checks/gpu_slot10_plane.png)、
[A7R V 猫](../debug/visual_checks/r5_cat_pose_boxes.png)、
[A1 II 猫](../debug/deepseek_pose_dispatch/vis/a1ii_s18_cat_box_002.png)、
[天际线对照](../debug/visual_checks/gpu_both_empty.png)和
[鸟与飞机同帧](../debug/visual_checks/gpu_both_mixed.png)。
A1 II 猫链接是筛选姿态关键点后生成的框；同目录不带 `_002` 的图只圈到
前肢局部，是早期探针。单图目视检查不等于精度或召回率评估。

**逐字节一致的范围**：指列出的输入上，CUDA 输出与固定版本 x86 SDK 模拟器的输出一致；不代表任意输入都相同，也不代表模拟器与相机芯片一致。没有采集实机张量，没有与目标芯片及固件对应的独立黄金向量。视觉核对能验证框是否落在物体上，不能验证硬件舍入或机内后处理。

两机槽 1 的 SHA-256 为 `a5d5ead9478e3b069598483bda95702283d7b0c3d825de02c67250eeea4394b6`。人像输出 0 和 1 的 SHA-256 分别为 `35d4c884a62b3d96e1f006464d4ef543e375cacd01d7222872f86d7cd67d49f7` 与 `72315942375fe0259371d5c2828b8a62871bce92d570f22448d6da3323b4ba12`。

## 为什么有点，也有框

量化张量按实际 `scale × (raw − zero_point)` 解码。样例的张量尺寸使用 OpenVX 顺序，第一维连续。横纵坐标先投回模型输入栅格，再撤销输入缩放与填充，得到原图像素坐标。

- **人体槽 1**：输入 `512×336×3×1`；输出 `64×42×40×1` 和 `64×42×19×1`。后者是实测能定位人体关键点的热图；前者为伴随的偏移／回归张量，完整通道定义未复原。`render_pose_boxes.py` 将热图躯干中心、头和肢体响应关联为人体，再由几何范围生成 `xyxy` 框。
- **鸟／飞机槽 8、10**：输入 `320×320×3×1`，输出是多分辨率的 21 个张量。鸟的主响应测试层为 `40×40`，每位置是 `6×5` 分类布局、`6×4` 回归布局和六路辅助值。飞机测试层为 `20×20`，分类布局为 `6×3`，回归布局为 `6×4`。实验解码以分类索引 1 筛选，平移探针支持前两个回归量为中心偏移；宽高和 NMS 仍采用实验假设，未把辅助值当作已知语义。验证图为 `bird_anchor_probe.png`、`plane_anchor_probe.png`、`empty_anchor_probe.png`、`mixed_bird_plane_probe_02.png`；对应 JSON 保留原始 delta 和 `xyxy`。
- **动物槽**：A7R V 槽 12 有 40 和 19 通道、`52×36` 的输出；A1 II 槽 18 有 12 个 `52×36` 的输出，通道数依次为 `18,1,18,36,2,36,1,2,2,2,2,1`。当前图只标绘热图候选，不假定每组都是概率或框坐标。

## CPU 运行路径

1. `parsers/nab_extract.py` 根据重复目录与实际模型字节范围提取两个 NAB。A7R V 15 图；A1 II 17 图且无槽 12、13。所有模型保留原始字节，输出目录的 `manifest.json` 记录 SHA-256。`.rsc` 的槽值不是零基扫描索引。
2. 使用 VeriSilicon `TIM-VX` 提交 `702715d714277376d7ff466b7b72833ff57564c1` 的 `prebuilt-sdk/x86_64_linux`。SDK `VERSION` 为 `6.4.15_CL690884A_D690855_A690484_R690194_T690259_O688896`。模拟目标是 `VIP9000NANODI_PID0XD4`。
3. `vsim_nbg_mode.c` 在模拟器初始化后启用已有的 NBG 执行路径，避免 NN/TP 尝试读取不存在的着色器调试对象。官方 FP32 NBG 对照的启用前后输出严格相同，SHA-256 `3da0956626aafee384dbba210eb6f1c705c96a9702726e4f3c80a2e5e84c8805`。
4. `vsim_legacy_sram_reloc.c` 只调整运行时副本中旧版、来源为 VIP SRAM 的 64 字节单位地址重定位。例：`0x7fc00` 字节边界在加 SRAM 基址后才编码，不能把它当已右移地址。模型、系数和磁盘上的图未修改。

Windows 上用 `tools/prepare_nbg_inputs.py assets_aux debug/cpu_inputs/512x336_rgb 512 336` 生成 RGB 平面张量。Ubuntu 22.04 x86-64 VM 内，装好 `gcc`、`libc6-dev`，把 `tools/` 中运行器和两个兼容层、SDK、原图和输入复制过去：

```bash
mkdir -p outputs
bash run_sony_nbg_cpu.sh x86_64_linux build-cpu r5_slots/slot01.nb \
  outputs/Human cpu_inputs/512x336_rgb/Human_DSC03106.raw > outputs/Human.log
```

同一 Python 环境中，用 `render_pose_boxes.py` 输入上述输出前缀、运行日志、输入 manifest、原 JPG 和输出 PNG。输出 JSON 同时保留原图框坐标和参与构框的关键点。模拟器速度约为分钟级，不应把短超时当成失败。

## 补充固件提取与 A7 IV

从先前由原固件提取的 `nflasha3` FAT 分区，用 7-Zip 单独取得 A7R V 和 A1 II 的 `adam_app.bin` 与 `av-cam.bin`，在 `extracted/{r5,1m2}/adam/`。A7R V 分区 SHA-256 `7f5a573ce2d491035238f3a676562b49ae51eb6f21e4f6e92bacf4793f7bdce4`，A1 II 为 `8db1cca4650fdcc98f18d0b5c8d754c1e8d736d62175c547ed467e143d38abe1`。Windows 环境没有 unblob；两处均是已识别 FAT 成员，未执行提取出的二进制。

`parsers/fdn_sections.py` 可走到两份 `103316` 字节字典末尾：前部 11 个带头的 i16 系数块和 7 个 `[144,1]` 紧凑块，经 12 个值为 2 的 u32 后，还有 6 个系数块和 1 个 `[144,1]` 块。对三机各两份文件实测余数皆为零。带头块的候选卷积参数来自形状和量化布局证据，不是已确认的运行拓扑。A7 IV 已解密 tar 中也未找到 `VPMN` 或 `TFL3` 魔数，个别压缩子模块仍可能需要继续提取。

## GPU 等价对照

**当前状态：槽 8 已有独立 GPU 前向。** 六个尺度的 18 个生成张量与三个不写入的接口槽全部在鸟图、天际线和合成图上与 CPU 逐字节一致。图解、包、运行命令和 `22.0 ms/张` 实测范围见 [槽 8 GPU 推理](SLOT08-CUDA.md)。下面保留 `40×40` 的阶段证据。

本机 RTX 5060 Ti，PyTorch `2.8.0+cu129`。`tools/trace_nbg_coefficients.c` 是仅用于上述固定 x86 模拟 SDK 的可选跟踪器；在 V8 解压写系数和内积前后记录原值，不改变模型。初版跟踪漏传模拟库布尔返回，出现 SRAM 报错；这是跟踪 ABI 错误，不是原始图失败。修正后，同一鸟图带跟踪／无跟踪的 21 个输出全部逐字节一致。

- `verify_nbg_cuda_tiles.py` 在 GPU 上以记录的有符号操作数重算 120,384 个卷积子块累加增量，误差全为 0。
- `verify_nbg_cuda_first_conv.py` 解出原始槽 8 首卷积 `32×3×3×3` 的有符号整数系数。直接读取原始平面 RGB 张量，减输入零点 128，进行步长 1、填充 1 的 GPU 卷积。鸟和天际线各自抽查四块区域的 19,200 个完整三通道累加值，两图都与 CPU 精确一致。
- 同一脚本加 `--layer-trace` 后，从模拟器回读原图的 32 个定点偏置和 32 个 FP32 重定标因子。在 GPU 执行卷积、偏置、ReLU 和 `uint8` 重定标，产出完整 `32×320×320` 中间张量。鸟和天际线各自对照 65,536 个模拟器写出值，差异均为 0。测试报告 `first-layer-{bird,empty}-cuda.json`，中间张量 `first-layer-{bird,empty}.u8`。
- `verify_nbg_cuda_backbone.py` 直接接上该 GPU 张量，继续运行 12 个连续卷积块，用原始量化参数，第二、第四和第八块带 `2×2` 最大池化。最终到达命令 21 的 `128×40×40` 中间张量。鸟图和天际线分别对照 425,984 个额外位置，偏置后实际进入 SDK 整数后处理的有符号低 32 位和最终 `uint8` 写出值均零差异。SDK 的 `PostProcessAccum` 在这条路径调用 ReLU 前对低字执行 `movslq`（`libEmulator.so + 0x141aaf`）；跟踪里留存的高字不参与这项对照。GPU 计算不复用 CPU 的中间激活。
- `verify_nbg_cuda_head40.py` 复现命令 21–25：命令 20 分流到 `21→22/23` 和 `24→25`。命令 22 和 25 使用原始输出零点 149 和 129，不对分数整流；显式卷积展开与 FP32 矩阵乘法避免默认卷积内核的整数误差。两图各有 161,792 个分支中间值与 CPU 对齐（其中命令 21 是上述主干对照重叠项）。命令 23 输出经过 `round(sigmoid(u) × 256)`，按通道连续写入，正式输出 15 的 9,600 字节在鸟图和天际线均逐字节一致。只验证了这两张图覆盖的输入值 `0,1,2`，不推定全部输入或辅助量的用途。
- **更新：分类输出 1 和回归输出 8 已逐字节复现。** `verify_nbg_cuda_tp40.py` 解码原图 TP 命令的零点、尾数和有符号移位：分类 `(149, 128, 53661, 14)`，回归 `(129, 129, 55185, 17)`。四条 10 行条带按 CHW 排列。独立对照 CPU 的 TP 输入、TP 输出和最终原始张量；鸟图和天际线每张的 48,000 分类字节、38,400 回归字节均零差异。
- NN 曾有一处精度分歧：命令 11、`(c,y,x)=(24,127,44)` 的绝对乘积为 `17.499999210238457`，FP32 乘法得 `17.5` 后误舍入到 `18`，CPU 是 `17`。`nbg_cuda_quantization.py` 的单次融合 FP32 内核只在半整数处用 FMA 恢复乘积误差方向，并合并 ReLU、取整和饱和写出。不使用 FP64 推理。RTX 5060 Ti 上 `32×320×320` 单步微基准（预热，五组各 100 次 CUDA Event，取中位数）：新路径 `0.173 ms`，旧多算子的 FP64 路径 `0.549 ms`，约 `3.2×`；不是整图性能。
- 用 GPU 输出 1 和 8 实际运行原来的实验框解码器。`gpu_slot08_bird.png` 与旧 CPU 探针的坐标 JSON 完全相同；目视两个框各选中一只鸟，仍有漏检和局部裁切。`gpu_slot08_empty.png` 目视无框。框解码尚未恢复机内参数。
- 32 命令的额外跟踪 `Bird32.nbc`、`Empty32.nbc` 的 SHA-256 分别为 `84d16ea774838b1f157ac5cdf36b7d0715e625587637891623121738c8cd2817` 和 `b79cc46239706af8a82969d621a9e9c21fa205f34e1d589762b973d1857ba265`。对应 VM 目录 `~/nbg/outputs/coeff_deep_rtuJGy/`；21 个原始输出逐一与各自旧版执行结果 `cmp` 均一致。
- 数据路径：`debug/cpu_raw/coeff_probe_01/Tile.nbc`、`cuda-parity.json` 和 `first-conv-cuda.json`。原始鸟输入和 NBG 未改动；trace SHA-256 `35b4447ece0f27d68528a03ba20db8de48ef3f13da519604de4bee5ae5e536e4`。

```powershell
$d = 'debug/cpu_raw/coeff_probe_01'
& 'E:/anaconda3/envs/test311/python.exe' tools/verify_nbg_cuda_backbone.py "$d/Bird32.nbc" "$d/first-layer-bird.u8" "$d/backbone-bird-cuda.json"
& 'E:/anaconda3/envs/test311/python.exe' tools/verify_nbg_cuda_backbone.py "$d/Empty32.nbc" "$d/first-layer-empty.u8" "$d/backbone-empty-cuda.json"
& 'E:/anaconda3/envs/test311/python.exe' tools/verify_nbg_cuda_head40.py "$d/Bird32.nbc" "$d/first-layer-bird.u8" "$d/NoTrace.15.raw" "$d/head40-bird-cuda.json"
& 'E:/anaconda3/envs/test311/python.exe' tools/verify_nbg_cuda_head40.py "$d/Empty32.nbc" "$d/first-layer-empty.u8" "$d/EmptyLayer.15.raw" "$d/head40-empty-cuda.json"
```

## 固件参数并行核查

已用原 ARM 指令确认两机的 [CP 区域缩放默认值](CP-REGION-SCALE.md)。
它们是 `TuningParamRegionScale` 字段为零时的区域缩放回退值，
不是 NBG 分类阈值。该验证也更正了此前以绝对地址直接索引 CP
文件的读表方法。槽 10 的飞机框仍是主机实测，未绑定到原厂槽选择。

两机的 [CP 置信度门限](CP-CONFIDENCE.md) 已找到具名 getter，并用
原固件代码验证回退值、配置字段、合法范围和精细校正。实际检测的
运行时字段与 NBG 张量到 CP 分数的变换仍未确认。

此外，[原始 XHD 结果转换](CP-XHD-RESULT.md) 已确认 CP 接收的
12 字节记录、Q14 双角坐标、Q15 分数和 50 条上限。它不是现有
实验框解码器读的 NBG 张量，二者之间的原厂转换仍待恢复。

[CP XHD/NMS 命令接口](CP-XHD-NMS.md) 又确认了独立命令
`(0x10, 3, +0x4bc)` 以及 NMS 阈值和开关的字段来源。
原指令回放只验证了参数和描述符，没有执行该命令；
不能把测试夹具里的阈值当作相机配置。

通过 `tools/dsh_web_cli.ps1` 在同工作区创建三个 DeepSeek 会话，分别核查鸟/飞机阈值和锚框、人物/动物及 A1 II 多类型调度，以及 A7 IV FDN。任务摘要在 `debug/deepseek_dispatch/`，各会话报告为 `debug/deepseek_{anchor,pose_dispatch,a7iv}/REPORT.md`。定位到数条阈值更新或比较路径及多条结果记录的结构；固件实际数值阈值和完整调度仍未确认。不能因背景候选通道 0 的原始 logit 在天际线偏高就判飞机误检：本报告的实验框使用槽 10 的 `20×20` 层、每个锚框内对全部 3 类做 softmax 后的索引 1，且天际线零框。不能从 `6×C` 的张量大小独立推出任何两个类别共享同一个图。

A7 IV FAT12 的 `cpImg.dat` 已用 unblob 26.6.4 递归扫描，日志在 `debug/a7iv_parts/cpimg-unblob.ykiMx4/`；格式未识别，但 `CPIR` 的 668 项图像描述符加像素块完整覆盖文件，RGB565 预览为 UI 框，不是检测代码。

A7 IV `nflasha5` 已按 Sony GPL 内核结构验证为 Linux WBI2 快照，里面两处 FDN／PDN 字典名均来自同一个 ext2 目录块。独立复核 CP 的虚表后已把 FDN、PDN、Keypoint 与 Rotate 的结果和参数路径区分开；Keypoint 的 Q16 分数换算不能当作 FDN 阈值。偏移、复现和待查项见 [A7 IV 证据](A7IV-WBI2-DETECTOR-EVIDENCE.md)。

PDN 的后续 DeepSeek 核查以 `danger-full-access` 开到独立同工作区会话 `session-4e6d2ae2-20a6-4cfd-8a18-2214a1e6e9b4`。CLI 最后报告会话停止，但未落盘约定的 `PDN-THRESHOLD.md`；不将该次委派当作已确认的参数。

另一次具同样权限的短任务交付了 `debug/deepseek_a7iv/PDN-INIT-EVIDENCE.md`。修正该报告的 Thumb 立即数扫描后，已独立找到 PDN、FDN 阈值指针的真实构造函数和目标地址。后续又确认三个地址在主核启动时被清零，但检测时实际读到的值仍待追踪。见 [A7 IV 证据](A7IV-WBI2-DETECTOR-EVIDENCE.md)。

飞机槽 10 现在也有 [独立 CUDA 实现](SLOT10-CUDA.md)：三个输入的全部 21 个张量对照 CPU 均逐字节一致，GPU 前向不用 FP64。鸟／飞机混合图现由两个 GPU 模型分别前向。

新增 [舍入 A/B 对照](NBG-ROUNDING-AB.md)：保持前向拓扑和框解码器不变，比较默认模拟器规则与普通 FP32 重量化。两槽各 12 个输入含变换探针，输出张量有差异，但这批样本的实验框未变。

额外用 `tools/verify_nbg_tp_rounding.py` 对分类和框回归的 TP 重量化分开验证。槽 8 正负图跟踪覆盖 1,297/3,072 种不同的参数／输入组合；槽 10 覆盖 1,392/3,584 种，观察值均与 CUDA 一致。独立纯整数实现按推定的 SE8M15 规则做两次中间舍入，然后取整、饱和；枚举 26 套参数各 256 个字节值，共 6,656 种，均与 CUDA 一致。这一遍证明的是该公式在这些配置上的实现，不是未观察组合的模拟器输出，也不是芯片输出。

最新的 `debug/deepseek_a7iv/MINNA-GLOBAL-INIT.md` 针对 `0x0123e528` 报告了更多指针和设置入口候选，仍未用实际写入或初始化源证明数值；“写入者不在该镜像”不能仅凭这一轮搜索下结论。

后续已确认 [A7 IV MINNA 备份加载](A7IV-WBI2-DETECTOR-EVIDENCE.md)
会把 `0x000b0000` 读到阈值块。首段备份没有该记录，发布成员的
第二段却含默认值，已用原 CP 指令验证。另两机第二段也有
[XHD 默认配置](CP-XHD-NMS.md)，其数值、运行时覆盖和照片解码
应分别验证。

后续定位并提取了 A7 IV FDN、PDN 的 SA-DSP 执行载荷。已验证容器索引和 FDN 启动段、第二段入口的反汇编；尚未运行其完整前向。参见 [A7 IV SA 载荷](A7IV-SA-FORWARD.md)。
