# Sony 相机视觉检测模型 —— 交接文档

> 目标：从 Sony 相机固件中提取视觉检测模型，并在主机侧正确推理 + 视觉核对
> 来源工程：`C:\Users\HeMu\Desktop\a7m4-fl23-works`
> 本目录：`C:\Users\HeMu\Desktop\Sony_Camera_Detection_Models`

> 暂停快照（2026-11-22）：以[根目录 README](README.md)为当前状态，以下旧交接保留为历史，不代表今日验收结论。`95c9834` 是 A7R V / A1 II 阶段性交付。
>
> A7 IV 恢复点：[BED 首次图像计算](docs/A7IV-SA-FORWARD.md)仍停在 `0x000a`／`0x0009`。[对照探针](tools/probe_sa_shared_image.py)是未完成的工作稿：AED 首字段为 `4` 时被原校验拒绝，显式改为 `3` 后到达原工作区 DMA；描述符 `0x930000` 表示 148 字节，探针仍只接受 BED 的 152 字节，因此会报错。AED2 不能由该次失败试跑推断。恢复时先逐机核对工作区尺寸和各原控制器的输入，再比较共享程序；不把搬运回放当作算术或真实检测输出。

---

## 0. 一句话状态

**固件与模型已完整复制；格式已完全逆向（NBG/VPMN，12 张表 + 量化参数 + 张量形状全部解出）；但主机侧"正确推理"尚未跑通——权重是 VeriSilicon VIP9000 的 NPU 指令流（`ppu instr` / `nld_istr`），不是标准卷积权重。**

---

## 1. 目录结构（4.1 GB）

```
Sony_Camera_Detection_Models/
├── HANDOVER.md                    本文件
├── firmware/                      原始固件 + 解密产物
│   ├── CamFw/
│   │   ├── 1M2_v400_BODYDATA.DAT     1,194,332,648 B  (ILCE-1M2)
│   │   ├── 7M4_v602_BODYDATA.DAT       757,480,856 B  (ILCE-7M4)
│   │   └── 7RM5_v400_BODYDATA.DAT      754,525,608 B  (ILCE-7RM5)
│   └── 7M4_v602/                  7M4 解密产物
│       ├── firmware.fdat               754,521,088 B
│       ├── firmware.tar                753,267,200 B
│       ├── updater.img                   1,253,376 B
│       ├── config.yaml                         417 B
│       └── nflasha3_fat/              /system 分区
│           ├── av-cam.bin               12,729,680 B
│           ├── bonobo.bin                  172,144 B
│           ├── idt_cam.bin                 167,312 B
│           └── wole_app.bin                 38,784 B
├── extracted/                      从各机型提取的模型
│   ├── 7m4/
│   │   ├── sabin/                     38 个 AF 字典
│   │   └── cp/cpapp-b.bin          16,236,816 B  CP 固件
│   ├── r5/
│   │   ├── sabin/                     42 个 AF 字典（含 scd_dic1/2/3）
│   │   ├── ai_dict/                   10 个（default.nab + 9 .rsc）
│   │   └── cp/cpapp-b.bin          17,617,008 B  CP 固件
│   └── 1m2/
│       ├── sabin/                     41 个 AF 字典
│       ├── ai_dict/                   11 个（default.nab + 10 .rsc）
│       └── cp/cpapp-b.bin          19,190,024 B  CP 固件
├── nab_graphs/                    15 个独立 NBG 图（31.4 MB）
├── parsers/                       7 个可跑 Python 脚本
├── tools/
│   ├── nbinfo.elf                    187,008 B  VeriSilicon NBG 分析器
│   └── fwtool_a7iv.zip                42,588 B  Sony 固件解密工具
├── docs/                          7 份分析报告
├── assets_aux/                     5 张验证图（人/猫/鸟/飞机/空景）
├── aed_layers.json               AED 42 层完整架构
├── nbg_header.json                15 图头 + 首层 8 段
├── nbg_layers.json               15 图 1,320 层类型
├── rsc_configs.json               19 个模型配置
├── dic_layout.json                fdn 4 段头 + aed2 s2f
└── model_manifest.json            45 字典 + 21 场景 + SHA-256
```

---

## 2. 数据来源链

```
BODYDATA.DAT
  → U-FU 容器（魔数 \x89UFU\r\n\x1a\n）
  → DATV / PROV / UDID / FDAT / DEND
  → FDAT 用 crypter CXD90057_k8 做 AES-CBC 解密
  → firmware.tar
  → 0700_part_image/dev/nflasha15（ext2 /usr）
  → 0700_part_image/dev/nflasha8（FAT，含 cpapp-b.bin）
  → 0700_part_image/dev/nflasha3（FAT /system，含 adam_app.bin）
```

工具：`tools/fwtool_a7iv.zip`（Sony 固件解密工具，来自 `repos/sony-a7m4-custom-firmware`）

---

## 3. 模型位置

### 3.1 AF 用字典：`/usr/sabin/*.bin`

| 文件 | 大小 | 用途 |
|---|---:|---|
| `fdn_front_dict.bin` | 103,316 | 人脸正脸 |
| `fdn_side_dict.bin` | 103,316 | 人脸侧脸 |
| `aed_dic1/2.bin` | 460,032 / 74,800 | 动物眼 |
| `aed2_dic1/2.bin` | 460,032 / 568,000 | 动物眼（第二套） |
| `aed2_dic{a,b}_s2f.bin` | 580 | 尺度转 float 表 |
| `bed_dic1/2.bin` | 460,032 / 78,828 | 鸟眼 |
| `pdn_dic.bin` | 72,988 | 人体姿态 |
| `udrt_dic1/2.bin` | 53,668 | — |
| `scd_dic1/2/3.bin` | 450,000 / 21,347 | 场景（仅 7RM5/1M2） |
| `dcv_psf_table_*.bin` | 10,728–63,608 | 虚化 PSF 表 |
| `spica_cetus_*.bin` | 42,504–51,000 | 星芒表 |
| `sa_sabin_*.bin` | 53,504–1,909,504 | 参数块 |
| `star_kernel_param.bin` | 19,338 | 星芒参数 |
| `sa_zoom_nr.bin` | 104,448 | 变焦 |

### 3.2 主体识别模型：`/usr/data/ai_dict/`

| 文件 | 大小 | 内容 |
|---|---:|---|
| `default.nab` | 31,406,992 | **15 个 NBG 子网络** |
| `AAAAAAI.rsc` 等 | 371–395 | 9–10 个 JSON 模型配置 |

---

## 4. 主体类型 ↔ 模型配置

### `ai_model_category` 与 `DETECTION_SUBJECT` 对齐

| category | 主体 | 7RM5 slots | 1M2 slots |
|---:|---|---|---|
| 0 | Human | `{}` / 5,6 / 14,15 | `{}` / 5,6 / 14,15 |
| 1 | Animal | 12, 13 | 18, 19 |
| 2 | Bird | 7, 8 | 7, 8 |
| 3 | Insect | 7, 8 | 7, 8 |
| 4 | Car | 7, 8 | 7, 8 |
| 5 | Train | — | 11, 17 |
| 6 | Plane | 9 | 9 |
| 7 | ? | 10 | 10, 16 |

`slots` 指向 `default.nab` 的 15 个子网络（0–14）。

---

## 5. NBG 格式（`default.nab` 的 15 个子网络）

### 5.1 身份

- `VPMN` 魔数 → **VeriSilicon Vivante VIP9000** 的 **NBG（Network Binary Graph）**；
- `adam_app.bin` 里字符串 `viplite_dump_run_%d.log`、`command`、`states`、`pool`、`verify` 证实是 **VIPLite** 驱动；
- `nbinfo` 是 VeriSilicon 的 NBG 分析器（X86-64 Linux ELF，187 KB，**带完整调试符号 181 个**）。

### 5.2 15 个子网络

| # | 偏移 | 大小 | 名字 |
|---|---|---|---|
| 0 | 0x490 | 0x1324a0 | `sony_net_opt_NHWC` |
| 1 | 0x132930 | 0x1f860 | `reid_NHWC` |
| 2 | 0x152190 | 0x61f20 | `reidpost_NHWC` |
| 3 | 0x1b40b0 | 0x23d420 | `tfl80_t49_e70_q49_c0_NHWC` |
| 4 | 0x3f14d0 | 0x26d9e0 | `model_NCHW` |
| 5 | 0x65eeb0 | 0x2749a0 | `model_NCHW` |
| 6 | 0x8d3850 | 0x1b9a60 | `aed_03_00_05_rc2_NHWC` |
| 7 | 0xa8d2b0 | 0x3cb4a0 | `dummy_network_name` |
| 8 | 0xe58750 | 0x3f8d20 | `dummy_network_name` |
| 9 | 0x1251470 | 0x3d7660 | `dummy_network_name` |
| 10 | 0x1628ad0 | 0x3e05e0 | `dummy_network_name` |
| 11 | 0x1a090b0 | 0x13e1a0 | `model.ckpt-1590000_416x288_NHWC` |
| 12 | 0x1b47250 | 0x13c020 | `model.ckpt-1590000_416x288_r_NHWC` |
| 13 | 0x1c83270 | 0x20320 | `tflite_djd5-U-resnet7-32_64-esch_lw_quan_max23_NHWC` |
| 14 | 0x1ca3590 | 0x150600 | `sfr_tf_resnet_454-80-160-320-512-nose_NHWC` |

已拆成独立文件：`nab_graphs/*.nb`。

### 5.3 NBG 文件头（`06_aed.nb`）

```
0x000  "VPMN"
0x004  0x0001000f          版本
0x008  0xd4
0x00C  "aed_03_00_05_rc2_NHWC"  (64 B)
0x04C  46                  层计数
0x050  91                  算子计数
0x054  1                   输入计数
0x058  4                   输出计数
0x05C  2
0x060  1                   核数
0x064  1
0x068–0x15C  保留（248 B）
0x160  64                  内存池对齐
0x164  0
0x168  0
0x16C  1048320             AXI SRAM
0x170  0
0x174  2097152             内存池
0x178  512                 VIP SRAM
0x17C  层 0 的 8 组 (size, offset)
```

**文件头 = `0x0`–`0x17C`（380 B）。**

### 5.4 层 0 的 8 组 `(size, offset)`

| 段 | size | offset |
|---|---:|---:|
| nn table | 116 | 628 |
| tp table | 464 | 1092 |
| ICD | 3496 | 4588 |
| coef header | 2184 | 50792 |
| coef data | 1936 | 52736 |
| ppu instr | 1756224 | 7924 |
| load states | 9088 | 17012 |
| dyna io | 8320 | 25332 |

`0x1EF4 + 0x1ACC40 = 0x1AEB34 = 1,762,100`，文件 `1,808,992`，差 `46,892` B。

### 5.5 层表（`0x444`，42 条 × 76 B）

每条 = **64 B 类型串 + 12 B 参数 `(id, kernel, stride)`**。

类型分布：
```
ConvolutionReluPoolingLayer2 × 32
PoolingLayer2               × 3
FullyConnectedReluLayer      × 6
ActivationLayer              × 1
```

完整 42 层表见 `aed_layers.json`。

### 5.6 张量形状与量化

| 张量 | 形状 | 量化 |
|---|---|---|
| `input[0]` | **128 × 128 × 3** | — |
| `uid_93_out_0` | **32 × 32 × 3** | scale 1/256, zp -127 |
| `uid_95_out_0` | **1 × 1 × 1** | scale 1/256, zp 0 |
| `output[2]` | **1 × 1 × 1** | scale 1/256, zp 0 |

**注意：输入是 128×128×3，不是 32×32×3。**

### 5.7 算子码表（`nbg_get_operation_name`）

| op | 名字 | 推测含义 |
|---:|---|---|
| 1 | SH | Setup / Header |
| 2 | NN | Neural Network |
| 3 | TP | Tensor Patch |
| 4 | SW | Software |
| 5 | SC | Scalar |
| 6 | SP | Special |
| 7 | NN_LUT | NN Lookup Table |
| 8 | NT | Tensor |
| 0xFFFE | Init | 初始化 |
| 0xFFFF | End | 结束 |

### 5.8 NBG 结构体 12 张表 + 12 计数（`nbinfo` 反编译交叉验证）

| 偏移 | 内容 | 记录大小 | 计数偏移 |
|---:|---|---:|---:|
| `0x1A0` | 输入表 | 112 B | `0x258` |
| `0x1B0` | 层表 | 76 B | `0x260` |
| `0x1B8` | 操作表 | 36 B | `0x264` |
| `0x1C0` | coef header 表 | 8 B | `0x268` |
| `0x1C8` | coef data 表 | 8 B | `0x26C` |
| `0x1D0` | ppu instr 表 | 8 B | `0x274` |
| `0x1D8` | load states 表 | 8 B | `0x278` |
| `0x1E0` | dyna io 表 | 8 B | `0x27C` |
| `0x200` | 补丁表 | 48 B | `0x288` |
| `0x208` | 层参数表 | 88 B | `0x28C` |
| `0x230` | nld_op 表 | 24 B | `0x2B0` |
| `0x238` | nld_istrtable | 8 B | `0x2B4` |
| `0x2A0` | NN_LUT 标志 | — | — |

---

## 6. 核心阻塞

**权重是 NPU 指令流，不是标准卷积权重。**

| 证据 | 数值 |
|---|---|
| AED `ppu instr` | **1,756,224 B** |
| AED `coef data` | **1,936 B**（索引表） |
| `nn table` 量化尺度 | 1/256 |
| `ppu instr` 首字节 | `0a 03 40 e0 42 10 08 02`（opcode + 操作数） |

- `coef data` 只占 `ppu instr` 的 **0.1%–1%**，装不下全部权重；
- `ppu instr` 是 **NPU 微码 + 内嵌权重**，按 tile / 量化格式编码；
- 要还原成标准卷积权重，需要 **NBG 解码器**（VIPLite 驱动或 `nbinfo` 的解析逻辑）。

### `nbinfo` 的真实作用

**它是静态分析器（parser + printer），不是执行器。**

| 选项 | 输出 |
|---|---|
| `-b` | 网络名 / 版本 / target / 核数 / SRAM / 内存池 / 层数 / 算子数 / 输入输出数 / 校验和 |
| `-n` | NBG 头 |
| `-l` | 每层的 `nn/tp/ICD/coeff header/coeff data/ppu instr/load states/dyna io` 的 `(offset, size)` |
| `-o` | 每个算子的 `type / op id / patch 数 / command size / instr size / coeff data size` |
| `-in` / `-out` | 输入输出张量的 dims / format / quant scale / zeropoint / 名字 |
| `-m` | 内存占用统计 |

**它不跑推理。** 真正跑推理的是 `vpm_run`，需要 **VIPLite 驱动 + 真实 VIP9000 NPU 硬件**——这违反"不允许使用机身"。

**所以装 WSL 跑 `nbinfo` 不会带来新信息**：它打印的段偏移 / 大小 / 形状 / 量化参数，已通过反编译 `nbinfo.elf` 全部解出。

---

## 7. 已排除的尝试（全部有实测结果）

| 尝试 | 结果 |
|---|---|
| 模板匹配（`fdn` 段当 12×24 模板） | NCC 峰值 0.23，最高点在墙上 |
| 6 层 CNN 反解（12 个精确解） | Human 框落在腿/脚/墙/伞，Empty 也出框 |
| 跨 12 拓扑稳健性 | 只有 1/12 通过"Human 唯一检出" |
| 零/非零判据 | 方法论错误，已弃用 |
| 阈值 0.3 归一化分数 | 只有 `human_face` 行通过；其余 4 模型全检/误检/无检出 |
| 形状推演 `(k,s)` | 128→64→11→2→1，与张量 32×32 不一致 |
| 36 / 48 / 88 B 定长扫描 | 全部 0 候选 |
| `coef header` 当段索引 | 25.5 GB vs 1.8 MB，排除 |
| 上网搜 `MINNA` / `adam_app.bin` / `sabin` / `ConvolutionReluPoolingLayer2` | 无结果 |
| 装 WSL | 本机未安装（`wsl --status` 空），需管理员 + 重启 |

---

## 8. 继续需要的三选一

1. **提供 VIP9000 的 PPU 指令集文档**（或 `adam_app.bin` 里 PPU 解码函数的地址）；
2. **授权继续反编译 `tools/nbinfo.elf`**（141 函数，已解出 12 张表布局 + `nld_istr` 段头，**还剩实际权重提取**）；
3. **接受"主机侧不可推理"的结论**，本目录的格式分析已足够完整。

---

## 9. 已排除的错误做法

- ❌ 把 NPU 指令当权重硬解
- ❌ 用猜的 stride 硬跑
- ❌ 把随机响应当检测结果
- ❌ 用 OpenCV / 其他检测器冒充固件模型
- ❌ 用机身（违反条件限制）

---

## 10. 复现命令

```powershell
# 环境
& 'E:\anaconda3\envs\test311\python.exe'

# 解析 15 个 NBG 图的头
python parsers/nbg_parse.py

# 导出 15 图全层类型
python parsers/nbg_layers.py

# 解析 .nab 图
python parsers/nab_graph.py

# 解析 dic 头
python parsers/dic_parse.py

# 解密 FDAT + 提取分区
python parsers/extract_fdat.py
python parsers/extract_all.py
```

---

## 11. 参考

- VeriSilicon VIP9000：<https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000>
- `nbinfo` / `vpm_run`：<https://docs.radxa.com/en/cubie/a7s/app-dev/npu-dev/cubie-nbinfo>
- TIM-VX：<https://github.com/VeriSilicon/TIM-VX>
- acuity-models：<https://github.com/VeriSilicon/acuity-models>
- Sony 固件解密工具：<https://github.com/Erlla/sony-a7m4-custom-firmware>
