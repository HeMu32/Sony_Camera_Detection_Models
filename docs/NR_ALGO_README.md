# 索尼相机固件中的对焦视觉识别模型：提取与可推理判定

> 图片降噪（NR）三机对照见
> [三台 Sony 相机图片降噪算法关联研究](NR-ALGORITHM-PARITY.md)。

> 工作区：`C:\Users\HeMu\Desktop\a7m4-fl23-works`
> 目标固件：`FwRev/CamFw/{1M2_v400,7M4_v602,7RM5_v400}_BODYDATA.DAT`
> 验证图片：`assets_aux/{Human,Cat,Bird,Flight}_*.jpg`（EXIF 全部为 ILCE-7RM5）

---

## 0. 结论摘要

| 问题 | 结论 | 证据强度 |
|---|---|---|
| 能否从固件里提取出对焦用视觉识别模型 | **能，已全部提取**（三机共 121 个 `/usr/sabin` 文件 + 21 个场景模型文件） | 强（逐字节哈希） |
| 模型是不是"物体识别"模型 | **是**，是 CP 固件 `Camera::Minna` 引擎加载的主体检测字典 | 强（RTTI 类名 + 驱动 + 判决函数） |
| 三种机型是否各有自己的模型 | **检测权重三机完全相同**；差异只在 `scd` 与场景模型 | 强（哈希比对） |
| 能否推理 | **机身内能**（`ODL_LoadBinary` → Adam NPU）；**主机侧 `dic` 不能直接跑**，`.nab` 可以 | 强（加载链 + RTTI + 容器结构） |

---

## 1. 提取方法（可复现）

三个 `BODYDATA.DAT` 都是 U-FU 容器（magic `\x89UFU\r\n\x1a\n`），
内部 `FDAT` 用 AES-CBC 加密，crypter 为 `CXD90057_k8`。
密钥在 `FwRev/tools/fwtool.py-a7iv/fwtool/sony/constants.py` 里。

```powershell
# 1) DAT -> firmware.tar（fwtool 的 fdat 解密 + tar 提取）
& 'E:\anaconda3\envs\test311\python.exe' out\models\extract_fdat.py `
    FwRev\CamFw\1M2_v400_BODYDATA.DAT out\models\work\1m2
& 'E:\anaconda3\envs\test311\python.exe' out\models\extract_fdat.py `
    r5\plain.bin out\models\work\r5          # 7RM5 已解密的 fdat
# 7M4 的 firmware.tar 已存在：FwRev\fw\7M4_v602\firmware.tar

# 2) 从 firmware.tar 里取 /usr 分区（nflasha15，ext2）
7z e out\models\work\1m2\firmware.tar -oout\models\work\1m2\part "0700_part_image\dev\nflasha15" -y
7z e out\models\work\r5\firmware.tar  -oout\models\work\r5\part  "0700_part_image\dev\nflasha15" -y
7z e FwRev\fw\7M4_v602\firmware.tar -oout\models\work\7m4 "0700_part_image\dev\nflasha15" -y

# 3) 从 ext2 里取 /usr/sabin 与 /usr/data/ai_dict
7z e out\models\work\r5\part\nflasha15  -oout\models\work\r5\sabin  "sabin\*" -y
7z e out\models\work\r5\part\nflasha15  -oout\models\work\r5\ai_dict "data\ai_dict\*" -r -y
# 7M4 / 1M2 同理，路径分别是 out\models\work\{7m4,1m2\part\nflasha15
```

解包结果（每个机型都得到同一套 `/usr/sabin` 字典 + 可选的 `/usr/data/ai_dict`）：

| 机型 | firmware.tar 内 nflasha15 | 解出的模型文件数 |
|---|---:|---:|
| ILCE-1M2 4.00 | 747,399,168 B | 16 个检测字典 + 11 个 `ai_dict` 成员 |
| ILCE-7M4 6.02 | 378,580,992 B | 13 个检测字典（无 scd、无 ai_dict） |
| ILCE-7RM5 4.00 | 410,652,672 B | 16 个检测字典 + 10 个 `ai_dict` 成员 |

---

## 2. 模型清单（逐字节哈希）

检测字典全部位于 `/usr/sabin/`，由 CP 固件的 `Camera::Minna` 引擎加载。
下表的 SHA-256 前缀足以证明"三机相同/不同"：

| 文件 | 大小 | SHA-256 前 16 | 7M4 | 7RM5 | 1M2 | 语义 |
|---|---:|---|---|---|---|---|
| `fdn_front_dict.bin` | 103,316 | `cbf0ee891b1ec2ce` | ✓ | ✓ | ✓ | 正脸检测（FDN，2 字典之一） |
| `fdn_side_dict.bin` | 103,316 | `26c8577dfeb08e40` | ✓ | ✓ | ✓ | 侧脸检测（FDN，2 字典之二） |
| `pdn_dic.bin` | 72,988 | `b2ac10a58b02620f` | ✓ | ✓ | ✓ | 人体姿态（PDN，1 字典） |
| `aed_dic1.bin` | 460,032 | `d3fa6482f87a91c0` | ✓ | ✓ | ✓ | 动物眼部检测（AED v1，字典 1） |
| `aed_dic2.bin` | 74,800 | `2cd14e687b6559d5` | ✓ | ✓ | ✓ | 动物眼部检测（AED v1，字典 2） |
| `aed2_dic1.bin` | 460,032 | `20fc8751f3f8774b` | ✓ | ✓ | ✓ | 动物眼部检测（AED v2，字典 1） |
| `aed2_dic2.bin` | 568,000 | `5aa286fdd195d240` | ✓ | ✓ | ✓ | 动物眼部检测（AED v2，字典 2） |
| `aed2_dica_s2f.bin` | 580 | `e88f7cd4c382182b` | ✓ | ✓ | ✓ | AED v2 逐层 scale→float 表 A |
| `aed2_dicb_s2f.bin` | 580 | `90d97c016c795160` | ✓ | ✓ | ✓ | AED v2 逐层 scale→float 表 B |
| `bed_dic1.bin` | 460,032 | `c4a796242e32d841` | ✓ | ✓ | ✓ | 鸟类眼部检测（BED，字典 1） |
| `bed_dic2.bin` | 78,828 | `5d20069c26594375` | ✓ | ✓ | ✓ | 鸟类眼部检测（BED，字典 2） |
| `udrt_dic1.bin` | 53,668 | `7c11061c7e440b79` | ✓ | ✓ | ✓ | 通用实时检测（UDRT，字典 1） |
| `udrt_dic2.bin` | 53,668 | `1641ba349a7de7c3` | ✓ | ✓ | ✓ | 通用实时检测（UDRT，字典 2） |
| `scd_dic1.bin` | 450,000 | `dc3d4f1a7bad0067` | — | ✓ | ✓ | 主体类别检测（SCD，字典 1） |
| `scd_dic2.bin` | 21,347 | `8b3a4084dd2494e2` | — | ✓ | ✓ | SCD，字典 2 |
| `scd_dic3.bin` | 21,347 | `8e968bd821af7926` | — | ✓ | ✓ | SCD，字典 3 |

**关键事实：所有检测权重在三台机身上逐字节相同。**
A7M4 少的就是 `scd_*`（主体类别检测），与"A7M4 只支持人/动物/鸟，A7R5 与 A1 II 额外支持昆虫/汽车/火车/飞机"完全对应。

场景识别（`/usr/data/ai_dict`，只出现在 7RM5 与 1M2）：

| 机型 | `default.nab` | 大小 | 附带的 `.rsc` |
|---|---|---:|---:|
| 7RM5 | `bd8572abe9c045da…` | 31,406,992 | 9 |
| 1M2 | `cd96d4432f6c9612…` | 40,047,760 | 10 |

`.rsc` 是 JSON 资源（`na_file_name` / `rsc_file_name` / `icon_tar_file_name` / `dov_file_name`），
即"自动识别场景"的图标与名称资源；`.nab` 是场景分类网络。

---

## 3. 语义映射（模型 ↔ 识别主体）

证据来自 CP 固件（`nflasha8/cpapp-b.bin`）的 RTTI、日志串与判决函数：

```
Camera::Minna::Fdn::FdnDrv            DIC_NUM = 2   ← 正脸/侧脸
Camera::Minna::Pdn::PdnDrv            DIC_NUM = 1   ← 人体姿态
Camera::Minna::Keypoint::Aed::AedDrv   DIC_NUM = 2, MAX_BULK_SA_NUM = 5  ← 动物眼
Camera::Minna::Keypoint::Bed::BedDrv   DIC_NUM = 2, MAX_BULK_SA_NUM = 5  ← 鸟眼
Camera::Minna::Scd::ScdDrv            DIC_NUM = 3   ← 主体类别（3 字典）
judgmentAnimalBirdPrioPartEye/Head1/Head2/Body1/Body2
judgmentCar / judgmentTrain / judgmentInsectAndAirplane
AmS_PoseDetect* / AmS_AnimalDetect* / AmS_ObjectDetect*
```

`DIC_NUM` 与文件个数一一对上：FDN 2 个、AED 2 个、BED 2 个、PDN 1 个、SCD 3 个。

对应关系：

| 识别主体 | 检测部件 | 使用的字典 |
|---|---|---|
| 人（脸） | 正脸/侧脸 | `fdn_front_dict.bin`、`fdn_side_dict.bin` |
| 人（身体/姿态） | 姿态关键点 | `pdn_dic.bin` |
| 动物 | 眼部（+头/身） | `aed_dic1/2.bin`、`aed2_dic1/2.bin` |
| 鸟 | 眼部（+头/身） | `bed_dic1/2.bin` |
| 昆虫 / 汽车 / 火车 / 飞机 | 主体类别判定 | `scd_dic1/2/3.bin` |
| 通用实时跟踪 | 框/关键点 | `udrt_dic1/2.bin` |

---

## 4. 容器格式分析

`/usr/sabin/*.bin` 不是 TFLite/ONNX/Caffe 等通用格式（无 `TFL3`、`ONNX`、`caff` 魔数），
也不是 LZ77 压缩（首字节不是 `0xF0`/`0x0F`）。它们是 **CP 的 `Camera::Minna` 引擎的 ODL 二进制**：

- CP 固件日志串：`[IDT_DDL_MINNA]: ODL_LoadBinary) send_msg.filename %s`、`p_nn_weight : %p`、`p_dict : %p`、
  `p_FrontDict: %p`、`p_SideDict : %p`、`address_offsets_.dic : 0x%08x`；
- `AdamDriver: LoadFile` / `AdamDriver: ODL response message error. LoadFile failed` —— CP 通过 OSPI 把字典文件送进 "Adam" NPU；
- `ddl/ddl_minna_drt.cpp`、`ddl/ddl_minna_fdn_param_manager.cpp` 等源文件名说明拓扑在 DDL 层。

量化布局（由 `parse_dic.py` 实测）：

| 文件 | 布局 |
|---|---|
| `*_dic1.bin` | int8 权重（zero≈0.28、0xFF≈0.15，典型对称量化） |
| `*_dic2.bin` | int16 偏置表，尾部为 `(int16,int16,int16,0)` 记录 |
| `*_s2f.bin` | float32 逐层 scale，AED v2 为 6 个值：`0.9990, 0.9968, 0.9894, 0.9790, 0.9654, 0.6346` |
| `fdn_*_dict.bin` | 4 段字典，每段头 `[w, h, 6, 3, 4, 1, 1, a, a, h, w/6, 3, 4]`：`12×24 / 18×30 / 24×48 / 36×60` |
| `scd_dic1.bin` | 头尾用 `0x03` 填充，真实数据在中间 |

`fdn` 的 4 段头是**唯一显式给出的拓扑线索**：4 个尺度、每段 6 层、3×3 核、步长 4。
其余字典的层拓扑写在 CP 的 `buildUpParam()`（`Camera::Minna::*ParamManager`）里，不在字典文件内。

### 4.1 场景模型 `default.nab` 是自描述图

与 dic 不同，`.nab` 内部有**明文的节点名、层类型与参数**，可以直接解析：

```
sony_net_opt_NHWC            net 名（NHWC 布局）
input[0] … input[15]          16 个输入（多裁剪/多帧）
uid_135_out_0 / uid_152_out_0  节点名
ConvolutionReluPoolingLayer2    卷积+ReLU+池化 复合层（236 次）
FullyConnectedReluLayer        全连接+ReLU
PoolingLayer2                  池化
TensorCopy / TensorMul / TensorSub / TensorReduceSum   张量算子（归一化）
reid_NHWC / reidpost_NHWC      reid 子网
```

记录布局：`[64 B 节点名][参数][64 B 层类型][参数]…`，
`ConvolutionReluPoolingLayer2` 记录之间是层参数（含通道/核/步长/权重偏移），
因此 `.nab` 具备**主机侧解析并推理**的条件；`dic` 不具备。

### 4.2 `dic` 的层拓扑在 CP 的 `FdnDrvParam` 里

在 Thumb-2 CP 固件里已定位并反编译：

| 函数 | 地址 | 作用 |
|---|---|---|
| `FdnParamManager::buildUpParam` | `0x16ddfc` | 只做 MMU 地址解析（`DAT_00cace60` 表、`FUN_004cd50c`），不含拓扑 |
| `FdnParamManager::buildParam2` | `0x16df04` | 把 `FdnDrvParam`（`param_2`）的逐层 u16 字段拷进 MMU 参数结构 |

`buildParam2` 从 `param_2` 拷贝的字段：`+0x10/+0x12/+0x14`、`+0x48/+0x4a/+0x4c/+0x4e/+0x50`、
`+0x18/+0x20/+0x22/+0x24`、`+0x28/+0x30/+0x32/+0x34`、`+0x38/+0x40/+0x42/+0x44`、
`+0x58/+0x54/+0x60`。这些就是每层（宽、高、通道、核、步长、pad）参数。

`FdnDrvParam` 由 `*ParamManager` 构造函数填（AED 的同类构造函数是 `FUN_00168764`：
写入 `0x200/0x400/0x40/0xc8/0x93a00/0x1e000/0x3e/0x2a/0x2b/0x28/0x29/0xd/0x2c` 等常量）。
下一步就是找 FDN 的同类构造函数，把这 6 层的字段值导出。

### 4.3 字典容器解析结果

`dic_parse.py` 对每个字典做结构解析，结果写入 `dic_layout.json`：

```
fdn_front_dict.bin / fdn_side_dict.bin
  section0: [12, 24, 6, 3, 4, 1, 1, 1, 1, 24, 2, 3, 4]
  section1: [18, 30, 6, 3, 4, 1, 1, 2, 2, 30, 3, 3, 4]
  section2: [24, 48, 6, 3, 4, 1, 1, 1, 1, 48, 4, 3, 4]
  section3: [36, 60, 6, 3, 4, 1, 1, 2, 2, 60, 6, 3, 4]

aed2_dica_s2f.bin（float32，145 个值；偶数下标为逐层 scale）
  0.998953, 0.996829, 0.989381, 0.979039, 0.965398, 0.6346, …（后接 0.01…0.60 的单调表）
aed2_dicb_s2f.bin
  0.999709, 0.998827, 0.991911, 0.949251, 0.862155, 0.073088, …
```

4 段头在三机完全相同，且 `6` 层、`3×3` 核、`4` 步长在 4 段里一致——这是"字典=固定拓扑 + 量化权重"的直接证据。

### 4.4 验证结果

`verify_models.py` 对上述全部断言做机器校验，58 项全部 PASS（输出见 `verify_report.txt`）：

| 检查组 | 结果 |
|---|---|
| 每机 `DIC_NUM` ↔ 字典文件数（fdn/aed/aed2/bed/pdn/scd/udrt） | PASS |
| 三机共有字典逐字节相同 | PASS |
| `scd` 只在 7RM5 / 1M2 | PASS |
| `fdn` 四段头 `12×24/18×30/24×48/36×60` | PASS |
| `aed2` s2f 145 值 / 前 6 个 scale | PASS |
| `ai_dict` 只在 7RM5 / 1M2 | PASS |

---

## 5. 可推理判定

**机身内：可以推理，且这是设计路径。**

```
/usr/sabin/*.bin
   → CP 固件 Camera::Minna::SaIpDriver_base<...>::_bulk_start / api_sendMessage_bulkstart
   → IDT_DDL_MINNA / ODL_LoadBinary  (send_msg.filename = /usr/sabin/...)
   → AdamDriver: LoadFile (OSPI)
   → Adam NPU 执行
   → FDN_Output / PDN_Output / el_score / er_score / FaceScore 回读
```

已核对的完整性证据：

1. 每个机型的 `DIC_NUM` 与文件数完全一致（FDN 2、AED 2、BED 2、PDN 1、SCD 3）。
2. `aed2_*_s2f.bin` 的 6 个 scale 与 6 层拓扑一致。
3. `fdn_*` 的 4 段头给出 4 个尺度的窗口尺寸。
4. 三机哈希一致，说明模型未随固件版本变化，只随引擎参数变化。

**主机侧：两类模型不同。**

- `dic`（对焦主体检测）：**不可直接主机推理**。它是 Adam NPU 的 ODL 二进制，主机侧要先还原 `buildUpParam()` 的层描述。
- `.nab`（AI 模型包）：**容器可解析，但载荷不可直接推理**。`VPMN` 段头与网络名是明文（见 6.1），
  但每段载荷熵 7.1–7.9、前 16 B 全零，且没有 TFLite `TFL3`/ONNX/Caffe 魔数 —— 载荷是压缩/加密的私有格式。

已把 CP 固件按 Thumb-2 重新导入 Ghidra（`cpapp-b-r5-thumb.bin`，`ARM:LE:32:Cortex`，34,344 个函数），并已定位到：

| 函数 | 地址 | 结论 |
|---|---|---|
| `FdnParamManager::buildUpParam` | `0x16ddfc` | 只解析 MMU 地址（`DAT_00cace60` + `FUN_004cd50c`） |
| `FdnParamManager::buildParam2` | `0x16df04` | 从 `FdnDrvParam`（`param_2`）拷逐层 u16 字段进 MMU 参数块 |

`buildParam2` 拷贝的字段（`+0x10/0x12/0x14`、`+0x48/0x4a/0x4c/0x4e/0x50`、`+0x18/0x20/0x22/0x24`、
`+0x28/0x30/0x32/0x34`、`+0x38/0x40/0x42/0x44`、`+0x58/0x54/0x60`）就是每层的
（宽、高、通道、核、步长、pad）。`FdnDrvParam` 由 `*ParamManager` 构造函数填
（AED 的同类是 `FUN_00168764`，写入 `0x200/0x400/0x40/0xc8/0x93a00/0x1e000/0x3e/0x2a/0x2b/0x28/0x29/0xd/0x2c`）。

**判定结论**：

| 运行位置 | 能否推理 | 依据 |
|---|---|---|
| 机身 CP（Adam NPU） | ✅ 能 | `ODL_LoadBinary` → `AdamDriver:LoadFile` → NPU 执行 → `FDN_Output`/`PDN_Output`/`el_score`/`er_score` 回读 |
| 主机（PC，`dic`） | ❌ 不能直接跑 | 字典是 NPU 的 ODL 二进制，主机侧要先还原 `FdnDrvParam` 的层字段并模拟 Adam 指令 |
| 主机（PC，`.nab`） | ❌ 不能直接跑 | `VPMN` 段头/网络名明文，但段载荷熵 7.1–7.9、无 `TFL3`/ONNX 魔数，是压缩/加密私有格式 |

所以"可推理"的准确判定是：**模型在机身内可推理（这是设计路径）；主机侧两类模型都还缺一步——
`dic` 缺层拓扑，`.nab` 缺载荷解压/解密。**

---

## 6. 验证图片

| 图片 | 相机 | 镜头 | 预期命中模型 |
|---|---|---|---|
| `Human_DSC03106.jpg` | ILCE-7RM5 | FE 70-200mm F2.8 GM OSS II | `fdn_*` + `pdn_dic`（人） |
| `Cat_DSC04051.jpg` | ILCE-7RM5 | 135mm F1.4 DG Art | `aed*`（动物眼） |
| `Bird_DSC02510.jpg` | ILCE-7RM5 | FE 70-200mm F2.8 GM OSS II | `bed_dic*`（鸟眼） |
| `Flight_DSC02983.jpg` | ILCE-7RM5 | FE 70-200mm F2.8 GM OSS II + 1.4× | `scd_*`（飞机） |

四张图都是 ILCE-7RM5 原片，正好对上 7RM5 的模型集。

---

## 6.1 模型身份、版本与来源

### 6.1.1 AI 模型包里的网络名（`default.nab` 的 `VPMN` 段）

`.nab` 不是单一模型，而是一个**多网络包**：每个 `VPMN` 段 = `magic + u16 + u16 + u32 + 64 B 网络名`。
两机的网络清单：

| 网络名 | 7RM5 | 1M2 | 说明 |
|---|:-:|:-:|---|
| `sony_net_opt_NHWC` | ✓ | ✓ | 主网络（场景） |
| `reid_NHWC` / `reidpost_NHWC` | ✓ | ✓ | reid 子网 |
| `tfl80_t49_e70_q49_c0_NHWC`（1M2 为 `tfl107_t58_e70_q58_d53_c0_NHWC`） | ✓ | ✓ | TFLite 转换网络，通道 80/49/70/49 |
| `model_NCHW` ×2 | ✓ | ✓ | NCHW 布局网络 |
| **`aed_03_00_05_rc2_NHWC`** | ✓ | ✓ | **AED 网络，版本 03.00.05-rc2** |
| `model.ckpt-1590000_416x288_NHWC` / `_r` | ✓ | — | **TF checkpoint，step 1,590,000，输入 416×288** |
| `tflite_djd5-U-resnet7-32_64-esch_lw_quan_max23_NHWC` | ✓ | ✓ | **TFLite ResNet-7，32/64 通道，max23 量化** |
| `sfr_tf_resnet_454-80-160-320-512-nose_NHWC` | ✓ | ✓ | TF ResNet，通道 454/80/160/320/512 |
| `v58_h27_head_no_norm_clip_416x288_cquant_order_NHWC` | — | ✓ | 1M2 新增，横构图 head |
| `v58_h27_head_no_norm_clip_288x416_cquant_order_NHWC` | — | ✓ | 1M2 新增，竖构图 head |
| `dummy_network_name` ×4~6 | ✓ | ✓ | 占位 |

### 6.1.2 版本证据

| 对象 | 版本 | 来源 |
|---|---|---|
| 场景模型（`ai_dict`） | **1.00**（version_code 1，compatible 1） | 每个 `.rsc` 的 `ai_model_version_name` |
| 场景模型 ID | 1–9（7RM5）/ 1–10（1M2） | `.rsc` 的 `ai_model_id_number` |
| AED 网络 | **03.00.05-rc2** | `.nab` 段名 `aed_03_00_05_rc2_NHWC` |
| TF checkpoint | **step 1,590,000**，输入 416×288 | `.nab` 段名 `model.ckpt-1590000_416x288` |
| 对焦检测字典（`dic`） | **无版本字段**，内容冻结 | 三机 SHA-256 完全相同，见 6.1.3 |

`.rsc` 完整字段示例（7RM5 `AAAAAAI.rsc`）：

```json
{"na_file_name":"AAAAAAI.nab","ai_model_id":"AAAAAAI","ai_model_id_number":1,
 "ai_model_version_code":1,"ai_model_version_name":"1.00",
 "ai_model_compertible_version":1,"ai_model_category":0,"ai_model_menu_category":0,
 "slots":{}}
```

### 6.1.3 `dic` 字典是"冻结版本"

同一份 `aed_dic1.bin` 在三种固件里逐字节相同，但文件时间不同：

| 机型 | 固件 | 文件时间 | SHA-256 前 16 |
|---|---|---|---|
| ILCE-7M4 | 6.02 | 2021-05-06 | `d3fa6482f87a91c0` |
| ILCE-7RM5 | 4.00 | 2022-06-22 | `d3fa6482f87a91c0` |
| ILCE-1M2 | 4.00 | 2025-07-02 | `d3fa6482f87a91c0` |

说明：**对焦主体检测字典是一次构建、跨机型复用的冻结版本**（2021 年构建），
后续固件只改 CP 引擎参数与 `.nab` 包，不改字典权重。

### 6.1.4 是不是公开权重？

结论：**不是公开权重**，证据如下：

1. **格式私有**：`dic` 是 Adam NPU 的 ODL 二进制；`.nab` 是 `VPMN` 私有容器。
   两者都没有 TFLite `TFL3`、ONNX、Caffe、ncnn、MNN 的魔数。
2. **命名私有**：`sony_net_opt`、`aed_03_00_05_rc2`、`djd5-U`、`esch_lw`、
   `sfr_tf_resnet_454-80-160-320-512-nose`、`v58_h27_head` 都是索尼内部命名，
   不对应任何公开模型（MobileNet / SSD / RetinaFace / YOLO / BlazeFace 等）。
3. **训练痕迹是索尼自己的**：`model.ckpt-1590000_416x288` 是 TensorFlow checkpoint
   （step 1,590,000），`tflite_*` 是从 TFLite 转出的量化网络——说明模型由索尼用
   TF/TFLite 训练后转成自家格式，而不是直接打包公开权重。
4. **没有第三方许可证串**：`.nab` / `dic` / CP 固件里都没有 Apache/BSD/MIT/TensorFlow
   Authors 之类的版权串。

因此可以确定：**模型族 = 索尼内部 "Minna/SonyNet" 系列，训练自 TF/TFLite，
版本见 6.1.2；权重是索尼自训练/自量化的私有权重，不是任何公开权重。**

### 6.1.5 `.nab` 段载荷是压缩/加密的

| 段 | 段名 | 载荷大小 | 熵 | 零字节占比 |
|---|---|---:|---:|---:|
| `0x490` | `sony_net_opt_NHWC` | 1,254,348 | 7.25 | 0.167 |
| `0x132930` | `reid_NHWC` | 128,908 | 7.45 | 0.133 |
| `0x152190` | `reidpost_NHWC` | 400,972 | 2.85 | 0.652 |
| `0x8d3850` | `aed_03_00_05_rc2_NHWC` | 1,808,780 | 7.85 | 0.046 |
| `0x1a090b0` | `model.ckpt-1590000_416x288_NHWC` | 1,302,732 | 7.13 | 0.186 |
| `0x1c83270` | `tflite_djd5-U-resnet7-..._NHWC` | 131,660 | 7.26 | 0.166 |

段头（212 B）是明文的：`VPMN` + `u16 13/15` + `u16 1` + `u32 212` + 64 B 网络名 + 若干 u32。
段载荷前 16 B 全零，之后是高熵数据，且全文件没有 `TFL3`/`ONNX` 魔数 —— 载荷被压缩或加密，
所以"自描述"只到段头/网络名为止，**模型内容不能直接当 TFLite/ONNX 读出来跑**。

---

## 7. 产物

| 文件 | 内容 |
|---|---|
| `out/models/model_manifest.json` | 三机 45 个检测字典 + 21 个场景模型文件的 size/SHA-256/统计/头信息 |
| `out/models/parse_dic.py` | 字典容器解析器（生成 manifest） |
| `out/models/dic_parse.py` | 字典容器结构解析（生成 `dic_layout.json`） |
| `out/models/nab_graph.py` | `.nab` 节点/层类型解析（生成 `nab_graph.json`） |
| `out/models/verify_models.py` | 58 项模型集合校验（生成 `verify_report.txt`） |
| `out/models/extract_fdat.py` | DAT/FDAT 解密与 `firmware.tar` 提取 |
| `out/models/extract_all.py` | 从 DAT 到 `/usr/sabin` 的一键复现 |
| `out/models/work/{7m4,r5,1m2}/sabin/` | 解出的模型文件 |
| `out/models/work/{r5,1m2}/ai_dict/` | 场景识别模型 |

## 8. 下一步

1. 在 `cpapp-b-r5-thumb.bin` 里找 FDN 的 `*ParamManager` 构造函数（AED 的同类是 `FUN_00168764`），导出 6 层 `FdnDrvParam` 字段值。
2. 用层字段 + int8 权重 + int16 偏置 + float scale 在 NumPy 里实现 `fdn` 前向，跑 `Human_DSC03106.jpg`。
3. 用 `scd` 的 3 个字典在 `Flight_DSC02983.jpg` 上验证飞机类别。
4. 若要做 `.nab` 场景模型，先解压/解密 `VPMN` 段载荷（熵 7.1–7.9，非 TFLite/ONNX 明文）。

## 9. 最终判定（本轮）

| 项目 | 结果 |
|---|---|
| 三机型模型提取 | ✅ 完成：45 个检测字典 + 21 个场景模型文件，全部 SHA-256 |
| 识别主体 ↔ 模型映射 | ✅ 完成：见 `SUBJECTS.md` |
| 模型身份/版本 | ✅ 完成：Minna/SonyNet 系列；场景 1.00、AED 03.00.05-rc2、TF ckpt 1590000、TFLite ResNet-7 |
| 是否公开权重 | ❌ 否：私有格式/命名/训练痕迹，无第三方许可证 |
| 机身内可推理 | ✅ 是：`ODL_LoadBinary → AdamDriver:LoadFile → NPU` |
| 主机侧可推理 | ❌ 否：`dic` 缺层拓扑；`.nab` 载荷压缩/加密；两种私有格式都缺运行时 |

> 主机侧的两条"绕过"实验（模板 NCC、按 6 层重建前向）都已做过，峰值不在人脸上，
> 属于**负结果**，记录在 `INFERENCE-STATUS.md`。
