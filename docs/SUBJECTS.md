# 识别主体 → 模型文件对照表（三机型）

> 所有文件位于各机型 `/usr/sabin/`（场景模型在 `/usr/data/ai_dict/`）。
> 检测权重三机逐字节相同；差异只在 `scd`（主体类别）与 `ai_dict`（场景）。
> 证据：CP 固件 `Camera::Minna` RTTI 的 `DIC_NUM`、`judgment*` 判决函数、`AmS_*Detect*` 消息。

## 1. 对焦主体（AF）

| 识别主体 | 检测部件 | 模型文件 | 7M4 | 7RM5 | 1M2 |
|---|---|---|---|---|---|
| 人 | 正脸 | `fdn_front_dict.bin` | ✓ | ✓ | ✓ |
| 人 | 侧脸 | `fdn_side_dict.bin` | ✓ | ✓ | ✓ |
| 人 | 身体/姿态 | `pdn_dic.bin` | ✓ | ✓ | ✓ |
| 动物 | 眼部（v1） | `aed_dic1.bin`、`aed_dic2.bin` | ✓ | ✓ | ✓ |
| 动物 | 眼部（v2） | `aed2_dic1.bin`、`aed2_dic2.bin`、`aed2_dica_s2f.bin`、`aed2_dicb_s2f.bin` | ✓ | ✓ | ✓ |
| 鸟 | 眼部 | `bed_dic1.bin`、`bed_dic2.bin` | ✓ | ✓ | ✓ |
| 昆虫 / 汽车 / 火车 / 飞机 | 主体类别 | `scd_dic1.bin`、`scd_dic2.bin`、`scd_dic3.bin` | — | ✓ | ✓ |
| 通用跟踪 | 框/关键点 | `udrt_dic1.bin`、`udrt_dic2.bin` | ✓ | ✓ | ✓ |

`scd` 只在 7RM5 / 1M2 出现，与"A7M4 只有人/动物/鸟，A7R5 与 A1 II 额外有昆虫/汽车/火车/飞机"完全对应。

## 2. 场景识别（AI 自动识别场景）

| 模型 | 文件 | 7M4 | 7RM5 | 1M2 |
|---|---|---|---|---|
| 场景分类 | `ai_dict/default/default.nab` + 9/10 个 `.rsc` | — | ✓ | ✓ |

`.rsc` 是 JSON（`na_file_name` / `icon_tar_file_name` / `dov_file_name`），
`.nab` 是自描述图（`sony_net_opt_NHWC`、`input[0..15]`、`ConvolutionReluPoolingLayer2` 236 次、`FullyConnectedReluLayer`、`reid` 子网）。

## 3. 验证图片对应

| 图片 | 相机 | 命中模型 |
|---|---|---|
| `Human_DSC03106.jpg` | ILCE-7RM5 | `fdn_*` + `pdn_dic` |
| `Cat_DSC04051.jpg` | ILCE-7RM5 | `aed*` |
| `Bird_DSC02510.jpg` | ILCE-7RM5 | `bed_dic*` |
| `Flight_DSC02983.jpg` | ILCE-7RM5 | `scd_*` |

四张图 EXIF 均为 ILCE-7RM5，正好对上 7RM5 的模型集（含 `scd`）。

## 4. 逐文件清单

见 `model_manifest.json`（45 个检测字典 + 21 个场景模型文件，含 SHA-256）
与 `dic_layout.json`（fdn 4 段头、aed2 s2f 145 值）。
