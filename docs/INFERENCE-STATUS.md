# 推理状态：dic 模型在主机侧的可行性核查

> 目标：用本机工具正确推理三台相机的视觉检测模型，并做视觉核对（框要选中对应物体）。
> 约束：不允许使用机身；不允许修改目标。

## 1. 已尝试的路径与结果

| 尝试 | 方法 | 结果 |
|---|---|---|
| 标准格式识别 | 在 7 个 dic 文件里搜 `TFL3` / `ONNX` / `caff` / `MNN` / `ncnn` | 全部无命中 |
| 压缩识别 | 首字节 `0xF0` / `0x0F`（Sony LZ77） | 不是压缩 |
| 模板匹配假设 | 把 `fdn` section0 当 4 个 12×24 int8 模板，做 NCC | 峰值 0.23，落在墙面/石头上，不在人脸 |
| 6 层重建 | 按 header 的 6 层 / 3×3 / stride 4，通道数按 section 尺寸反解，NumPy 前向 | 8 个响应峰落在腿、脚、墙、地面，**不在人脸**（见 `fig_fdn_detect.png`） |
| 容器结构 | `fdn` 4 段头 + 权重 + `aed2` s2f 表 | 结构可解析，但只有 fdn 有显式尺度头 |
| 层拓扑 | CP 固件 `FdnParamManager::buildUpParam`（`0x16ddfc`） | 只解析 MMU 地址，不含拓扑 |
| 层拓扑 | `FdnParamManager::buildParam2`（`0x16df04`） | 从 `FdnDrvParam` 拷逐层 u16 字段，拓扑在 `FdnDrvParam` |
| 构造函数 | AED 的 `FUN_00168764` 已反编译（写 0x200/0x400/0x40/0xc8/0x93a00/0x1e000/0x3e/0x2a/0x2b/0x28/0x29/0xd/0x2c） | FDN 的同类构造函数尚未定位 |

## 2. 结论

**当前不能在主机侧正确推理 `dic` 模型。**

- 权重是 int8 + int16 偏置 + float scale，但**层拓扑不在 dic 文件里**（只有 `fdn` 有 4 段尺度头，没有逐层通道/输入尺寸）。
- CP 固件的 DDL 层把拓扑放在 `FdnDrvParam` 里，由 `*ParamManager` 构造函数填；AED 的构造函数已找到，**FDN/Pdn/Scd/Bed 的构造函数尚未定位**。
- 没有拓扑时，任何"检测框"都只能是猜测，不满足"正确推理 + 视觉核对"。

## 2.1 模型身份与版本（已确定）

| 对象 | 身份 | 版本 |
|---|---|---|
| 对焦字典 `dic` | Sony `Camera::Minna` 主体检测字典（FDN/PDN/AED/AED2/BED/SCD/UDRT） | 无版本字段，三机哈希相同 → 2021 年构建的冻结版本 |
| AI 模型包 `.nab` | Sony `VPMN` 多网络包 | `.rsc`：version 1.00（code 1，compatible 1），model id 1–9 / 1–10 |
| `.nab` 内 AED 网络 | `aed_03_00_05_rc2_NHWC` | **03.00.05-rc2** |
| `.nab` 内 TF checkpoint | `model.ckpt-1590000_416x288` | **step 1,590,000**，输入 416×288 |
| `.nab` 内 TFLite ResNet | `tflite_djd5-U-resnet7-32_64-esch_lw_quan_max23_NHWC` | ResNet-7，32/64 通道，max23 量化 |
| `.nab` 内 TF ResNet | `sfr_tf_resnet_454-80-160-320-512-nose_NHWC` | 通道 454/80/160/320/512 |

`.nab` 的 `VPMN` 段头（212 B）是明文，段载荷熵 7.1–7.9、前 16 B 全零、无 `TFL3`/`ONNX` 魔数
→ 载荷是压缩/加密的私有格式，主机侧不能直接当 TFLite/ONNX 读。

**是否公开权重：否。** 格式私有（ODL / VPMN）、命名私有（sony_net_opt / aed_03_00_05_rc2 / djd5-U /
sfr_tf_resnet_454-80-160-320-512-nose / v58_h27_head）、训练痕迹是索尼自己的 TF checkpoint，
且无第三方许可证串。

## 3. 下一步（按可行性排序）

1. **定位 FDN 的 `*ParamManager` 构造函数**：按 AED 构造函数（`FUN_00168764`）的模式，在 CP 固件里找同样写 `0x200/0x400/0x40/0xc8/0x93a00/0x1e000` 的函数；找到后导出 6 层字段（宽/高/通道/核/步长/pad）。
2. **用层字段 + int8 权重 + int16 偏置 + float scale 在 NumPy 实现前向**，跑 `Human_DSC03106.jpg`，画出框并目视核对。
3. 若 `fdn` 跑通，再按同样方法做 `pdn`（人）、`aed2`（动物）、`bed`（鸟）、`scd`（飞机）。
4. `.nab` 的段头/网络名是明文，但载荷是压缩/加密私有格式；如要跑场景模型，需要先解压/解密载荷。

## 4. 相关产物

| 文件 | 内容 |
|---|---|
| `out/models/exp_ncc.py` | 模板匹配假设的 NCC 实验（负结果） |
| `out/models/fig_models.png` | 模型内部 + 样本图 + 声明窗口 |
| `out/models/fig_pipeline.png` | 推理路径与卡点 |
| `out/models/scan_param_writes.py` | 参数构造函数扫描器 |
