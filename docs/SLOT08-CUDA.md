# Sony 槽 8：独立 GPU 推理

A7R V 和 A1 II 从固件提取的原始 `slot08.nb` 逐字节相同，SHA-256
`28782d52ac73d7a537a2576bd8f9e4e6b5f7ccd23a420093e1a1a14b556d06f9`。
`tools/slot08_cuda.py` 从该图的原始执行记录提取实际系数、偏置、尺度和 TP
再量化参数，打包为 3,727,101 字节的 `debug/cpu_raw/coeff_probe_01/sony-slot08-cuda.npz`。
鸟图与天际线的独立跟踪解出的 253 个包字段均一致。推理时只读 `.npz` 和输入，
不需要 CPU 模拟器、跟踪文件、参考张量、机身或 VIP9000 硬件。

## 运行与画框

当前环境为 `test311`、CUDA PyTorch 和 CuPy。在工作区运行：

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
$d = 'debug/cpu_raw/coeff_probe_01'
& $py tools/slot08_cuda.py infer "$d/sony-slot08-cuda.npz" assets_aux/Bird_DSC02510.jpg "$d/DemoBird"
& $py tools/render_anchor_probes.py assets_aux/Bird_DSC02510.jpg debug/visual_checks/demo_gpu_bird.png --manifest "$d/DemoBird.manifest.json" --prediction 08 "$d/DemoBird" "$d/DemoBird.log"
```

`infer` 接受带 EXIF 方向的图片文件，或 `3×320×320` 平面 RGB `uint8` 文件。
它导出 OpenVX 编号的 21 份原始文件和一份接口形状／量化描述；图片输入还导出原图坐标变换。
渲染器仍使用 `40×40` 尺度和实验先验、阈值及 NMS，不是 Sony 的机内解码。

重建包：

```powershell
& $py tools/slot08_cuda.py pack nab_graphs/r5_slots/slot08.nb "$d/Bird69.nbc" "$d/Bird.nbt" "$d/sony-slot08-rebuilt.npz"
```

## 原图与 GPU 的对照

- 前向包含 60 个去除硬件切片重复后的卷积块，使用原始 int8 系数、整数偏置、精确范围内的 FP32 GEMM、FP32 FMA 重量化。运行时没有 FP64。对 `40、20、10` 边长的特征采用 TP `2×2` 最大池化；`5→3→2→1` 使用原始零字节补边、`2×2` 重排到通道、原始 `512` 输入通道的 `2×2` 卷积。分类、框回归、辅助输出分别产生 `40、20、10、5、3、1` 六个尺度。
- 12 个分类／回归输出使用原始 TP 尾数、移位和零点。TP 先在乘积及加输出零点时保留 16 位有效精度，再取整至字节。`20×20` 回归的 `±25` 用例验证了单次直接舍入会错的边界。六个辅助输出按原图通道顺序写出 sigmoid 量化值。
- 输出 0、7、14 各为 6 字节。CPU 探针在 `vxVerifyGraph` 后预填独立非零图样，推理后三个图样均原样保留，其余 18 个输出不变。它们不含推理结果；GPU CLI 只把这些接口槽初始化为零，以匹配正常 CPU 分配行为。Python 推理 API 返回 18 个实际计算出的张量。
- 鸟图、天际线及 A1 II 的鸟／飞机合成图各有 21 个输出、128,118 字节，GPU CLI 对照 CPU 均零差异。`StandaloneComplete{Bird,Empty,Mixed}.*.raw` 保留独立 GPU 结果。`Bird69.nbc` 和 `Empty69.nbc` 覆盖全部 69 条 NN 指令；两次跟踪的原始图输出也都与各自基准完全一致。
- RTX 5060 Ti，输入及权重已在 GPU，预热 8 次，取 5 组各 20 次运行的 CUDA Event 中位数：**22.0 ms/张**。不含读图、预处理、一次性载入、主机回读、写盘或框解码。

目视核对见 `debug/visual_checks/gpu_slot08_bird.png`、`gpu_slot08_empty.png` 和
`gpu_slot08_mixed_with_cpu_slot10.png`。鸟图两框各选中一只鸟，仍有漏检；天际线零框。
上述合成图的鸟框来自 GPU 槽 8，飞机框来自 CPU 槽 10。后续合成图已改用两个 GPU
前向，见 [槽 10 GPU 核验](SLOT10-CUDA.md)。两者都不证明 A1 II 机内 Auto 的调度策略。

## 鸟类漏检与槽 7 裁片对照

在同一模型、同一主机解码阈值 `0.85` 下，用全图和四个固定重叠切片执行
槽 8 CUDA。切片宽高取原图的 `2/3` 向上取整；没有用目标位置选切片。
报告保留每张输入的 SHA-256、六层类别 1 的最高值、越阈候选及对应原始框回归量。
全图输入哈希和已存整图推理一致，两个最终框的原图坐标也一致。

- 鸟图全图 `40×40` 的 10 个越阈候选集中在两只已框到的鸟附近。
  下方两片在 `40×40` 的峰值只有 `0.279476`，但 `20×20` 各有 9、10 个越阈
  候选，峰值均为 `0.999954`。两片的最高格点投回原图均约为 `(1728,1775)`，
  是已检出的中央白鸟附近，不是新增主体。`20×20` 尚无已确认的先验，未把
  这些激活当作框。`40×40` 解码及跨片合并结果仍为两框，目视确认没有提高召回。
- 天际线五次推理均无越阈候选，合并零框，已目视核对。该阴性图不代表一般误检率。

两机鸟类 RSC（`AAAAABY.rsc`，类别字段 `2`）都列出 `"slot1":7,"slot2":8`。
配置证明配对，未证明运行顺序或裁片如何生成。两机原始槽 7 的 SHA-256 同为
`42f735d7b31471fe811e6cfbf2e49104be59d1e0a3f17879175cab4dfd619ea4`，
模型名 `aed_03_00_05_rc2_NHWC`。固定版 x86 SDK 验证接口 `128×128×3×1`
输入（零点 127），输出编号 `0..3` 分别为 `3×1`、`32×32×3×1`、
`3×1`、`3×1`。在虚拟机用未经修改的 R V 原图执行三个手选裁片：

| 裁片 | 输出 0（反量化） | 输出 1 三个通道峰值 | 视觉核对 |
|---|---|---|---|
| 槽 8 已检出的白鸟 | `0 / 0 / 0.99609` | `1.01265 / 0.94514 / 0.88954` | 前两路同点落在眼附近，第三路在喙附近 |
| 槽 8 漏检的棕鸟 | `0.60547 / 0.00391 / 0.39062` | `0.27401 / 0.24621 / 0.30975` | 较弱的响应仍在眼、喙附近 |
| 天际线同坐标裁片 | `0.98828 / 0.00391 / 0.00781` | `0.01588 / 0 / 0` | 唯一弱峰在 letterbox 填充区 |

这证明槽 7 对漏检主体的**已知裁片**还保留空间线索；没有证明槽 7 能自己定位
该裁片。没有给通道或分数赋原厂语义；第一路峰值大于 1，不能一律称为概率。
三个样例的输出 2 全零，尚未用非零预填验证是计算结果还是未写槽。
裁片的标记是峰值诊断，不是全图检测框，不计作召回改进。

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
$d = 'debug/cpu_raw/coeff_probe_01'
& $py -B tools/probe_slot08_tiled.py "$d/sony-slot08-cuda.npz" debug/visual_checks/bird_tiled_probe.png assets_aux/Bird_DSC02510.jpg
& $py -B tools/probe_slot08_tiled.py "$d/sony-slot08-cuda.npz" debug/visual_checks/empty_tiled_probe.png assets_aux/Empty_DSC03284.jpg
& $py -B tools/inspect_slot07_crops.py debug/cpu_raw/slot07_crop_probe_vm debug/visual_checks/slot07_crop_probe.png
& $py -B tools/test_slot08_tiled_probe.py -v
```

槽 7 原始输入、裁片坐标、输出和日志在 `debug/cpu_raw/slot07_crop_probe_vm/`。
可将同目录的三个无编号 `.raw` 分别交给 `run_sony_nbg_cpu.sh` 和
`r5_slots/slot07.nb` 重跑；输出前缀取对应裁片名。

DeepSeek 的原厂区域路径报告在 `debug/deepseek_anchor/BIRD-SLOTS-NATIVE-CROP.md`。
独立核对两份 CP 字节：R V 文件偏移 `0x1d1076`、A1 II `0x208f7e`
确实载入 `640.0f`，用于区域尺度归一化；R V `0x1d1b7a`、A1 II
`0x209d2a` 均按 `(基准尺寸 × u8 字段) >> 4` 计算一路正方形区域大小。
这些是区域几何参数，不是槽 8 分数阈值。报告中的四路 `RegionType` 分派
也不足以证明只有一路使用 XHD 输入；目前仍未把鸟类 RSC 对应的运行模式
连到具体的区域生成器和槽 7 提交点。
