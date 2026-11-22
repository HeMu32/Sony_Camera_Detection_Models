# Sony 相机视觉检测模型

从 Sony 固件提取原始视觉检测模型，在主机上运行并用实拍样例核对结果。
不使用相机机身、VIP9000 硬件或替代模型。

| 机型 | 当前进展 |
|---|---|
| **A7R V / A1 II** | 原始 NBG 可在 x86 软件模拟器执行；鸟和飞机有 CUDA 路径和原 Adam 框解码。人体和猫有宿主侧实验构框，不等于原厂后处理。 |
| **A7 IV** | 已定位 FDN／PDN 门限、AED／BED 参数和部分结果路径，并回放原 SA 控制流。首个图像算子的算术尚未复现，没有可验收的实图框。研究已暂停，未作为完成版交付。 |

## 可用交付物

- **独立运行模型：**[ONNX / TorchScript 模型表](docs/PORTABLE-EXPORT.md)链接到仓库中的五槽导出及接口说明。槽 `1/8/10`（人体／鸟／飞机）由两机共享，猫分别是 A7R V 槽 `12` 和 A1 II 槽 `18`。模型输出张量，不内置读图或框解码。
- **实图结果：**原 Adam 解码的[鸟](debug/visual_checks/adam-native-r5-bird.png)、[飞机](debug/visual_checks/adam-native-1m2-flight.png)、[天际线](debug/visual_checks/adam-native-r5-empty.png)叠图。鸟图有两条对位框但仍有漏检，飞机略切机鼻，天际线零框。[原解码与全部结果](docs/ADAM-XHD.md)
- **复现：**[CPU／CUDA 快速开始](docs/TWO-CAMERA-QUICKSTART.md)和[舍入对照](docs/NBG-ROUNDING-AB.md)。CUDA 不用 FP64；与固定 x86 SDK 的逐字节一致仅适用于已测样本，不代表与实机位一致。完整固件、CP 镜像、第三方 SDK 和大型中间转储不随仓库提供。

人体和猫的框是从网络输出构造的原图像素 `xyxy`，不是地标点。不同槽分别运行后在主机合并，也不是 A1 II 固件同帧跨类 Auto 的证明。

## 研究入口

- [检测流程与证据边界](docs/AF-VISION-PIPELINE.md)
- [A7 IV 固件、门限和未完成的主机前向](docs/A7IV-SA-FORWARD.md)
- [最新暂停点与历史交接](HANDOVER.md)

样例在 [`assets_aux/`](assets_aux/)，原始网络与解析脚本在 [`nab_graphs/`](nab_graphs/) 和 [`parsers/`](parsers/)，主机执行和回归工具在 [`tools/`](tools/)。
