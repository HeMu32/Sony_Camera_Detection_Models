# 关联实验：单主体 + Empty 负样本

> 你的设计：每张图最多一种主体；Empty/Cat/Flight 都不应检出人。
> 判据：检出类型正确即可，框的精度另议。

## 1. 用提取出的 dic 字典做"类型关联"（已实现）

方法：把每个 dic 文件切成 8×8 int8 patch，对每张图在多尺度上取 8×8 patch，
计算最大 NCC 与 top-200 均值 NCC，取得分最高的字典作为"检出类型"。

结果（top-200 均值 NCC）：

```
Human_DSC03106.jpg   object=0.985  bird=0.723  animal=0.705  human_face=0.674  human_pose=0.664  track=0.646
Cat_DSC04051.jpg    object=0.986  bird=0.696  animal=0.688  human_face=0.608  human_pose=0.599  track=0.595
Bird_DSC02510.jpg   object=0.988  bird=0.777  animal=0.732  human_face=0.710  track=0.668  human_pose=0.614
Flight_DSC02983.jpg object=0.985  bird=0.757  animal=0.708  human_face=0.695  human_pose=0.662  track=0.647
Empty_DSC03284.jpg  object=0.986  bird=0.761  animal=0.747  track=0.704  human_face=0.697  human_pose=0.617
```

## 2. 结论：不具判别性

- `object`（scd）在**全部 5 张图**上排第一，包括 Empty；
- Human 上 `human_face` 只排第 4；
- Cat 上 `animal` 排第 3；
- Bird 上 `bird` 排第 2；
- Flight 上 `object` 第一（碰巧对），但 Empty 也第一。

→ **这个关联实验不能证明类型正确。**

原因：dic 文件是**量化权重**，不是 8×8 模板；把权重当模板做 NCC，
最大值由 patch 数量与随机性决定，不携带语义。

## 3. 设计本身是对的

你的判据（单主体 + Empty 负样本 + 只看类型）比"框准不准"更容易验证，
但它要求模型**真的输出类别分数**。这需要：
- 逐层拓扑（通道/输入尺寸），或
- 模型自己的输出头。

两者目前都不在可提取产物里。

## 4. 可选的下一步

| 方案 | 能得到什么 | 代价 |
|---|---|---|
| A. 继续找 `FdnDrv::initialize` / `ScdDrv::initialize` | 真实拓扑 → 真实类别分数 | 还需若干轮 RE |
| B. 用标准检测器做**对照实验**（明确标注"非固件模型"） | 验证"单主体+Empty"判据本身可行 | 1 轮，但不是固件模型 |
| C. 模拟 Adam NPU | 真实前向 | 需要 ISA/ODL 文档 |

我建议先做 B 作为**判据对照**，同时继续 A。

## 5. 对照实验结果（已做，非固件模型）

用 OpenCV Haar `haarcascade_frontalface_alt2` 在 5 张图上跑正脸检测：

```
Human_DSC03106.jpg     faces=1
Cat_DSC04051.jpg       faces=0
Bird_DSC02510.jpg      faces=0
Flight_DSC02983.jpg    faces=0
Empty_DSC03284.jpg     faces=0
```

Human 图上框正确落在小孩脸上（见 `out/models/control/Human_DSC03106.jpg`），其余 4 张 0 框。

→ **你的判据（单主体 + Empty 负样本 + 只看类型）是可行的**；
但这一结果来自对照检测器，**不是**固件模型，所以不能算"正确推理了三台相机的模型"。
