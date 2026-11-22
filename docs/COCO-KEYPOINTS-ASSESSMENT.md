# Sony 人体检测模型与 COCO Keypoints 的兼容性评估

日期：2026-11-18。结论针对两机共用的原始槽 1 `sony_net_opt_NHWC`；未连接机身、未使用 VIP9000 硬件。

## 结论

**可以导出成 COCO Keypoints 的形状，但当前不能把它的 COCO AP 当作 Sony 模型的有效跑分。**

| 项目 | 判断 | 依据 |
|---|---|---|
| COCO 所需 17 个关键点位置 | 有条件可生成 | 原始输出有 17 个定位热图通道；可从峰或质心投回原图 |
| COCO 名称顺序 | **未确认** | 19 通道是 17+2 的强证据，但 0..16 与 nose/eyes/ears/... 的固定映射没有从固件或模型元数据取得 |
| 每点可见性/置信度 | **不能按 COCO 原义直接生成** | 热图峰值不是 COCO `v=0/1/2`；当前导出的 `v` 是主机阈值映射 |
| 17 点归属同一人 | 依赖主机分组 | 模型给出候选点；固件有 `JnpDetectPose` 的分组路径，但尚未复刻 |
| `bbox` | 仅实验可导出 | 当前由关键点几何外扩生成，非 Sony 原生人框 |
| `score` | 仅实验可导出 | 当前取人体上、下中心峰最低分，不是官方 person score |
| 输入预处理 | 部分可重做 | 当前 `letterbox 512×336, RGB, bilinear, pad=128` 是实验参数，不保证与 COCO 官方评估或机内一致 |
| 多实例 | 部分满足 | 样图 6 个可见人物全部出框；更远/被遮挡人物有漏检，不能据此断言 COCO 全量性能 |
| 人群标注 `iscrowd=1` | 不适用/不兼容 | 当前单实例关键点分组输出无法从 COCO crowd 区域推导关键点 |

因此合理表述是：**Sony 人体网络是 COCO 风格的人体关键点热图模型，可写一个 COCO 形状的适配器；当前主机后处理不是 COCO 官方解码器，也不具备可复核的 COCO Keypoints AP。**

## 原始数据契约

CPU 模拟器实测输入输出：

```text
input:  512 x 336 x 3 x 1, uint8, scale=1/255, zp=0
out 1:   64 x  42 x 40 x 1, uint8, scale=0.00762748485, zp=127
out 2:   64 x  42 x 19 x 1, uint8, scale=0.00384146371, zp=4
```

OpenVX 打印是 `W×H×C×N`；实际连续存储下读取为 `[N,C,H,W]`。两机槽 1 模型 SHA-256 相同：
`a5d5ead9478e3b069598483bda95702283d7b0c3d825de02c67250eeea4394b6`。

热图 19 通道的实测点与人体部位吻合。峰值最高的通道 0、1、17 与头/上身中心对应；通道 15、16 多次落在脚踝。`debug/visual_checks/human_pose_boxes.json` 的六个人分别拥有 13 到 16 个可见点。天际线图的同一热图最大值为 `0.0115244`，低于实验阈值 `0.25`，导出 0 人。

不过，“热图能定位人体”不等于“通道顺序已证明”。例如，人体图中最高峰附近有通道 0/1/17 的强响应；在 17+2 的设计里它们可能是鼻、眼、头中心之类，但目前不能据此把任意一个通道硬命名为 COCO nose。更不能仅根据左右对称就判定 `left/right`，因为人体朝向、镜像训练和相机坐标都可能改变语义。

40 通道伴随输出也已做局部向量检查：20 对通道，每对包含 `(dx,dy)`。在人体图上，成对向量经常沿两个实测关键点之间的方向；例如若干配对在场线上与肩、髋、膝、踝的连接方向一致。这个结果支持它是关键点连接场（PAF 类），而不是 COCO 检测框或逐像素分割；但具体 20 对对应哪 20 条人体骨架边尚未恢复，因此当前也不能靠它替代 COCO 通道名称。

## 可生成和不可生成的内容

`tools/export_pose_coco.py` 生成标准 COCO annotations JSON 的顶层结构：`info`、`images`、`categories`、`annotations`。每个人带 `id`、`image_id`、`category_id`、17×3 的 `keypoints`、`num_keypoints`、`bbox`、`area`、`iscrowd`、`score`，图片尺寸为原图 `3840×2560`。

产物：`debug/visual_checks/human_pose_coco.json`，六人；`empty_pose_coco.json`，零人。JSON 的 `_sony_export` 明确标出通道映射和可见性映射是占位规则，不是固件协议。因此它适合下游可视化、格式交换和适配器开发，不适合把数字交给 COCO 官方 AP 后宣称 Sony 成绩。

格式核验：`pycocotools` 的 `COCO` 与 `COCOeval('keypoints')` 能加载 `human_pose_coco.json`，六条记录的 `keypoints` 均为 `17×3`，图片为 `3840×2560`，空图导出 0 条。但这只证明“解析器能读”，不证明通道名称或可见性正确。若把导出的预测又当成 GT 自评，官方脚本会给出 `AP=1.0`；这是自我匹配的假结果，不能作为 Sony 的 COCO 分数。

当前可见性规则只是：

```text
未选中/落在原图外 -> 0
已选中且热图分数 < .5 -> 1
已选中且热图分数 >= .5 -> 2
```

它不是 COCO 的“遮挡/不可见/可见”语义。COCO 官方评测还会按标注的 `v` 和 bbox/area 处理，故照搬会把主机后处理的阈值错误地当作数据集标签。

## 如果要正式跑 COCO Keypoints AP

在跑分前需要补齐以下四件事：

1. 从固件/模型元数据确认 17 个热图通道到 COCO 17 个名称的精确映射，包含左右。
2. 取得 Sony 的分数阈值、点关联、姿态选择和框生成规则，或明确只评“主机适配器”。
3. 固定与 COCO 评测一致的预处理。当前输入模型是 `512×336` 宽高比 `1.5238`，COCO 原始图比例各异；letterbox、缩放插值、pad 和反变换必须由同一脚本处理。
4. 用 COCO GT 的 17 个 `v` 及 `iscrowd` 规则做匹配。不要拿本目录的 6 个实验框冒充标注，也不要按检测框 IoU 代替 OKS。

在这些条件满足前，能做的是：

- 输出可校验的 COCO 形状 JSON；
- 做独立的逐图视觉检查、点覆盖率、人体框召回率、空图假阳性等指标；
- 与 COCO 官方后处理/官方 AP 做“适配器质量”实验，并明确结果不是原模型的官方分数。

## 复现

```powershell
& 'E:\anaconda3\envs\test311\python.exe' tools/export_pose_coco.py `
  debug/visual_checks/human_pose_boxes.json `
  debug/visual_checks/human_spatial_peaks.json `
  debug/cpu_raw/r5_s01/Human `
  debug/cpu_raw/r5_s01/detail.log `
  debug/cpu_inputs/512x336_rgb/manifest.json `
  assets_aux/Human_DSC03106.jpg `
  debug/visual_checks/human_pose_coco.json
```

视觉对照：`debug/visual_checks/human_pose_boxes.png`、`empty_pose_boxes.png`。原始输出是 `debug/cpu_raw/r5_s01/{Human,Empty}.{0,1}.raw`。
