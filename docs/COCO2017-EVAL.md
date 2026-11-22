# Sony 7RM5 / 1M2：COCO 2017 交集评测

日期：2026-11-19。范围为 ILCE-7RM5（A7R V）与 ILCE-1M2（A1 II），不包含 7M4。本报告只计入 Sony 独立前向和 YOLO 的 COCO 推理结果。Sony 使用阶段交付 `95c9834` 的 CUDA 权重和实验解码方案，以及后续的 1M2 槽 18 可移植前向；没有连接相机或使用 VIP9000 硬件。`simulator` 是独立前向的舍入模式名，本报告中的 COCO 推理不调用 VeriSilicon SDK 模拟器。YOLO 检测和人体姿态各跑一次，分别对应不同任务。

## 八组覆盖

| 组 | Sony 框架 | 机型 | 任务 | 当前覆盖 |
|---:|---|---|---|---|
| 1 | 独立前向 `simulator` | 7RM5 | 非人物检测 | 鸟／飞机全量；动物缺少独立前向，未评测 |
| 2 | 独立前向 `simulator` | 7RM5 | 人体姿态 | 缺少独立前向，未评测 |
| 3 | 独立前向 `simulator` | 1M2 | 非人物检测 | 鸟、飞机、动物（猫狗）全量 |
| 4 | 独立前向 `simulator` | 1M2 | 人体姿态 | 缺少独立前向，未评测 |
| 5 | 独立前向 `simple` | 7RM5 | 非人物检测 | 鸟／飞机全量；动物缺少独立前向，未评测 |
| 6 | 独立前向 `simple` | 7RM5 | 人体姿态 | 缺少独立前向，未评测 |
| 7 | 独立前向 `simple` | 1M2 | 非人物检测 | 鸟、飞机、动物（猫狗）全量 |
| 8 | 独立前向 `simple` | 1M2 | 人体姿态 | 缺少独立前向，未评测 |

两机的槽 1（人）、8（鸟）、10（飞机）各自逐字节相同。相同权重、输入和解码只推理一次，组 1/3、5/7 的鸟飞机预测分别相同。两机的动物原图不同（7RM5 槽 12、1M2 槽 18），不能把 1M2 成绩套用到 7RM5。`pose-parameters` 中的 `.pt` 只有参数，不是可执行图；1M2 动物的全量评测还使用了独立的可移植计算图。

## 全量检测

COCO `val2017` 共 5,000 张，保留负图；鸟有 125 张正图／427 个实例，飞机 97 张／143 个实例。所有数值以百分比显示。P/R 为各方法在**相同两类**上的 Ultralytics 平滑平均 F1 最优工作点，不是统一数值置信度。

| 模型／框架 | 类别 | Precision | Recall | AP50 | AP75 | AP50–95 |
|---|---|---:|---:|---:|---:|---:|
| YOLO26n | 鸟 | 74.43 | 37.50 | 45.64 | 30.55 | 29.76 |
| Sony `simulator`（两机） | 鸟 | 14.78 | 5.62 | 3.26 | 1.57 | 1.69 |
| Sony `simple`（两机） | 鸟 | 14.78 | 5.62 | 3.48 | 1.83 | 1.90 |
| YOLO26n | 飞机 | 86.80 | 76.22 | 88.66 | 78.23 | 71.26 |
| Sony `simulator`（两机） | 飞机 | 18.24 | 9.79 | 8.90 | 7.15 | 6.25 |
| Sony `simple`（两机） | 飞机 | 18.24 | 9.79 | 8.90 | 7.15 | 6.21 |

- Sony 两种舍入：鸟有 188 张的输出记录变化（其中 10 张框数变动），飞机 89 张（1 张框数变动）。AP50–95 的最大变化为 0.216 个百分点，不改变上述结论。变化统计针对解码预测 JSON，不是原始张量差异率。
- 交付示例使用 `conf ≥ 0.85`。该门限下，Sony 两种模式在鸟类均为 10 TP / 17 FP，飞机均为 12 TP / 13 FP（IoU 0.50）；对应召回 2.34% 和 8.39%。YOLO 的同数值门限可另查 JSON，但数值相同不表示置信度校准相同。
- YOLO 在 5,000 张内的猫和狗 AP50–95 分别为 72.51% 和 65.20%。它们是基线自身的类别分数；不能拿去逐物种对比一个不分猫狗的 Sony `animal` 槽。

## 1M2 动物全量

槽 18 的两套独立前向使用相同拓扑和解码器。`simple` 仅将 NN 重量化乘法从 FP64 改为 FP32。

COCO 全量有 349 张猫狗正图和 420 个非 crowd 实例。P/R 按鸟、飞机和 `animal` 三类统一选工作点，完整逐类结果见 `metrics-sony-1m2-{simulator,simple}.json` 和 `metrics-yolo-1m2-intersection.json`。上节鸟／飞机表只包含两类；加第三类会改变 F1 工作点，Ultralytics 对同分候选的排序也可能使 AP 略有变化。

| 路径 | 候选框数 | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|
| 1M2 槽 18 `simulator` | 4,804 | 1.28% | 11.90% | 0.68% | 0.23% |
| 1M2 槽 18 `simple` | 4,805 | 1.30% | 12.14% | 0.70% | 0.25% |
| YOLO26n（猫狗合类） | 5,182 | 84.58% | 78.81% | 83.83% | 69.03% |

两版在 3,286 张图上的预测记录不同（其中 932 张框坐标或是否出框不同，21 张框数不同），但 AP 差距小。当前解码器接近每张都出一框；这是单框探针的性能，不是原厂动物检测的性能。

1M2 沿用交付的 `0.02` 热图阈值，合并所有入选峰，至多输出一框；按框内最大热图强度除以该输出槽的可表示上限排序，分数不是校准概率。

## 人体姿态

YOLO26n-Pose 在 5,000 张、6,352 个有关键点的非 crowd 人体上：P **82.82%**，R **74.61%**，OKS AP50 **80.98%**，AP75 **61.27%**，AP50–95 **56.42%**。逐关节误差、匹配数和 PCK 见 `metrics-yolo-pose.json`。

两台 Sony 机型均未交付可供本报告评测的独立人体前向，因此这里只列 YOLO 基线，不做 Sony 人体姿态对比。

## 评分口径

- 用本机 Ultralytics `8.4.26` 的 `BaseValidator.match_predictions`、`box_iou`、`kpt_iou`、`ap_per_class`，阈值 0.50/0.55/…/0.95。检测按同类 IoU 做一对一匹配；姿态按 keypoint OKS 做匹配，面积按 Ultralytics 的 `0.53 × GT bbox 面积`，沿用 COCO sigma。非 crowd、正面积；姿态另排除 `num_keypoints=0`。直接读 COCO JSON 使用原像素坐标，没有转换成 YOLO 标签文件。本表不是 `pycocotools` 的官方排行榜分数。
- 误差另以框 IoU ≥ 0.50 对人做最大 IoU 一对一配对，计算有预测的标注点的像素误差和以 `sqrt(GT bbox 面积)` 归一的误差；分别记录低阈值候选及各模型 F1 工作点上的匹配数。`PCK@0.10` 分母为全部相应的已标注点，漏人、漏点不算正确。用框配对只用于误差统计，绝不替代 OKS AP。
- YOLO 使用目录提供的本地权重、640×640、FP32、batch 16、`rect=False`、`conf=0.001`、`max_det=300`。检测只保留鸟、飞机、猫、狗；姿态只保留 person。用 `.txt` 文件清单流式分批；Python 路径列表在该版本被当成一整个内存批次，不能用它传 5,000 张。
- Sony 鸟／飞机用 RGB、320×320、双线性 letterbox、填充值 128、精确反投影。沿用交付的 40×40 鸟、20×20 飞机探针及其已知候选锚，类内索引 1 softmax、NMS IoU 0.35、最多 300 框/类；置信度下限由例图的 0.85 降到 0.001 以形成 PR 曲线。未假设未知输出尺度及锚框语义。
- 动物将猫、狗 GT 和 YOLO 类别合并到同一 `animal`，保持不分物种，按各表所用的同一批图片比较。昆虫不在 COCO 交集内；尚无能在此交付中评估的可靠汽车／火车解码器，不按零分处理。

## 复现

从项目根目录执行。生成数据和输出均在 `runs/coco2017-eval`，不修改上游推理文件。

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
$r = 'runs/coco2017-eval'
$source = 'C:/Users/HeMu/Desktop/RSCtrlApp-2nd/3rdparty/ultralytics-models'
New-Item -ItemType Directory -Force $r
Copy-Item -LiteralPath "$source/yolo26n.pt","$source/yolo26n-pose.pt" -Destination $r
git archive --format=tar --output="$r/sony-release.tar" 95c9834 tools parsers
& $py -B tools/coco_eval_infer.py prepare
& $py -B tools/coco_eval_infer.py yolo detect --batch 16
& $py -B tools/coco_eval_infer.py yolo pose --batch 16
foreach ($mode in 'simulator','simple') {
    foreach ($slot in '08','10') {
        & $py -B tools/coco_eval_infer.py sony-detect $mode $slot
    }
    & $py -B tools/coco_eval_metrics.py detect "$r/metrics-sony-$mode.json" `
        "$r/sony-08-$mode.jsonl" "$r/sony-10-$mode.jsonl"
}
& $py -B tools/coco_eval_metrics.py detect "$r/metrics-yolo-bird-airplane.json" "$r/yolo-detect.jsonl"
& $py -B tools/coco_eval_metrics.py detect "$r/metrics-yolo-detect.json" "$r/yolo-detect.jsonl" --classes bird airplane cat dog
& $py -B tools/coco_eval_metrics.py pose "$r/metrics-yolo-pose.json" "$r/yolo-pose.jsonl"
& $py -B -m unittest discover -s tools -p test_coco_eval.py -v
```

1M2 动物全量使用已固定的 `runs/coco2017-eval/sony_slot18_candidate.py` 和 `runs/portable-release/pose-parameters/sony-cat-1m2-weights.pt`；运行器还会检查同一工作目录下已有的 `slot18-reference-check.json` 是否匹配设备、版本、源码和权重。候选源和权重哈希分别为 `66809988312873dddc265522731ea1079460d4e376d77760d18c4c7a81d29576` 与 `c73a628ef286ad8c72ddd940de6a64769b0e2492c0027542b1fb30058fa6a43b`。下面的全量推理命令不调用模拟器。

```powershell
foreach ($mode in 'simulator','simple') {
    & $py -B tools/coco_eval_slot18.py $mode
    & $py -B tools/coco_eval_metrics.py detect "$r/metrics-sony-1m2-$mode.json" `
        "$r/sony-08-$mode.jsonl" "$r/sony-10-$mode.jsonl" "$r/sony-18-$mode.jsonl" `
        --classes bird airplane animal
}
& $py -B tools/coco_eval_metrics.py detect "$r/metrics-yolo-1m2-intersection.json" `
    "$r/yolo-detect.jsonl" --classes bird airplane animal
```

完整 val 图和标注由本地 `datasets/val2017.zip` / `annotations_trainval2017.zip` 解包。权重文件 SHA-256（源和副本一致）：

| 文件 | SHA-256 |
|---|---|
| `yolo26n.pt` | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `yolo26n-pose.pt` | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |

环境：Windows、RTX 5060 Ti、PyTorch `2.8.0+cu129`、Ultralytics `8.4.26`。

## 局限

这里评估 **源自 Sony NBG 的独立前向 + 主机实验预处理与解码**，不是 Sony 机身端到端对焦或跟踪。机内槽选择、分辨率／ROI、完整后处理、原生锚框和所有尺度、置信度、NMS、动物和人体的原厂分组未恢复。不同网络输入分辨率、任务训练目标与输出不对齐，不能把差距归因给原始权重或舍入中的某一个因素。缺少独立实现的部分保留为未评测，不以其他结果顶替。
