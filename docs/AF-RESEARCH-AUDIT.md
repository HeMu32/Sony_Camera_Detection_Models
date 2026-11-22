# Sony 相机视觉对焦研究：会话与交付物审计

审计日期：2026-11-19

审计对象：

- 会话 `01a1f231-42dd-76e1-a253-efc1c57b4a35`，标题“研究 Sony 相机视觉对焦算法”。
- 会话 `01a1f232-ab66-72c3-bc0d-bc95bff41d01`，标题“研究 Sony 对焦检测流程”。
- 两个会话生成的 AF、CP、XHD、A7 IV、COCO 和发布验证文档。

本审计不连接机身、不执行 VIP9000 硬件、不提交固件作业。结论只覆盖本地镜像、已有输出、主机侧原指令回放和现有测试。没有对整张 CP 镜像做“所有函数、所有路径”的形式化穷举；因此下文说“完整”时，均指所审计的交付物和对应判定链，而不是整个固件宇宙。

## 0. 结论摘要

总体判断：

1. **高层结论基本成立。** 两个会话没有把单个最高分直接当作镜头合焦；视觉检测、候选接受、跨帧状态、AF 测量和镜头控制被分开描述，这个分层是正确的。
2. **数值复现基本成立。** COCO 指标、候选上限、XHD 结果转换、A7 IV FDN/PDN 发布默认值和发布导出验证均能复现。
3. **存在一个明确的静态分支错漏。** `judgmentFrameAndXHD` 的 `apply_type=3` 被现有验收文档漏掉；该分支既不执行框路径，也不执行分数路径，并清除 `+0x755`。现有测试也没有覆盖它。
4. **存在一个明确的评测措辞错误。** `COCO2017-EVAL.md` 说两台 Sony 机型“缺少独立人体前向”，但本地实际有 9 张图的 SDK 人体 pilot 输出，以及 20/5000 张的 partial `sdk-full` 输出。正确说法应是“没有全量 Sony 人体评测”，而不是“没有独立前向”。
5. **XHD 配置字节存在未解决的读法冲突。** 发布字节和 XHD 命令路径可按两个独立字节使用 `+0x186`、`+0x187`，但另一个配置序列化分支把两者读成 big-endian `s16`。原文档没有把这个条件差异写清。
6. **还有若干重要执行分支没有闭合。** 主要包括 `FUN_006ecd36` 的 `+0x75a==0` 直接返回、`FUN_006ece30` 的竞争/重算分支、`XedResultInvalidater` 的内部类型分支、XHD 描述符到 SA 成员的映射，以及 A1 II 的 AutoRecog 到槽位运行时映射。
7. **没有发现伪造数值或把合成数据写成实机结果的迹象。** 主要问题是分支遗漏、措辞过宽、旧笔记未更新和测试覆盖不足。

## 1. 证据等级

| 等级 | 含义 |
|---|---|
| 已回放 | 执行了原 Thumb 片段、原虚表或原算法代码，并检查返回字节/状态 |
| 已复现 | 对已有 JSON、日志或测试重新运行，得到相同数值 |
| 静态 | 从镜像字节、调用点或字符串定位到结构，但未执行完整上下文 |
| 公开 | Sony 手册或公开资料描述的用户行为 |
| 未闭合 | 只找到名称、字段或局部调用，没有证明生产者和最终消费者 |

本审计中的“正确”不等于“实机验证”。除非特别说明，验证都是离线、主机侧或固定模拟器结果。

下文路径均相对于当前工作区 `Sony_Camera_Detection_Models`；涉及 7M4、R5、A1 II 的镜像路径按原研究文档不变。

## 2. 实际错漏

### 2.1 [P1] `apply_type=3` 被验收文档漏掉

文件：`docs/AF-VISION-ACCEPTANCE.md:14`

原文：

> `apply_type` 为 1/2 时检查部位/候选并可置 `+0x750`；为 0/2 时经额外条件检查调用 `+0x6ece30`。

原 CP `judgmentFrameAndXHD` 的实际分支是：

| `apply_type` | 框路径 | 分数路径 | 结果 |
|---:|---|---|---|
| 0 | 跳过 | 执行 `+0x6ece30` | score-only |
| 1 | 执行 | 跳过，并在 `+0x1c3fa8` 清 `+0x755` | box-only |
| 2 | 执行 | 执行 | box + score |
| 3 | 跳过 | 跳过，并在 `+0x1c3fa8` 清 `+0x755` | 两者都不执行 |

依据：

- `0x1c3e40` 读 `apply_type`。
- `0x1c3e42..0x1c3e48` 对 `apply_type - 1` 做 `cmp #1` / `bhi`。
- `0x1c3f44..0x1c3f4a` 用 `(apply_type | 2) == 2` 选择分数路径。
- `0x1c3fa8..0x1c3fb0` 是两者都不执行时清 `+0x755` 的分支。

结论：原文不是数值错误，但分支描述不完整。应改为：

> `0=score-only，1=box-only，2=box+score，3=两者皆不执行并清有效位`。

对应测试缺口：`tools/test_sony_cp_detection_gate.py` 只测试了 `+0x6ece30` 门限和 `+0x6f5892` 计数，没有回放 `judgmentFrameAndXHD` 的 `apply_type=3` 分支。

### 2.2 [P1] COCO 人体“缺少独立前向”表述过强

文件：`docs/COCO2017-EVAL.md:9-16,57`

实际存在的本地结果：

- `runs/coco2017-eval/sdk-pilot/sony-human.jsonl`：9 张图的预测。
- `runs/coco2017-eval/sdk-pilot/metrics-sony-human.json`：`image_count=9`，6 个正图、10 个 GT 人体、6 个预测。
- `runs/coco2017-eval/sdk-full/out/human-*.done.json`：20 个已完成人体作业，但 `job.json` 规划了 5000 个 human 作业。

因此正确边界是：

> 没有 5000 张全量 Sony 人体评测；有 9 张 SDK pilot 输出和 20 张 partial SDK 输出。现有 9 张指标是实验性经验映射，不是全量成绩。

现有 9 图指标可复现为：P 63.29%，R 30.00%，OKS AP50 42.25%，AP75 42.25%，AP50-95 27.50%。该结果使用了 16 个通道的经验映射，`right_ear` 未被接受，不能称为固件通道名称已确认。

### 2.3 [P1] XHD 配置字节的读法冲突没有写清

文件：`docs/CP-XHD-NMS.md:23-24,99-101`

当前文档把 `+0x186` 写成 `bgfg_thr`，把 `+0x187` 写成短边过滤字节。实际有两类读取方式：

- XHD 命令打包路径可按两个独立字节使用：
  - 发布记录 `0x60013` 的 `+0x186/+0x187` 为 `0x81/0x09`；
  - NMS 参数块中回放为 `bgfg=129`、`minimum_side=9`。
- A7R V 的另一条配置序列化分支把 `+0x186/+0x187` 读成 big-endian `s16`，高字节在 `+0x186`，低字节在 `+0x187`。

这不是“哪一个名字绝对正确”，而是不同消费者、不同分支对同一对字节的解释不同。文档应明确写“两套读法均已在各自路径回放，运行时选择哪条分支仍未统一证明”。

### 2.4 [P1] `FUN_006ecd36` 的零结果直接返回分支没有记录

原 CP 函数 `FUN_006ecd36`（A7R V 文件偏移 `0x6ecd36`）开头先写：

```text
score = 0x7f7fffff
valid = 0
if (*(u8 *)(object + 0x75a) == 0)
    return
```

只有 `+0x75a` 非零时才构造 5 条记录并遍历 8 个 `0x18` 字节框。`AF-VISION-ACCEPTANCE.md` 和 `AF-DETECTION-DECISION.md` 只描述了下游的几何门/分数门，没有把这个“没有结果就直接保留无效状态”的分支单列出来。

结论：这是遗漏的执行分支，不是文档中的数值错误。

### 2.5 [P1] `FUN_006ece30` 的竞争和重算分支只被概括

`AF-DETECTION-DECISION.md` 已正确写出候选分数严格小于阈值时清 `+0x755`，相等不清。但同一函数还有：

- `*param_6 != 0` 时直接返回。
- `*param_5 < candidate` 时进入“低于前一名”的日志分支。
- 记录 `+0x168` 非零时调用 `FUN_001c4170` 重算。
- `FUN_001c4170` 的模式 0 走 `DIST`，其他走 `IOU`。

这些分支对“为什么一个局部阈值不足以解释最终接受状态”很关键。现有高层文档说“另有竞争分支”，但没有列出完整的条件顺序，也没有回放后段几何/合并公式。

### 2.6 [P1] 7M4 的 Staple 证据范围应进一步限定

文件：`docs/AF-VISUAL-TRACKING.md:45`

重新核对三份 CP 镜像的结果：

| 镜像 | `[Staple]FrameCalc` | `[Staple]update` | `HumanStaple` | `GetStapleTrackResult` | `histMergeFactor` / `clipHog` / `color_respMergeMethod` |
|---|---:|---:|---:|---:|---:|
| A7 IV | 0 | 0 | 0 | 0 | 有 |
| A7R V | 有 | 有 | 有 | 有 | 有 |
| A1 II | 有 | 有 | 有 | 有 | 有 |

因此“三份 CP 镜像都有 Staple 与前/背景直方图、HOG、响应合并、AF 区域和失焦字段”可以保留，因为 A7 IV 确有 `histMergeFactor`、`clipHog`、`respMergeMethod`、`af_area_num`、`defocus` 等字段。
但“A7R V 还有 `FrameCalc`/`update`”容易被读成 A1 II 没有，实际 A1 II 也有。更准确的表述是：

> A7R V 和 A1 II 明确有 `[Staple]FrameCalc`/`[Staple]update`；A7 IV 只确认了 Staple 驱动和参数/响应字段，没有确认同名帧计算/更新路径。

### 2.7 [P2] A1 II XED 偏移的措辞可更精确

文件：`docs/AF-DETECTION-DECISION.md:11`

`+0x1ef29c/+0x1ef2a4` 是 `movw r3,#0xe3a6` / `movt r3,#0x104` 指令的位置，形成 `0x104e3a6`；字符串本身在文件偏移 `0xf463a6`。文档写“引用”并没有把指令位置误当成字符串，但应说明这是“构造字符串地址的指令位置”，不是字符串起点。

### 2.8 [P2] 回归计数已过期

会话早期写过“23 项”或“28 项相关回归通过”。这些是当时的阶段性计数，不是当前完整计数。

本次重新执行：

- `test_sony_cp_*.py`：14/14。
- `test_a7iv_fdn_cp_*.py`：15/15。
- 合计 29/29。
- `test_sa_*.py`：73/73。
- `parsers`：11/11。
- `test_coco_eval.py`：6/6。
- `test_slot08_tiled_probe.py`：4/4。

结论：文档若继续引用 23/28，应标明“会话当时计数”；当前审计口径应写 29 项 CP 回归，另列 SA 和 parsers。

### 2.9 [P2] `INFERENCE-STATUS.md` 是旧状态

文件：`docs/INFERENCE-STATUS.md:21`

该页写“当前不能在主机侧正确推理 dic 模型”，并列举 FDN/Pdn/Scd/Bed 构造函数尚未定位。该结论是 2026-11-18 的阶段性状态，早于后续 `A7IV-SA-FORWARD.md` 和 `A7IV-WBI2-DETECTOR-EVIDENCE.md`。

后续已定位 FDN/PDN 参数管理器和发布默认值，并回放了 CP 请求、结果消费者、严格 `>` 门限和局部 SA 执行段。但完整真实图像 DSP 前向仍未跑通。因此应把旧页标为“历史状态”，不能与后续文档混为同一当前结论。

## 3. 需要保留的未闭合项

以下项目在原文档中大多已正确标为未闭合，审计不把它们误判为“文档错误”：

### 3.1 原始张量到 XHD 记录的原厂生产链

现有证据能证明：

- CP `AdamXhdDriver` 构造短/长描述符。
- `(0x10,3,BLOCK+0x4bc)` 描述符被写出。
- XHD 结果最多 50 条，`51` 进入错误路径。
- CP 将 Q14 坐标、类型、部位和 Q15 分数转换成整帧框。
- 小框可清分但保留记录。

没有证明：

- NBG 分类/回归张量如何被原厂 NMS 作业消费。
- 锚框、先验、NMS 顺序和每个门限在作业内部怎样组合。
- `engine=3` 对应哪个 SA 成员。
- 描述符的第一个真正派发消费者。

这是整条视觉分支中最大的缺口。文档已承认，保留。

### 3.2 A1 II 的空间门

A7R V 的 `judgmentFrameAndXHD` 下游空间门已经用原指令回放。A1 II 没有同等的空间门回放证据，现有文档明确限定为 A7R V，保留。

### 3.3 A1 II 的 XED 三路峰

A1 II 的 `peak_num eyeL/eyeR/nose` 字符串位于 `0xf463b9`；`+0x1ef29c/+0x1ef2a4` 是形成 `0x104e3a6` 的指令位置。`[eyeR]` 和 `[nose]` 字符串分别在 `0x1ef36c`、`0x1ef3a0`。现有文档已说明 A7R V 的三路峰消费器有原指令回放，但 A1 II 没有同等的三路循环回放。保留为未闭合项。

### 3.4 `XedResultInvalidater`

`+0x1cdc80` 路径通过虚表 `+0x184/+0x188/+0x18c` 取计数/阈值，再按内部类型 `0..4` 读 `+0x69`、`+0x7d` 等字节。当前能确认它不是单纯的分数阈值，但未恢复每个内部类型的业务名称、门限来源和最终 AF 含义。现有文档已把它列为待查，保留。

### 3.5 7M4 的真实图像 FDN/PDN 前向

已有证据包括：

- CP 请求打包、方向/搜索窗/地址字段。
- 发布记录默认门限。
- FDN 按方向选门限，PDN 左右眼独立比较，均为严格 `>`。
- SA 局部执行段和结果写回。

没有证据包括：

- 真实图像经 FDN/PDN 得到数值结果。
- 最终 AF 选点。
- 完整 DSP 前向。

现有 `A7IV-SA-FORWARD.md` 已明确区分“执行段”和“端到端”，保留。

## 4. 抽象算法分支审计

| 分支 | 已证内容 | 状态 |
|---|---|---|
| 常规 AF，无识别结果 | 按 AF 区域/测焦信号工作，不经过视觉检测候选 | 公开与固件分层支持 |
| 主体识别 AF | 视觉候选参与主体/部位选择 | 高层成立，具体仲裁未闭合 |
| 专用识别 AF | 可在全画面寻找识别主体 | 公开手册支持 |
| 已识别眼/脸/头/身体 | 部位可回退 | 公开行为 + 固件部位判定线索 |
| 跟踪未识别主体 | Staple/普通物体跟踪可先锁定，识别后接管 | 公开行为 + Staple 线索；完整状态机未闭合 |
| 跟踪丢失后回退 | 有旧锁定状态和丢失判定 | 局部回放成立，帧节拍未闭合 |
| A1 II Auto 识别类别 | 官方支持自动选择类别 | 固件有 AutoRecog 字符串；类别到槽位运行时映射未闭合 |
| A1 II 搜索模式切换 | `SearchModeSwitcher` 读取时间/ZPD 状态 | 状态机局部成立，ROI/检测器映射未闭合 |
| 登记人脸优先 | `Face Memory` 与注册优先设置 | 公开行为 + 旧机属性路径；匹配器未闭合 |
| 相位/失焦辅助选主体 | A1 II 有 `PriorityDefocusSubjectSelector` 线索 | 功能路径成立，完整公式未闭合 |
| A7 IV FDN/PDN | 方向门限和双眼门限 | CP 结果回放成立，真实图像前向未闭合 |
| XHD/NMS | 候选记录转换与门限接口 | 结果侧成立，生产侧未闭合 |
| XED 局部峰 | A7R V 三路计数回放 | 生产热图与最终主体绑定未闭合 |

结论：两个会话的高层抽象分支没有把“未识别主体跟踪”“Auto 类别选择”“登记人脸匹配”和“最终 AF 仲裁”混成同一条已证调用链，这一点是正确的。需要补的是分支的边界和顺序，不是推翻整个框图。

## 5. 静态代码执行分支审计

| 函数/路径 | 已确认分支 | 文档状态 |
|---|---|---|
| `judgmentInsectAndAirplane` | `prio_part==3 -> class 1`，否则 `class 3`；共用昆虫/飞机 | 基本正确 |
| `judgmentTrain` / `judgmentCar` | 按 `prio_part` 选不同 part 列表；训练/汽车各可走两条 part | 文档未全部列出，保留 |
| `judgmentFrameAndXHD` | `apply_type 0/1/2/3`；0 score，1 box，2 both，3 neither | **文档漏 3，测试也漏 3** |
| `FUN_006ecd36` | `+0x75a==0` 直接返回；否则遍历 5 条记录 × 8 个框 | **零结果分支未在验收文档单列** |
| `FUN_006ece30` | 分数严格低于阈值清 `+0x755`；相等不清；竞争/前一名/重算路径存在 | 部分正确，后段未回放 |
| `FUN_001c4170` | mode 0 `DIST`，否则 `IOU` | 已记录，公式未闭合 |
| `FUN_001c5b60` | 遍历 `0x148` 记录和 `0x18` 框 | 已记录，几何/合并公式未全回放 |
| `XedResultInvalidater` | 内部类型 0..4，读不同计数/阈值字段 | 未完整枚举 |
| XHD 结果转换 | 0..50 接受，51 拒绝；Q14 坐标；Q15 分数；类型码 >8 归 8 | 已回放，基本正确 |
| XHD/NMS 描述符 | 短表 1 条、长表 3 条；`engine=3` 选择器 | 已回放，消费者未闭合 |
| A7 IV FDN/PDN | 严格 `score > threshold`；方向选门限 | 已回放，真实图像前向未闭合 |

补充说明：

- `judgmentFrameAndXHD` 的 `apply_type=1` 还会清 `+0x755`，因为分数路径被跳过；`apply_type=3` 同样清。文档只提“可置 `+0x750`”，没有把两个有效位的分离写出来。
- `FUN_006ecd36` 的 `+0x75a` 是候选遍历使能/状态门，不等于 NBG 类别索引，也不等于主体有效位。
- `FUN_006ece30` 的 `param_5`、`param_6` 和 `+0x168` 分支不应被概括为“单一阈值”。

## 6. 交付物正确性核对

| 交付物 | 审计结果 | 说明 |
|---|---|---|
| `AF-VISION-PIPELINE.md` | 基本正确，需限定 | 高层分层和“原厂候选生产未闭合”成立 |
| `AF-VISUAL-TRACKING.md` | 基本正确，需限定 | 三机 Staple 字段不等价，A1 II 也有 `FrameCalc/update` |
| `AF-DETECTION-DECISION.md` | 基本正确，有遗漏 | 漏 `FUN_006ecd36` 零结果分支和 `apply_type=3` 的完整上下文 |
| `AF-VISION-ACCEPTANCE.md` | **有明确遗漏** | 漏 `apply_type=3`，并缺对应测试 |
| `CP-XHD-NMS.md` | 数据正确，解释有冲突 | 两套字节读法未统一，文档应写条件性 |
| `CP-XHD-RESULT.md` | 正确 | 0/50/51、Q14/Q15、旋转/裁剪/短边清分均复现 |
| `CP-CONFIDENCE.md` | 正确 | 发布值和 getter 已回放，未声称运行时覆盖 |
| `CP-REGION-SCALE.md` | 正确 | 选择器、8 个 u16 字段和回退值已回放 |
| `COCO2017-EVAL.md` | 数值正确，人体覆盖措辞需改 | 指标全部复现；人体不是全量评测 |
| `COCO-KEYPOINTS-ASSESSMENT.md` | 正确 | 明确“可导出 COCO 形状但不等同官方 AP” |
| `A7IV-WBI2-DETECTOR-EVIDENCE.md` | 基本正确 | 发布默认值和 CP 消费路径成立，运行时覆盖未证 |
| `A7IV-SA-FORWARD.md` | 基本正确 | 局部 SA 执行段成立，真实图像 DSP 前向未闭合 |
| `INFERENCE-STATUS.md` | 已过期 | 早期结论，不能作为当前总状态 |
| `debug/deepseek_anchor/AIRPLANE-NATIVE-SLOTS.md` | 证据较强，需限定 | 已明确 category 7 到 slot 9 的运行时绑定未证 |

## 7. 数值与产物复现

### 7.1 COCO 指标

重新读取 `runs/coco2017-eval/metrics-*.json`：

| 路径 | Precision | Recall | AP50 | AP75 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| Sony 鸟 `simulator` | 14.78 | 5.62 | 3.26 | 1.57 | 1.69 |
| Sony 鸟 `simple` | 14.78 | 5.62 | 3.48 | 1.83 | 1.90 |
| Sony 飞机 `simulator` | 18.24 | 9.79 | 8.90 | 7.15 | 6.25 |
| Sony 飞机 `simple` | 18.24 | 9.79 | 8.90 | 7.15 | 6.21 |
| 1M2 动物 `simulator` | 1.28 | 11.90 | 0.68 | 0.11 | 0.23 |
| 1M2 动物 `simple` | 1.30 | 12.14 | 0.70 | 0.12 | 0.25 |
| YOLO 猫 | 86.58 | 83.17 | 89.12 | 75.87 | 72.51 |
| YOLO 狗 | 79.07 | 72.02 | 77.88 | 71.65 | 65.20 |
| YOLO pose | 82.82 | 74.61 | 80.98 | 61.27 | 56.42 |

结论：文档中的主要数值可复现。YOLO 鸟/飞机表和 `metrics-yolo-detect.json` 一致。

### 7.2 候选上限

重新执行 `tools/coco_af_candidate_ceiling.py`：

| 分支 | 正图有候选 | Top-1 命中 | Oracle 命中 | Oracle 实例召回 | 负图有候选 |
|---|---:|---:|---:|---:|---:|
| 鸟 | 88/125 | 28/125 | 29/125 | 46/427 | 1747 |
| 飞机 | 32/97 | 22/97 | 24/97 | 24/143 | 492 |
| 动物 | 345/349 | 56/349 | 56/349 | 56/420 | 4459 |

结论：文档的“单改排序不够”判断成立。

### 7.3 发布和导出

- `verify_two_camera_release.py`：槽 8/10 的正图和空景共 4 组，21 个输出逐字节一致。
- `verify_sony_portable.py`：槽 1/8/10/12/18 的 ONNX、TorchScript、CUDA 参照和 SDK 参照均 0 差异字节。
- `verification-summary.json`：ONNX 和 TorchScript 各 290 个张量、5,166,750 字节，0 差异。
- `slot18-ab-check.json`：5000 张，3286 条预测记录变化，932 张框坐标/是否出框变化，21 张框数变化，923,093 个不同字节，最大量化码误差 39。

这些结果支持“导出和可移植前向一致”，不支持“已经等于原厂完整相机流水线”。

## 8. 仍未闭合的分支

1. NBG 张量到 XHD 12 字节记录的 NMS/解码生产者。
2. `(0x10,3,+0x4bc)` 的第一个真正派发消费者和 `engine=3` 到 SA 成员的映射。
3. A1 II 是否具有与 A7R V 等价的空间覆盖门；目前没有同等回放。
4. A1 II 的 XED 三路峰消费器；目前只有字符串/地址线索和 A7R V 回放。
5. `XedResultInvalidater` 的内部类型到业务阈值、最终有效位的完整映射。
6. A1 II `AutoRecog` 与 `SearchModeSwitcher` 到具体模型槽、ROI 和帧间隔的运行时绑定。
7. 7M4 的 Staple 帧计算/更新是否存在于未检查模块；当前只在 CP 字符串级确认了部分字段。
8. 7M4 真实图像 FDN/PDN 数值前向和最终 AF 选点。
9. 槽 10 和 `ai_model_category=7` 的实际菜单运行时绑定；RSC 只证明配置配对。
10. 人体 5000 张全量 COCO 评测。

## 9. 复现命令

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'

& $py -B -m unittest discover -s tools -p 'test_sony_cp_*.py' -v
& $py -B -m unittest discover -s tools -p 'test_a7iv_fdn_cp_*.py' -v
& $py -B -m unittest discover -s tools -p 'test_sa_*.py' -v
& $py -B -m unittest discover -s parsers -p 'test_*.py' -v
& $py -B -m unittest discover -s tools -p 'test_coco_eval.py' -v
& $py -B -m unittest discover -s tools -p 'test_slot08_tiled_probe.py' -v
& $py -B tools/verify_two_camera_release.py
& $py -B tools/verify_sony_portable.py runs/portable-release --extra-sdk runs/portable-release/detector-sdk
& $py -B tools/coco_af_candidate_ceiling.py runs/coco2017-eval/data/annotations/instances_val2017.json runs/coco2017-eval/sony-08-simulator.jsonl bird bird
& $py -B tools/coco_af_candidate_ceiling.py runs/coco2017-eval/data/annotations/instances_val2017.json runs/coco2017-eval/sony-10-simulator.jsonl airplane airplane
& $py -B tools/coco_af_candidate_ceiling.py runs/coco2017-eval/data/annotations/instances_val2017.json runs/coco2017-eval/sony-18-simulator.jsonl animal cat dog
```

本次实际结果：

- CP 14/14 OK。
- A7 IV FDN/PDN 15/15 OK。
- SA 73/73 OK。
- parsers 11/11 OK。
- COCO eval 6/6 OK。
- tiled probe 4/4 OK。
- 两机发布验证全部 exact。
- 可移植导出验证 0 差异字节。

## 10. 建议修订顺序

1. 修改 `AF-VISION-ACCEPTANCE.md` 的 `apply_type` 表，补 `3=neither` 和 `+0x755` 清除。
2. 修改 `COCO2017-EVAL.md` 的人体措辞，区分“全量评测缺失”和“9 张 pilot 已有”。
3. 修改 `CP-XHD-NMS.md`，把 `+0x186/+0x187` 标为条件性读法冲突。
4. 修改 `AF-VISUAL-TRACKING.md`，把 7M4 的 Staple 证据限定到实际存在的字段。
5. 在 `AF-DETECTION-DECISION.md` 补 `FUN_006ecd36` 零结果分支、`FUN_006ece30` 竞争分支和 `apply_type=3` 的全局上下文。
6. 更新旧 `INFERENCE-STATUS.md` 的日期和状态，或把它明确标为历史页。
7. 新增一个 `apply_type=3` 的原指令回归测试。

## 11. 审计结论

两个会话的总体算法结论成立，主要交付物的数值也能复现。明确需要修正的，是 `apply_type=3` 的静态分支遗漏、COCO 人体“没有前向”的过宽措辞、XHD 字节读法冲突、旧回归计数和旧状态页。

仍属研究边界而非文档错误的，是 NBG 到 XHD 记录的原厂生产链、`engine=3` 到 SA 成员、A1 II AutoRecog 到槽位运行时绑定、A1 II 空间门、XED 完整峰消费器、7M4 真实 DSP 前向和人体全量评测。
这些未闭合项不能用现有局部回放或主机实验替代，也不能据主机侧 softmax、NMS 或单图样例直接推出原厂实现。
