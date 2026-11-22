# A7R V / A1 II：原始 Adam CPU 框解码

把原 NBG 的字节输出交给原固件的 XHD CPU 作业，再按
[CP 结果格式](CP-XHD-RESULT.md) 生成原图 `xyxy`。这条路径不再
使用实验先验、softmax 或宿主 NMS。两机的槽 8、10 已有正例框；
A1 II 的槽 9、11 也完成了网络到原解码器的整帧试跑，但没有出框。
机身如何选搜索窗口、曝光或缩放上游画面，以及 Auto 的多图调度仍未验证。

## 作业与缓冲区

地址为 `adam_app.bin` **文件偏移**。该载荷由 `0x20` 字节头开头，
工具按物理 `0x0fe10000` 和启动时置入 VBAR 的 `0x00e10000`
两个地址映射原字节和显式 RAM，不启动整机。

| 原指令环节 | A7R V | A1 II |
|---|---:|---:|
| 按 8 字节步长遍历描述符 | `0x110bc` | `0x1135c` |
| 初始化 `proc_id=0x10` 的控制器表项 | `0x10d04` | `0x10fa4` |
| 读 `(proc_id, operation, address)` 并经虚表 `+0x18` 调用 | `0x10f4c..0x10ff4` | `0x111ec..0x11294` |
| 控制器操作 `3` 进入 CPU 解码 | `0x18fb0` | `0x1a6ec` |

原 CP 短表是 `(0x10, 3, block+0x4bc)`。Adam 用 `0x10` 查到控制器
`3`，第二个半字则是交给该控制器的操作号 `3`。两者不是一个字段，
也不是 SA 包成员编号。回放执行原控制器静态构造、表初始化以及
描述符查表，停在虚表调用前核对目标，再独立执行方法中的命令分支
与实际 CPU 解码。
传输指针只重定位到本地 RAM，未启动原队列或 NPU。

CPU 方法从参数相对 `+0x00/+0x04` 取分类与回归缓冲区，`+0x08`
是候选写回地址。`+0x54` 指向回归缓冲区后 `0x32030` 字节处的
`0xc80a` 字节辅助数据。离线回放让未知的硬件复制调用返回失败，走固件
自己的 `memcpy` 回退，并逐字节核对复制结果；缓存服务只在已映射
RAM 上省略。分类和回归来自真实模型输出，不给原程序注入候选。

- 网格为 `80/40/20/10/5/3/1`，每格六个候选。
- 分类各层按九类容量推进；槽 8、10 分别装入五类、三类的原字节，
  空余容量显式留空。回归四个量依次绑定该层原张量。
- 辅助缓冲按层拼原字节，**不**按 SDK 日志形状转置。转置负对照
  在所测正图上都漏检。槽 8 的三份 `80×80` 接口输出不由图写出，
  保留其地址空间，原发布配置的跳过字节为 `1`。
- 原机 `TuningParamIP` 从 BK4 第二段读取；本次显式运行
  `(slot, driver, target)=(8,8,3)` 和 `(10,10,4)`，
  并分别核对回写结构及整帧转换。不由这两组实验反推菜单路由。

## 图像核对

| 输入 | 原 Adam CPU 输出 | 视觉结果 |
|---|---|---|
| 鸟 | 两机后处理各两条，`type 1 / part 0`；320 栅格 `xyxy=(121,180,163,223)`、`(1,178,21,224)` | [鸟图](../debug/visual_checks/adam-native-r5-bird.png) 两框确实各选中一只，其余有漏检 |
| 飞机 | 两机各一条，`type 7 / part 1`；`xyxy=(92,144,199,185)` | [A7R V](../debug/visual_checks/adam-native-r5-flight.png)、[A1 II](../debug/visual_checks/adam-native-1m2-flight.png) 框住主机身和尾翼，略切机鼻 |
| 天际线 | 两机、两槽均零条 | [空景](../debug/visual_checks/adam-native-r5-empty.png) 无框 |
| 鸟和飞机拼图一 | A1 II 分别运行两套参数，各一条候选 | [叠图](../debug/visual_checks/adam-native-1m2-mixed.png) 鸟框对位，飞机框横跨画面且包含多只鸟；不通过 |
| 鸟和飞机拼图二 | RTX 5060 Ti 分别推理，接 A1 II 原解码，各一条候选 | [叠图](../debug/visual_checks/adam-native-1m2-mixed02.png) 两框分别选中鸟与飞机；仍是主机并列输出，不是原生 Auto |

A7R V 与 A1 II 的槽 8 图逐字节相同；两次鸟尾部解码均用
同一份已存张量。飞机分别使用两机已存的输出。另以 RTX 5060 Ti
重新运行鸟、飞机原图的 CUDA 模型，接此 CPU 解码，结果记录与
SDK 张量路径逐字节一致；这不是与真实芯片逐字节对照。
人物及猫在此仍使用旧的独立热图工作流。A7 IV 不使用这套 Adam XHD。

另对 A1 II 原始 Auto 管理器做了[区域与驱动回放](../debug/deepseek_dispatch/a1ii-auto-schedule.md)：
其构造器为图 8、10 分别选目标 1、6，而非上述实验默认的 3、4。
用已存 SDK 张量及对应 BK4 配置重跑后，鸟原图仍为两框、飞机
原图仍为一框，框坐标与上表相同；图 8 对天际线仍为零框。
这只核对各自独立的解码路径，未复现同帧 Auto 发布。CLI 可通过
`--tuning-target 1` 或 `--tuning-target 6` 显式复现这组参数。

## A1 II 图 9、11

从原 CP `0x1f04d4` 执行到 NPU 描述符，确认 21 个输出地址与容量：
分类 `0x0ff00000`（每格 54 字节）、回归 `0x0ff80000`（24 字节）、
辅助 `0x0ffb2030`（6 字节），每组七级按容量连续。固定版 x86 SDK
实际返回每格 42 个分类字节，主机按固件容量放入各级，未写容量补零；
这不是对机身空余缓冲区内容的断言。

用 A1 II 原图在本地 Ubuntu 虚拟机跑五张样本，输入 SHA-256 逐一
与 Windows 保存的预处理字节核对，再以 BK4 原参数分别运行图 9／
目标 5、图 11／目标 4。十次原解码均正常返回零候选，飞机叠图也
目视确认无框。图 9 的飞机输出 21 份张量与既有 SDK 捕获完全相同。
两图的分类和回归随输入变化；辅助输出在五图间不变，图 9 全为
`128`，图 11 最后六字节为 `0`、其余为 `128`。仅降低 `bgfg_thr`
到 `128/127/0` 不能使飞机样本出框。结论仅限整帧输入与当前 CPU
解码路径；尚未验证原厂 ROI、辅助数据生成或该路径的类别含义。

对飞机样本，另在 SDK 图验证后、运行前分别用 `0x3c` 与 `0xc3`
预填全部输出张量。图 9、11 各有 614,520 字节，两次运行均逐字节
恢复未预填基准，辅助输出也不例外。对照图 8 在输出 `0/7/14`
仍保留写入的 `0x3c`（各六字节），其余张量与原鸟图一致；图 10
的全部输出与原飞机图一致。因而图 9、11 的固定辅助值不是未被
触碰的 OpenVX 输出初值，但这不区分图中计算、内部填充和拷贝，
也不证明芯片行为。单独强制 `skip_max_decode=1` 的诊断同样没有
使两图的飞机或天际线样本出框；交付参数未被修改。

只读跟踪进一步定位了飞机样本的 `1×1` 辅助输出（SDK 第 20 号，
从零编号）。三图均执行 74 条 NN，第 67 条实际各写出六个零。
图 9 的 TP 第 75 条从该写入地址取数，以模式 `0x80a` 和输出零点
`128` 重量化，独立整数公式与最终六个 `128` 一致。图 11 的该 NN
直接写到外部地址，中间没有 TP，最终是六个零。图 10 的 TP 第
95 条使用查表路径，最终也是六个 `128`。图 10 共 109 条 TP，
其中 25 条带相同查表模式标志；图 9 的 85 条和图 11 的 84 条
均无此标志。该对照没有解释另外六个尺度为何输出固定值。

```bash
bash run_sony_nbg_cpu.sh x86_64_linux build-a1ii-s09 1m2_slots/slot09.nb s09-flight cpu_inputs/320x320_rgb/Flight_DSC02983.raw
NBG_PROBE_OUTPUT_SEED=0x3c bash run_sony_nbg_cpu.sh x86_64_linux build-a1ii-s09 1m2_slots/slot09.nb s09-seed3c cpu_inputs/320x320_rgb/Flight_DSC02983.raw
```

第二组预填改用 `0xc3`；回传输出后运行 `verify_nbg_output_seed.py`
逐张量比较未预填、`0x3c` 和 `0xc3` 三组结果。
NN 单条跟踪可用 `NBG_NN_WRITES_FROM=67 NBG_NN_WRITES_OPS=67`
配合 `NBG_NN_WRITES`；`verify_nbg_aux_trace.py` 核对局部写入、
NN/TP 命令和两次采集的最终张量。

回传 21 份输出后用下述 Windows 命令的参数格式，将 `--slot` 设为
`9` 或 `11`，并改用 A1 II CP/Adam 及 `debug/a1ii_fw/firmware.tar` 的
`0110_backup/SYSIPSX-DSLR/HC/CX89900_ALLLANG.bin`。默认目标已按
Auto 构造器选择。

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
& $py -B tools/probe_sony_adam_xhd.py extracted/r5/cp/cpapp-b.bin extracted/r5/adam/adam_app.bin debug/cpu_raw/r5_s10/Flight --slot 10 --published-backup debug/r5_xhd_fw/firmware.tar --backup-member 0110_backup/SYSIPSX-DSLR/GL/CX89200_ALLLANG.bin --source assets_aux/Flight_DSC02983.jpg --manifest debug/cpu_inputs/320x320_rgb/manifest.json --overlay debug/visual_checks/adam-native-r5-flight.png
& $py -B -m unittest discover -s tools -p 'test_sony_adam_xhd.py' -v
```
