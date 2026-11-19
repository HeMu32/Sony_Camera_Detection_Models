# CUDA 舍入 A/B 验证

A7R V 与 A1 II 的鸟槽 8、飞机槽 10 均提供两种无 FP64 的前向。
两者共用原始权重、FP32 整数卷积、图拓扑、张量布局、输入和实验框解码器，
只改变重量化。`simulator` 是默认值：保留 NN 的 FMA 半整数修正，以及 TP
的两次 SE8M15 中间舍入。`simple` 用直接 FP32 乘法后整数舍入，TP 缩放加
零点后直接取整。`simple` 并不是一个声称模拟器逐位等价的新模型。

## 实测

每槽 12 个输入：五张用户样图、同一帧的鸟／飞机合成图、六张针对该槽的
镜像／平移／缩放探针。统计仅包含实际写出的输出，槽 8 的三个空接口槽不计。

| 模型 | 变化字节比例 | 单字节最大绝对差 | 框数（默认 / 简化） | 前向中位数（默认 / 简化） |
|---|---:|---:|---:|---:|
| 鸟，槽 8 | 0.416% | 4 | 10 / 10 | 20.909 / 21.284 ms |
| 飞机，槽 10 | 3.640% | 38 | 7 / 7 | 24.666 / 25.465 ms |

17 对保留框的坐标和分数全部完全相同（IoU 1.0，任何边的位移均为 0）。鸟正图
两框选中两只鸟，其余仍有漏检；飞机框选中机身但略切机头；混合图两槽分别框中
一只鸟和一架飞机；天际线两槽均无框。最大 38 的差异只出现在飞机辅助输出；
例如鸟图送入飞机模型时，输出 18、19 各有一个值从 187 变成 225。当前
实验解码器不用辅助输出，不能外推到真实机内后处理或未测试输入。

RTX 5060 Ti，每种模式预热 8 次、6 组各 20 次，交替先测哪种模式；
计时是已在 GPU 上的第一个样本的前向，不含解码或回读。波动范围相交，
没有测到简化舍入稳定提速。默认模式也重跑了鸟正图／空景／混合图以及
飞机正图／空景／混合图，六次各 21 个接口输出仍对上既有 CPU 模拟器参考。
这不是任何一条路径与实机逐位一致的结论。

## 使用与证据

以飞机模型为例，在已有 `infer` 命令末尾加 `--rounding simple` 切换模式，
原始张量输出日志会注明模式。A/B 可直接使用已准备好的平面 RGB：

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
$d = 'debug/cpu_raw/r5_s10'
& $py tools/compare_nbg_cuda_rounding.py 10 "$d/sony-slot10-cuda.npz" debug/cpu_inputs/320x320_rgb/Flight_DSC02983.raw debug/cpu_inputs/320x320_rgb/Empty_DSC03284.raw --report debug/rounding-repro.json --preview-dir debug/rounding-repro
```

完整记录：`debug/cpu_raw/rounding-slot08.json`、`rounding-slot10.json`
（逐张量差异、框和逐组计时）。视觉图在 `debug/visual_checks/rounding-ab/`，
两槽简化模式合成框另见 `debug/visual_checks/gpu_both_mixed_simple.png`。
