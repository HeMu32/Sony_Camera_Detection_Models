# Sony 槽 10：无 FP64 的独立 GPU 推理

A7R V 和 A1 II 固件中的 `slot10.nb` 字节相同，SHA-256 为
`22a46a633b90f780026f57b21b2d34fa1dbd7b49237276a415eb109537da39be`。
两份独立 CPU 跟踪（飞机、天际线）各覆盖 74 条 NN、109 条 TP 指令，
重建出的整个权重包一致（275 个字段，3,775,518 字节）。
包的 SHA-256 为 `f4b40c3a237f9bd5c6ba15e4cb90873166c3febfefeed88ebc11b2fb06a1440b`。

## 运行

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
$d = 'debug/cpu_raw/r5_s10'
& $py tools/slot10_cuda.py pack nab_graphs/r5_slots/slot10.nb "$d/Flight.nbc" "$d/Flight.nbt" "$d/rebuilt.npz"
& $py tools/slot10_cuda.py infer "$d/sony-slot10-cuda.npz" assets_aux/Flight_DSC02983.jpg "$d/DemoFlight"
& $py tools/verify_slot10_cuda.py "$d/sony-slot10-cuda.npz" debug/cpu_inputs/320x320_rgb/Flight_DSC02983.raw "$d/Flight" --tp-trace "$d/Flight.nbt"
& $py tools/render_anchor_probes.py assets_aux/Flight_DSC02983.jpg debug/visual_checks/demo_gpu_plane.png --manifest "$d/DemoFlight.manifest.json" --prediction 10 "$d/DemoFlight" "$d/DemoFlight.log"
```

`infer` 只读取权重包和输入，不需要 CPU 模拟器、CPU 参考、机身或 VIP9000。
它用 FP32 GEMM（禁用 TF32）对整数操作数精确累加，FP32 FMA 修正半整数边界，
原始 TP 参数重现 SE8M15 舍入。RTX 5060 Ti 上预热 8 次，以 5 组各 20 次
的 CUDA Event 计时中位数为 **23.2 ms/张**。不含预处理、权重加载、回读、写盘和画框。

## 输出与核对

- 尺度为 `80、40、20、10、5、3、1`。输出 0–6 每网格位置是 6 组 3 类分数，
  输出 7–13 是 6 组 4 个框回归量，均为平面字节；输出 14–20 是 6 通道辅助值。
  辅助输出声明的形状是 `g×g×6×1`，实际字节按 HWC 交错；`20×20` 的非零值可区分这两种布局。
  模型返回分类与回归张量，不直接返回像素框或检测点。
- 飞机、天际线和混合图各 21 份最终输出（每组 409,680 字节）均与 CPU 参考逐字节一致。
  飞机和天际线另核验每个输入的 50 个分类／回归 TP 切片：输入、算术和最终写入均一致。
  `80×80` 层的四个切片按列覆盖整层，其余较大层按行或通道分片。
- 目视检查：`debug/visual_checks/gpu_slot10_plane.png` 的框落在飞机上但略切机头；
  `gpu_both_empty.png` 无框；`gpu_both_mixed.png` 中 GPU 槽 8 和 10 分别框中一只鸟和飞机，
  其他鸟未检出。实验框仍用推定的先验、阈值和 NMS，不能代表 Sony 机内后处理。
  合成图是主机分别运行两个原始模型，不是 A1 II 的 Auto 调度。
