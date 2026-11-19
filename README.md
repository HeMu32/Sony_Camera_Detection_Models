# Sony A7R V / A1 II 主机推理

这是 A7R V（ILCE-7RM5）和 A1 II（ILCE-1M2）的阶段交付。检测网络来自
原始固件，没有使用机身、VIP9000 硬件或替代网络。A7 IV（ILCE-7M4）
按用户要求延迟交付，完成后另行提交；本次提交不包含其未完成实现。

## 验证结果

| 输入 | 主机输出 | 目视核对 |
|---|---|---|
| 人 | 两机槽 1，CPU | [六个框分别落在六个人上](debug/visual_checks/human_pose_boxes.png) |
| 鸟 | 两机槽 8，CPU / CUDA | [两个框分别落在两只鸟上，仍有漏检](debug/visual_checks/gpu_slot08_bird.png) |
| 飞机 | 两机槽 10，CPU / CUDA | [框覆盖机身和尾翼，略切机鼻](debug/visual_checks/gpu_slot10_plane.png) |
| 猫 | A7R V 槽 12、A1 II 槽 18，CPU | [A7R V](debug/visual_checks/r5_cat_pose_boxes.png) 覆盖大部分主体；[A1 II](debug/deepseek_pose_dispatch/vis/a1ii_s18_cat_box_002.png) 只覆盖部分主体 |
| 天际线 | 无目标对照 | [鸟和飞机零框](debug/visual_checks/gpu_both_empty.png)，[人零框](debug/visual_checks/empty_pose_boxes.png) |
| 鸟与飞机同帧 | 槽 8 和 10 独立推理、主机合并 | [框分别选中一只鸟和飞机](debug/visual_checks/gpu_both_mixed.png)，不等于机身 Auto |

鸟和飞机模型返回分类、框回归及辅助张量；人和猫的实验路径用关键点
热图构框。交付图和 JSON 都使用原图像素 `xyxy`，不是只画峰值。
图中的阈值、关联、先验及 NMS 仍有实验成分，不能作为原厂框算法的证明。

## 运行

本机 Python 3.11 `test311` 环境包含 NumPy、Pillow、SciPy，CUDA 路径
另需 CUDA PyTorch 和 CuPy。本机验证使用 RTX 5060 Ti 和 PyTorch
`2.8.0+cu129`。下面命令从项目根目录运行，输出目录须为本次新的目录。

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
$bird = 'debug/cpu_raw/coeff_probe_01/sony-slot08-cuda.npz'
$plane = 'debug/cpu_raw/r5_s10/sony-slot10-cuda.npz'
New-Item -ItemType Directory runs/first
& $py -B tools/slot08_cuda.py infer $bird assets_aux/Bird_DSC02510.jpg runs/first/Bird
& $py -B tools/render_anchor_probes.py assets_aux/Bird_DSC02510.jpg runs/first/bird.png --manifest runs/first/Bird.manifest.json --prediction 08 runs/first/Bird runs/first/Bird.log
& $py -B tools/slot10_cuda.py infer $plane assets_aux/Flight_DSC02983.jpg runs/first/Flight
& $py -B tools/render_anchor_probes.py assets_aux/Flight_DSC02983.jpg runs/first/plane.png --manifest runs/first/Flight.manifest.json --prediction 10 runs/first/Flight runs/first/Flight.log
& $py -B tools/verify_two_camera_release.py
```

`infer --rounding simulator`（默认）以固定版 x86 SDK 为对照；
`infer --rounding simple` 是普通 FP32 重量化。两种都不用 FP64。
实测的逐字节一致不外推到未测试输入，也不保证模拟器与芯片一致。
详见 [两种舍入的对照](docs/NBG-ROUNDING-AB.md)。

CPU 路径在 Ubuntu x86-64 中使用固定版 VeriSilicon `TIM-VX`
SDK（提交 `702715d714277376d7ff466b7b72833ff57564c1` 的
`prebuilt-sdk/x86_64_linux`），需要 C 编译器、libc 开发文件和 Python 3。
`run_sony_nbg_cpu.sh` 会验证兼容的模拟库哈希。例如在 Linux 项目根目录：

```bash
mkdir -p runs/cpu
bash tools/run_sony_nbg_cpu.sh /path/to/x86_64_linux runs/cpu/build \
  nab_graphs/r5_slots/slot01.nb runs/cpu/Human \
  debug/cpu_inputs/512x336_rgb/Human_DSC03106.raw >runs/cpu/Human.log
python3 tools/render_pose_boxes.py runs/cpu/Human runs/cpu/Human.log \
  debug/cpu_inputs/512x336_rgb/manifest.json assets_aux/Human_DSC03106.jpg runs/cpu/human.png
```

动物槽使用 `debug/cpu_inputs/416x288_rgb/` 下的 Cat 或 Empty 输入。
A7R V 的输出交给 `tools/render_animal_pose_boxes.py`，A1 II 的输出交给
`debug/deepseek_pose_dispatch/tools/a1ii_animal_box.py`（照片示例用
`--thresholds 0.02`）。两者的参数顺序都是输出前缀、日志、输入 manifest、
原图、输出 PNG。

## 文件索引

| 模块 | 文件与用途 |
|---|---|
| 固件数据与原始网络 | `parsers/nab_extract.py` 从 NAB 提取原始 NBG 并记录哈希；`nab_graphs/{r5,1m2}_slots/` 保留本次用到的原图，原目录清单还记录其他未随提交携带的槽 |
| CPU | `tools/run_sony_nbg_cpu.sh`、`nbg_cpu_probe.c` 和 `vsim_*.c`：执行原始图、校正旧的 SRAM 重定位；`trace_nbg_*.c` 可选采样权重和量化 |
| CUDA | `tools/slot08_cuda.py`、`slot10_cuda.py` 独立推理并导出字节张量；`nbg_cuda_*.py`、`verify_nbg_*.py` 分别负责运算、图的分层恢复和对照 |
| 输入与框 | `tools/prepare_nbg_inputs.py` 记录缩放与补边；`render_{pose_boxes,animal_pose_boxes,anchor_probes}.py` 生成叠图和 JSON |
| CP 证据 | `tools/probe_sony_cp_*.py` 和 `parsers/xhd_cp_result.py`；[区域缩放](docs/CP-REGION-SCALE.md)、[置信度](docs/CP-CONFIDENCE.md)、[NMS 接口和备份配置](docs/CP-XHD-NMS.md)、[结果格式](docs/CP-XHD-RESULT.md) |
| 样例和参考 | `assets_aux/` 为测试输入，`debug/cpu_inputs/` 为预处理字节，`debug/cpu_raw/` 为 GPU 权重包及选定 CPU 参考，`debug/visual_checks/` 为样例框和坐标 |

完整固件、CP 镜像、第三方 SDK 和大型中间跟踪不随本提交。CP 回归
要另外提供文档钉定的 CP 镜像；缺失时相关原指令测试会跳过。
[槽 8](docs/SLOT08-CUDA.md) 与 [槽 10](docs/SLOT10-CUDA.md)
记录了可选的跟踪与权重重建过程，其中部分路径指向未入库的中间数据。
