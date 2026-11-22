# NBG / VPMN 格式分析（Adam NPU）

> 结论先行：Adam NPU 是 **VeriSilicon Vivante VIP9000**，模型格式是 **NBG (Network Binary Graph)**。
> `ai_dict/default.nab` 是 **15 个 NBG 图的容器**，不是单一场景模型。
> NBG 是 **NPU 机器码 + 索引表**，不是可转换的标准卷积权重格式；主机侧推理需要 NBG 解码器。

## 1. 身份

| 证据 | 内容 |
|---|---|
| `adam_app.bin` 字符串 | `viplite_dump_run_%d.log`、`command`、`states`、`pool`、`verify` |
| `adam_app.bin` 字符串 | `VPMN`（唯一一次，`0x6a1bc`） |
| `.nab` 魔数 | `VPMN`（15 次） |
| 公开工具 | `nbinfo`（VeriSilicon NBG 分析器）、`vpm_run`（VIPLite 运行器） |

→ Adam NPU = **Vivante VIP9000**，运行时 = **VIPLite**。

## 2. `.nab` 是 15 个 NBG 图的容器

`VPMN` 在 `.nab` 里出现 15 次，每个图有名字：

| # | 偏移 | 大小 | 名字 |
|---|---|---|---|
| 0 | 0x490 | 0x1324a0 | `sony_net_opt_NHWC` |
| 1 | 0x132930 | 0x1f860 | `reid_NHWC` |
| 2 | 0x152190 | 0x61f20 | `reidpost_NHWC` |
| 3 | 0x1b40b0 | 0x23d420 | `tfl80_t49_e70_q49_c0_NHWC` |
| 4 | 0x3f14d0 | 0x26d9e0 | `model_NCHW` |
| 5 | 0x65eeb0 | 0x2749a0 | `model_NCHW` |
| 6 | 0x8d3850 | 0x1b9a60 | `aed_03_00_05_rc2_NHWC` |
| 7 | 0xa8d2b0 | 0x3cb4a0 | `dummy_network_name` |
| 8 | 0xe58750 | 0x3f8d20 | `dummy_network_name` |
| 9 | 0x1251470 | 0x3d7660 | `dummy_network_name` |
| 10 | 0x1628ad0 | 0x3e05e0 | `dummy_network_name` |
| 11 | 0x1a090b0 | 0x13e1a0 | `model.ckpt-1590000_416x288_NHWC` |
| 12 | 0x1b47250 | 0x13c020 | `model.ckpt-1590000_416x288_r_NHWC` |
| 13 | 0x1c83270 | 0x20320 | `tflite_djd5-U-resnet7-32_64-esch_lw_quan_max23_NHWC` |
| 14 | 0x1ca3590 | 0x150600 | `sfr_tf_resnet_454-80-160-320-512-nose_NHWC` |

已拆成独立文件：`out/models/nab_graphs/*.nb`。

## 3. NBG 头（以 `06_aed.nb` 为例）

```
0x000  "VPMN"
0x004  0x0001000f          版本
0x008  0xd4
0x00c  "aed_03_00_05_rc2_NHWC"  (64 B)
0x04c  46                   Layer Count
0x050  91                   Operation Count
0x054  1                    Input Count
0x058  4                    Output Count
0x05c  2
0x060  1                    Core Count
0x064  1
0x160  64                   Memory Pool Alignment
0x16c  0x0fff00             AXI SRAM Size
0x174  0x200000             Memory Pool Size
0x178  0x200                VIP SRAM Size
0x17c  Layer Table 起始
```

## 4. Layer Table 字段（来自 `nbinfo` 字符串）

每层 8 组 `(offset, size)`，顺序：

```
nn table / tp table / ICD / coeff header / coeff data / ppu instr / load states / dyna io
```

首层实测（`0x17c`）：

| 段 | size | offset |
|---|---:|---:|
| nn table | 116 | 628 |
| tp table | 464 | 1092 |
| ICD | 3496 | 4588 |
| coeff header | 2184 | 50792 |
| coeff data | 1936 | 52736 |
| ppu instr | 1756224 | 7924 |
| load states | 9088 | 17012 |
| dyna io | 8320 | 25332 |

- `ppu instr` 在 `0x1EF4`，大小 `0x1ACC40`，`0x1EF4 + 0x1ACC40 = 0x1AEB34 = 1,762,100`；
- 文件 `1,808,992`，差 `46,892` 字节 —— 正好是剩余段。

## 5. `coeff data` 不是权重

把首层 `coeff data`（`0xCE00`，1936 B）当 int8 权重：

```
n=1936  min=-128  max=127  mean=-0.7  std=67.6
前 32 值: [-118, 2, 1, 8, 17, 0, 0, 0, 19, 14, 1, 8, 2, 0, 0, 0,
          78, 14, 1, 8, 0, 0, 0, 0, 79, 14, 1, 8, 0, 0, 0, 0]
```

每 8 字节重复 `(X, 14, 1, 8, 0, 0, 0, 0)` —— **索引/描述符**，不是卷积权重。

## 6. 为什么主机侧推理需要解码器

- NBG 的 `ppu instr`（1.7 MB）是 **NPU 指令流**；
- `nn table` / `tp table` 是 **NPU 命令流**；
- `coeff data` 是 **索引表**；
- 真正的权重嵌在指令流里，按 NPU 的 tile/量化格式编码；
- 要还原成标准卷积权重，需要 **NBG 解码器**（`nbinfo` 或 VIPLite 驱动）。

## 7. 本机限制

- `nbinfo` 已下载（`out/models/nbinfo`，187,008 B，x86-64 ELF，只依赖 `libc.so.6`）；
- 本机 **无 WSL 发行版**（`wsl --status` 空，`wsl --list --all` 只有帮助）；
- 因此 `nbinfo` **无法直接运行**。

## 8. 可行路径

1. **装 WSL + Ubuntu**，跑 `nbinfo -a 06_aed.nb`，一轮拿到 46 层算子/量化/权重；
2. **逐函数反编译 `nbinfo.elf`**（141 函数），复刻层表解析；
3. 提供 **NBG/VPMN 格式文档**。

## 9. 不做的事

- 不用错位的 stride 硬解 46 层；
- 不把索引表当权重；
- 不把随机响应当检测结果。
