# A7 IV 6.02：启动快照、检测参数和结果边界

## `nflasha5` 是 Linux warm-boot 镜像

原始输入为 `firmware/7M4_v602/firmware.tar` 的 `0700_part_image/dev/nflasha5`，本地副本 SHA-256 `13120416073b09b13f8501324839d37b84545267dac3d183041fceb8af5a53ca`。同包 `0300_partconf/partinf.conf` 明确将该分区标成 **Warm Boot Image**，不是 CP 文件系统。Sony A7 IV 6.00 GPL 内核归档（SHA-256 `e9f708acebf62bd5659cc578101db0d9aa308a4fc413586502451e8a58c80b0e`）中的 `include/linux/warmboot.h` 和 `kernel/power/wbi_raw.c` 对应本镜像的字段与写入顺序。

`unblob 26.6.4` 未识别外层 WBI2，报告保留在实验 VM 的 `~/a7iv-a5-unblob.dv9RuV/`，退出码 1；其中识别到的 ELF 片段不是完整文件系统。`parsers/wbi2_sections.py` 按内核格式映射原始 WBI2，并核对每个描述符的 XOR 元数据校验：

| 结构 | 文件范围（左闭右开） |
|---|---|
| 64 位 `wb_header` 所在扇区 | `0x0..0x200` |
| 未压缩快照数据 | `0x200..0xc883200` |
| 1442 个 `wb_section` 描述符 | `0xc883200..0xc899a80` |
| 描述符表扇区填充 | `0xc899a80..0xc899c00` |
| 表外尾部，语义未确认 | `0xc899c00..0xc8d9880` |

1399 个有数据的段，43 个仅作为零初始化的段。所有段都在物理范围 `0x400002000..0x424000000` 内，64 位 Linux VA 按 `PA - 0x4400000000` 模 `2^64` 映射。`0x200` 的设备树 16841 字节与同包 `0800_appli/setting/cxd90057.dtb` 完全相同。内核 resume 向量 `PA 0x4004a87f0` 对应第 3 段、文件位置 `0x4349f0`。

`fdn_front_dict.bin` 字符串恰有两处，分别在：

| 文件偏移 | 段号（零基） | PA | Linux VA |
|---|---:|---|---|
| `0x70d529c` | 1257 | `0x41e79a09c` | `0xffffffc01e79a09c` |
| `0x820d29c` | 1388 | `0x41fcdc09c` | `0xffffffc01fcdc09c` |

名字之前的 8 字节可解析为 ext2 目录项头：正脸、侧脸、PDN inode 分别为 4037、4039、4050。两个命中所在的 1 KiB 块（从 `0x70d5200` 和 `0x820d200` 开始）与 `/usr` 的 ext2 分区 `nflasha15 + 0x1100400` 的同一个块逐字节一致。这是两份目录块副本，不是两份运行参数。整个镜像中没有这三份原始字典的完整匹配，也没有这三份字典在偏移 0、128、1024 各取 128 字节的匹配。由此不能排除另一种编码的参数或模型；不能把低位的 32 位 CP VA 直接解释为该镜像的物理地址。

```powershell
& 'E:/anaconda3/envs/test311/python.exe' -m unittest parsers.test_wbi2_sections -v
& 'E:/anaconda3/envs/test311/python.exe' parsers/wbi2_sections.py debug/a7iv_parts/nflasha5 --file-offset 0x200 --find-ascii fdn_front_dict.bin --find-ascii fdn_side_dict.bin --find-ascii pdn_dic.bin
```

## CP 结果约定：不是同一种分数

以下是 `extracted/7m4/cp/cpapp-b.bin`，Thumb 虚拟地址基址 `0x108000`，可用 Ghidra 工程 `debug/ghidra/Sony7M4` 和 `tools/ghidra/DumpSonyFunctions.java` 复查。虚表项的最低位是 Thumb 标志。

| 分支 | 类型来源（RTTI） | 虚表项 → 函数 | 已确认 |
|---|---|---|---|
| FDN | `0xc17e64 → 0xc17e98 → FdnParamManager` | `0xc17e8c → 0x264d31` | `FaceDirection == 0` 用 `**(this+0x64)`，否则 `**(this+0x68)`；有符号 `FaceScore > threshold` 才置位。原始记录为分数和方向各一字、左右眼与鼻根各一对坐标、随后是 `execId`。该函数不直接输出 `xyxy`。 |
| PDN | `0xc17fb0 → 0xc17fe4 → PdnParamManager` | `0xc17fd8 → 0x266aa1` | 两个有符号 `el_score`、`er_score` 分别与同一 `**(this+0x5c)` 做严格大于比较；输出 `+8/+9` 两个标志、`+0xc/+0x10` 两个原始分数。它不是两个独立主体框。 |
| Keypoint | `0xc17ea8 → 0xc17edc → KeypointParamManager` | `0xc17ed0 → 0x265e31` | `0x265c9c` 对指定模式 2/3 做分数查表插值；用 `1/65536` 读入，乘 `65536` 并截断写回。`0x265ef0` 引用另一个相同倒数用于打印原值和换算值。两个常量是 `0x265e20/0x266198` 的 `0x37800000` 和 `0x265e2c` 的 `0x47800000`，并非 FDN 的格式证据。 |
| Rotate | `0xc180e0 → 0xc18114 → RotateParamManager` | `0xc18104 → 0x266fd1` | 这里确实向 `this+0x58` 复制一条记录，但对象类型不同，不能当作 PDN 阈值指针的写入点。 |

`PdnDrv` 构造函数 `0x2661d0` 设置内联整数 `+0x5c = 0x11`、`+0x64 = 0x2ee0`，而 PDN 比较发生在另一个类型 `PdnParamManager`。两个常量均没有证据可作为检测阈值。`parsers/pdn_sections.py` 已确认字典中的定点边界对和系数填充，但不能由此推定算子顺序或分数格式。

第二轮 DeepSeek 报告中的 FDN 分数比较可以从指令复核；其中把 Keypoint 的 `face_score(conv)`、上层主体选择的 `update thresh` 日志归到 FDN 阈值的说法证据不足。FDN 两套门限与 PDN 共用门限的实际数值仍需沿各自参数管理器或驱动构造链追查。A7 IV 原始前向与视觉验证尚未跑通。

## 阈值指针的构造来源

后续 DeepSeek 报告 `debug/deepseek_a7iv/PDN-INIT-EVIDENCE.md` 对比较点的反汇编是可用的，但“构造器不在该文件”不成立。其 `addrref.py` 用第二半字的低位区分 `MOVW`/`MOVT`，并要求该半字的最高位为 1；实际 Thumb-2 立即数指令应以第一半字的 `0xfbf0` 掩码区分 `0xf240`/`0xf2c0`，第二半字最高位为 0。改用正确解码后得到：

- **PDN**：`PdnDrv` 从 `0x26621c` 调用 `0x6fd292`，传入 `this+0xb8`。该构造函数在 `0x6fd29a..0x6fd2ae` 实例化 `0xc17fac+8`（`PdnParamManager` 虚表地址点）。`0x6fd2b2` 将 `(0, 0x0123e528+0xb0)` 写到 `manager+0x58/+0x5c`。`0x266aa0` 对左右眼都读取 `int32[0x0123e5d8]`，实际数值仍未定位。
- **FDN**：`0x6fd016` 在 `0x6fd020..0x6fd038` 安装 `0xc17e60+8`（`FdnParamManager` 虚表地址点），随后 `0x6fd046` 把 `0x0123e5c8`、`0x0123e5cc` 写进同一个对象的 `+0x64/+0x68`。这些地址确实是比较路径的阈值**指针目标**，并不是阈值数值本身。`0x6fcb4a` 构造了 10 个使用同一对指针的管理器。

三个目标地址均超出当前 `cpapp-b.bin` 文件映射末端 `0x01084110`。应追踪 `0x0123e528` 全局块的初始化，而不是把该裸文件的空洞读值、`PdnDrv` 的 `0x11` 或 `0x2ee0` 当作阈值。

## CP Boot Cross-Check

Follow-up to `debug/deepseek_a7iv/CP-INIT-FOLLOWUP.md`: the primary-core
boot path **does** populate the boot descriptor and zero the threshold
storage. This does not establish the thresholds at detection time.

Independently disassembled `extracted/7m4/cp/cpapp-b.bin` with Capstone ARM
mode, base `0x108000`, SHA-256
`7b9b137d696e5972d20db7f9119076172ed8e0730805d7f0b4c31b87689e2c2b`.
The reset stub uses the core ID at `0x108048`. Core 0 continues via
`0x10806c -> 0x108398`; nonzero cores wait at `0x1080b8` before the indirect
branch at `0x108100`. The on-disk zero at `0x108da0` is not an external-loader
dependency:

- At `0x1083a4..0x1083bc`, core 0 calls `0x108a04` with `r0=0x108da0`,
  `r1=0x128000`, `r2=0x17c`. The callee copies from `r1` to `r0` (see
  `ldm r1!`/`stm r0!` at `0x108a34`/`0x108a38`). The source descriptor is
  at file offset `0x20000`; word 0 is `0x0058184c` (`4c 18 58 00`).
- At `0x10850c..0x108520`, the page mapper `0x108768` is called with
  `(VA, length, PA) = (0x1046000, 0x51e3fc, 0x1046000)`, using descriptor
  offsets `+0x150`, `+0x14c`, and `+0x148` plus `+0x16c`. In the mapper,
  `0x108900` derives the page index from `r0` and `0x10890c..0x108918`
  inserts the page address from `r2`. This range includes all three words.
- Core 0 calls `0x58184c` through the copied descriptor at `0x108758`.
  That entry branches to `0x574cd0` for core 0. At `0x574ce0..0x574d00`,
  it calls `0x5866ac` to clear `0x1046418..0x15643fc` (end exclusive).
  At `0x5866ac` the callee sets the fill byte to zero; `0x586700` begins
  writing zero register pairs. The FDN words `0x123e5c8`/`0x123e5cc` and
  PDN word `0x123e5d8` are inside this interval. The reset-ready flag is
  set and `sev` follows at `0x574d14..0x574d1c`, after the clear.
- At `0x574d44..0x574d68` the primary core also invokes the
  null-terminated function pointer list at `0xe82ad8`, after clearing.
  The boot clear is not evidence that those words remain zero after
  constructors, task startup and detector setup.

The old negative result therefore applies to the value consumed at
detection time, not to the existence of a mapping or startup initializer.

## MINNA 的实际备份读取路径

`0x6f7730..0x6f776e` 确实先用两张表注册配置更新回调，但同一函数
随后于 `0x6f7770..0x6f7780` 把 ID 表 `0xf608cc` 和目的地表
`0xf608b8` 传给批量备份读取器 `0x4c3040`。其循环在
`0x4c309e..0x4c30a6` 从目的地表取 `r1`，调用原始 BK4
读取器 `0xac95de(key, dest, 0, 0)`。首对是
`0x000b0000 → 0x0123e528`，相应门限字段在 `+0xa0/+0xa4/+0xb0`。
这更正了先前把 `0x000b0000..3` 仅归为消息 ID 的结论。

每个 6.02 `0110_backup` 成员实际包含两段 BK4。首段的 `0x000b`
没有记录，原批量读取在第一项失败后返回 `-1`，不会再读后三项，
测试填充字节未变。第二段起于成员 `0xf3000`，可由首段头部
`+0x18/+0x1c` 的数据偏移和长度相加，再向上对齐到 4 KiB 得出。
先前的阴性只覆盖首段。DeepSeek 先在 FDAT 中定位该记录；对照
归档头后确认它正是同一 `0110_backup` 成员中的第二段。

检查全部 32 个成员的第二段，四条 `0x000b0000..3` 记录各自跨
地区一致。第一条位于第二段内 `+0x35cd0`，长 `0x100`，SHA-256
`81588ef4d258ae860b23dc3eafe5e03fb7587e7d63039ac2c7885c3838986343`。
原 CP 批量读取器从第二段确实复制四条记录、返回 `4`；加载后
不另填阈值，原 FDN 消费者就地验证两方向的有符号严格大于规则。
原 PDN 消费者也就地对左右眼分别判定，门限相等为假，大一为真，
并原样写出两眼分数。

| 字段 | 6.02 发布记录中的值 |
|---|---:|
| FDN 方向 0，`+0xa0` | `-287995` (`0xfffb9b05`) |
| FDN 其它方向，`+0xa4` | `-287995` (`0xfffb9b05`) |
| PDN 双眼，`+0xb0` | `590558003` (`0x23333333`) |

PDN 的单位后来由原 SA 写回指令确认：FP32 分数乘以 `2^30` 后写出
32 位整数，因此该门限对应约 `0.55`。这是分数编码，不等同于
校准概率。[原标量置信度回放](A7IV-SA-FORWARD.md#pdn-六路工作器与标量置信度)
使用显式特征夹具验证写回；真实图像的 PDN 前向尚未完成。

同一注册回调还会复制匹配键的内联 BK4 更新；已用合成消息
验证命中 `0x000b0000` 时写入阈值块、不匹配时不改写。以上证明
的是发布默认值与 CP 消费路径，未从机身取 RAM，未证明运行时
无覆盖，亦未用真实 DSP 分数执行 FDN。

```powershell
& 'E:/anaconda3/envs/test311/python.exe' -B -m unittest discover -s tools -p 'test_a7iv_fdn_cp_result.py' -v
& 'E:/anaconda3/envs/test311/python.exe' -B parsers/sony_bk4.py firmware/7M4_v602/firmware.tar --firmware-tar --segment second --key 0xb0000
```
