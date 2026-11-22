# A7R V / A1 II：原始 CP XHD/NMS 命令接口

两份固定 SHA-256 的 CP 镜像按其启动代码映射到 `0x108000`。
`tools/probe_sony_cp_xhd_nms.py` 执行 `AdamXhdDriver` 原 Thumb
参数构造和虚表方法；测试只读回命令描述符和共享缓冲区，不提交作业。

| 命令字段 `(操作号, 选择器, 参数偏移)` | 原指令返回值 |
|---|---|
| 短列表 | `(0x10, 3, +0x4bc)` |
| 长列表 | `(0, 2, +0)`, `(1, 2, +0xa8)`, `(0x10, 3, +0x4bc)` |
| 参数视图 | `+0`, `0x520` 字节 |
| 结果视图 | `+0x520`, `0x260` 字节 |

块 `+0x4bc` 首四个 `u32` 是三个输入指针和 NMS ID。块 `+0x510`
由第二个输入指针加 `0x32030` 得到，原日志称为 NPU 输出地址。
驱动 ID `8,9,10,11` 分别传入 NMS ID `0,2,3,1`，A1 II 的 `16,17`
在这条路径上与 `10,11` 一致。此映射不能单独证明相机菜单的飞机分支。

| `TuningParamIP` 字节 | 共享块用途 |
|---|---|
| `+0x171..+0x184` | 十对 `thr/thr_per`；按驱动散布到 `+0x4cd` 与 `+0x4ed` 两张 `8×4` 表，不激活的项保留原字节 |
| `+0x185` | `thr_glbl`，块 `+0x50d` |
| `+0x186` | `bgfg_thr`，块 `+0x514` |
| `+0x18a` | `skip_max_decode`，非零转为 `1`，块 `+0x4cc` |

A7R V 可在文件偏移 `0x1b8944/4c`、`0x1b89a0/aa`、
`0x1b89e8/f0` 复核这些日志的非相邻 `movw/movt` 引用。
`0x1b9122/30` 的结果日志把第二个单字节字段称为 `part`；
框和分数的格式见 [CP XHD 结果转换](CP-XHD-RESULT.md)。

下列回归在两机原指令上验证散布规则、未覆盖的字节、布尔化和
地址加法。后续已将短作业绑定到 [原始 Adam CPU 解码](ADAM-XHD.md)，
并在槽 8、10 的实图张量上运行。发布包的默认值见下文，和实机运行时
覆盖以及显式夹具分开。

### 调用者与日志读者

用显式栈和 `AdamXhdDriver` 虚表夹具回放，两机的区域设置代码都通过
虚表 `+8` 调用原描述符构造方法。调用者传入工作区 `+4`，
然后把返回的数量记入 `+0`。紧接的调试转储从同一工作区
按 `8` 字节步长读三个字段。对短表与长表，读出的元组都与上表相同。

| 原始 CP 文件偏移 | A7R V | A1 II |
|---|---|---|
| 间接调用 | `0x1ba778` | `0x1f1dd2` |
| 写入数量 | `0x1ba7ac` | `0x1f1dfc` |
| 读操作号、选择器、指针 | `0x1ba88e/90/92` | `0x1f1f18/1a/1c` |

该 CP 切片只验证区域设置代码中的交接，不执行日志或提交作业。
Adam 应用的实际收件端以第一个半字 `0x10` 查控制器，再把第二个半字 `3`
作为控制器操作号传入；它不是 SA 引擎编号。详见 [Adam XHD](ADAM-XHD.md)。

## 配置来源

下表为链接地址，读文件时须先减 `0x108000`。已回放两机
`TuningParamIP` 构造函数，检测目标 `1..6` 按顺序选择块
`0,1,0,2,3,4`。`TuningParamIP+4` 确实是指向首个字节的指针，
与 XHD 读取路径里驱动 `+0xcc` 的用法一致。

| 项 | A7R V | A1 II |
|---|---|---|
| 构造函数 | `0x2c74a0` | `0x2fe318` |
| 目标指针表 | `0xc9b820` | `0xd74c70` |
| 五块首地址／步长 | `0x0172407c` / `0x600` | `0x01738f40` / `0x600` |
| 批量初始化传入的 ID／目的地表 | `0x10fd758` / `0x10fd70c` | `0x11e910c` / `0x11e90c0` |
| 批量读取器 | `0x532eb4` | `0x5ac7d8` |

两机的批量配置表都把五块依次配到 BK4 的 `0x00060011..0x00060015`。
A7R V 的读取器于 `0x532f12..0x532f1a` 把对应指针传给
`0xba7bac`，该函数查 BK4 索引并复制命中数据；A1 II 对应
函数 `0xc7ffe6`。

### 发布默认值

每个 `0110_backup` 成员包含两段完整 BK4。第一段确实没有
命名空间 `0x0006`；第二段从 `align4KiB(u32[+0x18] + u32[+0x1c])`
开始。A7R V 的成员偏移为 `0xef000`，A1 II 为 `0x100000`。
已检查每机全部 12 个成员：第二段均有 `0x00060011..0x00060015`，
每条 `0x600` 字节，同机各地区逐字节一致。

`--published-backup` 执行原始读取器，将整条记录复制到原 RAM 地址，
再执行原 NMS 参数构造。用 `ALLLANG` 成员核对了以下组合。
`thr/thr_per` 仅列出在对应非空行里写入的元素。数字是原始整数，
不是宿主实验解码器的概率阈值。`target` 和 `driver` 在回放中显式选取，
不据此确认相机菜单或运行时覆盖。

| CP / 组合 | BK4 键 | NMS 行 | `thr` / `thr_per` | `thr_glbl` / `bgfg_thr` / 短边字节 / `skip` |
|---|---|---:|---|---|
| 两机，目标 1、驱动 8 | `0x60011` | 0、1 | 每行 `128,128,0,0` / `76,76,76,76` | `229 / 129 / 18 / 1` |
| A7R V，目标 4、驱动 10 | `0x60013` | 7 | `128,218` / `76,76` | `229 / 129 / 9 / 0` |
| A1 II，目标 4、驱动 10 | `0x60013` | 7 | `128,128` / `76,76` | `229 / 129 / 9 / 0` |

用已有 FDAT 提取器读取本地 A7R V 4.00 发布包
`firmware/CamFw/7RM5_v400_BODYDATA.DAT`，输入 SHA-256
`51e1176a255a9a4b4782dfbcfa6c5f332b71173bdfe9ed2bcf4fca6bcbc6c2e2`。
验证版本后取 `debug/r5_xhd_fw/firmware.tar`，SHA-256
`8b03700edb87d29315377a044b477792e3c47b149e4e300618b38dbe7a565813`。
用 `tarfile` 和已有 BK4 解析器仅读取归档成员，没有执行提取出的文件。
A1 II 归档为 `debug/a1ii_fw/firmware.tar`。

A7R V 另有配置序列化分支同时读取 `+0x186` 和 `+0x187`；
这不否定已经回放的 XHD 代码在各自路径上分别使用两个字节。
`+0x187` 短边过滤见 [XHD 结果转换](CP-XHD-RESULT.md)。

```powershell
& 'E:/anaconda3/envs/test311/python.exe' -B -m unittest discover -s tools -p 'test_sony_cp_xhd_nms.py' -v
& 'E:/anaconda3/envs/test311/python.exe' -B tools/probe_sony_cp_xhd_nms.py extracted/r5/cp/cpapp-b.bin --tuning-bindings
& 'E:/anaconda3/envs/test311/python.exe' -B parsers/sony_bk4.py debug/r5_xhd_fw/firmware.tar --firmware-tar --segment second --key 0x60011 --key 0x60012 --key 0x60013 --key 0x60014 --key 0x60015
& 'E:/anaconda3/envs/test311/python.exe' -B tools/probe_sony_cp_xhd_nms.py extracted/r5/cp/cpapp-b.bin --published-backup debug/r5_xhd_fw/firmware.tar --backup-member 0110_backup/SYSIPSX-DSLR/GL/CX89200_ALLLANG.bin --target 4 --driver-id 10
```
