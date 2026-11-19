# A7R V / A1 II：区域缩放参数

两份原始 CP 镜像的首三条 ARM 指令从文件 `+0xc` 取链接地址
`0x0010800c`，加 `4` 并跳转到 `0x00108010`。这落在原文件
`+0x10` 的 ARM 初始化代码；因此执行时以 `0x00108000` 映射该
镜像，绝对地址引用必须减去此基址后才是文件偏移。相对分支、
PC 相对字符串的文件偏移仍成立。直接从文件 `+0xc9b860` 取出的
不规则数据不是 A7R V 这张指针表。

| 机型 | 选择器文件位置 | 表 VA / 文件位置 | 区域配置基址 |
|---|---|---|---|
| A7R V | `0x1c0a98` | `0x00c9b860 / 0x00b93860` | `0x0172407c` |
| A1 II | `0x1f7d28` | `0x00d74cb0 / 0x00c6ccb0` | `0x01738f40` |

在两机上执行原 Thumb 选择器。输入模式 `0` 读取目标编号 `1..6`，
按顺序选配置块 `[0,1,0,2,3,4]`。模式 `1..8` 的固定配置块为
`[0,1,2,2,2,3,3,4]`；大于 `8` 的值走块 `0`（实测 `9`）。
块间距为 `0x600` 字节。这些都是区域配置索引，**不是 NAB 槽号
或用户界面类别**。配置内容在固件文件映射之外的运行时地址。

以绝对地址解析虚表及其 RTTI，两个镜像都指向
`Camera::FC::Keypoint::TuningParamRegionScale`。连续八个 getter
读取配置的 u16；非零时返回 `float32(值 / 1000)`，为零时才使用
下列字面回退值。用原 VFP 指令分别验证全零和八个互不相同的夹具
输入，从返回位模式逐一核对。

| 字段位移 | `+0x02` | `+0x04` | `+0x06` | `+0x08` | `+0x0a` | `+0x5e` | `+0x60` | `+0x62` |
|---|---|---|---|---|---|---|---|---|
| 零值回退 | `0.666` | `1.0` | `1.5` | `1.0` | `1.5` | `0.666` | `1.0` | `1.5` |

未证明检测时配置块的八个值为零，未将这八项绑定到具体部位，也未
查明原厂分类阈值、锚框、NMS 或飞机模式与槽 9/10 的关系。
诊断报告 `debug/deepseek_anchor/AIRPLANE-NATIVE-SLOTS.md` 的
零基绝对地址读表和由此得出的数据解释，以本实测为准；`.rsc` 的
数字类别到槽号映射仍须独立绑定到用户界面和运行时提交。

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
& $py -B tools/probe_sony_cp_region_scale.py extracted/r5/cp/cpapp-b.bin --mode 0 --target 5
& $py -B tools/probe_sony_cp_region_scale.py debug/deepseek_pose_dispatch/fw/a1ii_cp/cpapp-b.bin --mode 0 --target 5
& $py -B -m unittest discover -s tools -p 'test_sony_cp_region_scale.py' -v
```
