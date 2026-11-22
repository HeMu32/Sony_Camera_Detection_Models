# Sony 三机照片图像降噪关联研究

分析日期：2026-11-22

分析对象：

- ILCE-7M4 v1.05 与 v6.02
- ILCE-7RM5 v4.00
- ILCE-1M2 v4.00

本文回答三个问题：

1. 三台相机是否编译了相同的照片图像降噪算法？
2. 7M4 从 v1.05 升级到 v6.02 时，照片图像降噪算法有没有更新？
3. 如果三机不完全相同，算法移植需要做什么？

本文只使用固件静态证据：CP 的 ARM/Thumb 指令、SA/DSP 成员、成员哈希、启动程序和参数表。
跨机同场景 RAW 实拍不是判断“固件编译了什么”的必要条件。三台机身像素总数、传感器、镜头和
采样方式不同，换镜头拍摄也无法保证像素级相同输入；相机固件也不是桌面 RAW 软件，不能假定它可以
接收任意 RAW 并给出可比较输出。

## 术语与路径边界

本文讨论的是照片图像降噪，不是视频或音频降噪。

照片图像降噪证据包括：

- 静态照片控制器 `CamNrs`；
- `CAMNRS::job_YC_CalcParam`、`sjob_IsCamNrsDisabled`、
  `YC_SetDevelopingMain_NrParam`、`YC_SetModeDependent_NrParam`；
- `writeCamNrsSgcTable`；
- `CAM0 NRS SGC`、`CAM1 NRS SGC`、`CLV NRS SGC`；
- `BFNR`；
- 照片后处理 SA 包 `sa_sabin_cp_cam_post.bin`。

以下路径已经排除，不能作为照片降噪更新证据：

- `CamNrv`、`NRV_0`、`NRV_1`、`writeCamNrvSgcTable` 是视频/动态降噪路径；
- `MN_SAZNRSA07`、`sa_zoom_nr.bin`、`SetZoomNrMode`、`SetNoiseTableAdr`、
  `SetSaZoomNr*`、`SaDriverZoomNr*` 是音频输入/变焦录音降噪路径；
- 7M4 CP 中的 `[AUDIO]:prcSetZoomNrEe`、`[AIP][ZNR_EE]`、`SetZoomLpcmWp`、
  `SetSaZoomLpcmParam` 直接证明 ZNR 属于音频路径；
- `SA_FDN__SA07`、`SA_PDN__SA07` 是检测/图像几何处理任务，不能仅因与照片包同容器就归为降噪内核。

## 总结论

### 1. 三机是否编译了相同的照片图像降噪算法

**不是完整的同一份固件实现，也不是三套互不相关的算法。已经解析的照片图像降噪计算程序和
照片 `CamNrs` 控制语义相同；完整容器、机型专用成员和外围图像流水线不完全相同。**

#### 1.1 照片 SA/DSP 计算层

三台相机的 `sa_sabin_cp_cam_post.bin` 中，以下成员大小和 SHA-256 均相同：

| 成员 | 大小 | SHA-256 |
|---|---:|---|
| `DEFDET__0x02` | `0xeb00` | `407b0932ab2bbd97…` |
| `SA_DNB__SA07` | `0x10700` | `3df9d8ee0ab7bc84…` |
| `SA_DTCF_SA07` | `0x10700` | `2ac01ac2b85914cc…` |
| `SA_SMOD_SA07` | `0x13a00` | `4ad734235464d293…` |
| `SA_ZPHD20x21` | `0x7800` | `0780ec4a26e0409a…` |

这些比较按成员边界取出完整字节，不是只比较同名。当前证据支持三机复用同一批 SA/DSP
计算程序。尚未完成每个槽位到具体数学公式的逐项命名，因此不能进一步扩大成“每个像素结果必然
相同”。

#### 1.2 照片 CP 控制层

三台相机的 `CamNrs` 虚表均已定位：

| 版本 | 虚表文件偏移 | 指令模式 | 五个方法入口 |
|---|---:|---|---|
| 7M4 v1.05 | `0xe5eb90` | ARM | `0x44217c`、`0x442198`、`0x4421e8`、`0x442208`、`0x442218` |
| 7M4 v6.02 | `0xaead80` | Thumb | `0x6cfcf3`、`0x6cfd09`、`0x6cfd3d`、`0x6cfd51`、`0x6cfd5d` |
| 7RM5 v4.00 | `0xc08da8` | Thumb | `0x74ba9d`、`0x74baab`、`0x74baf9`、`0x74bb3d`、`0x74bb49` |

按各自的指令模式反汇编后，控制语义一致：

- `+0x14c` 保存照片降噪使能状态；
- `+0x150` 保存照片降噪模式值；
- 初始化时从源参数块复制多组 `u32` 到对象内部；
- 清零、置一和读取使能状态使用同一字段；
- 照片降噪 SGC 表通过 `writeCamNrsSgcTable` 写入；
- 三机都保留 `CAM0/CAM1 NRS SGC`、`CLV NRS SGC` 和 `BFNR` 任务接口。

这些方法不是逐字节相同，因为地址、常量、日志位置和编译模式不同；但不能把地址变化解释成算法
更新，控制结构和字段偏移没有观察到变化。

#### 1.3 完整照片 SA 容器与机型专用外围

三台相机的完整 `sa_sabin_cp_cam_post.bin` 容器不同：

- 7M4 有 `SA_STPL_SA07`；
- 7RM5 和 1M2 没有 `SA_STPL_SA07`；
- 7RM5 和 1M2 有 `SA_SCD__SA07`；
- 1M2 另有 `SA_ZUP__SA07` 的 `0x6e00` 成员。

因此可以确认完整后处理包不是同一份编译产物；当前证据没有证明上述差异就是照片降噪主计算内核的
替换。

### 2. 执行分支、硬件限制与编译差异

| 候选解释 | 当前判定 | 依据 |
|---|---|---|
| 编译时没有编同一算法 | **共享计算成员相同；完整流水线不是同一份编译产物** | 五个 SA/DSP 成员逐字节相同，容器和专用成员不同 |
| 实际执行分支不同 | **已解析的照片 `CamNrs` 方法未观察到差异** | 使能、模式、参数复制和 SGC 写入接口相同 |
| 硬件功能限制 | **当前静态证据不能单独证明** | 三机都有照片 NRS/BFNR 接口和控制器；缺少寄存器时序和硬件吞吐证据 |

精确结论是：

> **三机编入了同一批照片降噪 SA/DSP 计算程序，并保留同源的照片降噪控制框架；完整固件和
> 外围图像流水线并不相同。现有证据没有显示照片降噪主算法被整体替换，也没有证明差异来自硬件能力。**

## 7M4 v1.05 → v6.02 的照片降噪更新

### 1. 照片 SA 包：未更新

| 项目 | v1.05 | v6.02 |
|---|---:|---:|
| `sa_sabin_cp_cam_post.bin` 大小 | 1,952,512 B | 1,952,512 B |
| SHA-256 | `6ea5cda332e49b02…` | `6ea5cda332e49b02…` |

两版整个文件逐字节相同，因此以下成员也全部相同：

```text
DEFDET__0x02
SA_DNB__SA07
SA_DTCF_SA07
SA_SMOD_SA07
SA_ZPHD20x21
SA_FDN__SA07
SA_PDN__SA07
```

这是照片后处理 SA 包层面的强证据：升级没有替换该包中的照片降噪计算成员，也没有替换包内
其他成员。

### 2. 照片 `CamNrs` 控制层：未观察到更新

7M4 v1.05 的 `CamNrs` 使用 ARM，v6.02 使用 Thumb。分别解码后，两者以下行为一致：

- 初始化复制同一尺寸的源参数块；
- `+0x14c` 保存使能状态；
- `+0x150` 保存模式值；
- 清零、置一和读取使能状态；
- 从源参数块复制多组 `u32`；
- 使用 `writeCamNrsSgcTable` 写入静态照片 SGC 表。

CP 文件大小和整体哈希改变，不能据此声称所有 CP 代码逐字节相同；但当前没有观察到照片
`CamNrs` 控制行为更新。

### 3. 版本间真实变化与照片降噪的关系

| 变化 | 实际性质 | 是否属于照片图像降噪更新 |
|---|---|---|
| `sa_sabin_cp_zima_post.bin` 中 `MN_SAZNRSA07` 哈希变化 | 音频/变焦录音 ZNR 成员载荷替换 | **否** |
| `sa_zoom_nr.bin` 两版逐字节相同 | 音频/变焦录音参数表未变 | **否** |
| `sa_sabin_mp_cam_post.bin` 大小与成员变化 | MP 包新增 `SA_SHA250x02`，已有 PCMCC/NV12/ROT 成员相同 | **否** |
| 多数 `/usr/sabin` 文件逐字节相同 | AED/BED/FDN/PDN 字典、静态 PSF 表等未变 | 不是照片降噪更新 |

ZNR 哈希变化是真实版本差异，但 CP 字符串把相关路径明确标为 `[AUDIO]` 和 `LPCM`，所以不能拿它
回答照片图像降噪是否更新。

### 4. 7M4 版本结论

> **7M4 v1.05 → v6.02：没有观察到照片图像降噪算法或照片 `CamNrs` 控制行为更新。
> 已确认的版本变化是音频/变焦录音 ZNR 成员替换，以及 MP 包加入 `SA_SHA250x02` 等非照片 NR
> 变化。**

## 7M4 v6.02 与 7RM5 v4.00 的照片降噪区别

### 1. 已解析的相同部分

7M4 v6.02 与 7RM5 v4.00：

- 五个照片 SA/DSP 成员逐字节相同；
- 照片 `CamNrs` 的五项控制方法语义一致；
- 都有 `CAM0/CAM1 NRS SGC`、`CLV NRS SGC` 和 `BFNR` 任务接口；
- 都使用 `+0x14c/+0x150` 的照片降噪状态和模式字段。

### 2. 已确认的不同部分

两台相机的完整 `sa_sabin_cp_cam_post.bin` 容器不同：

- 7M4 有 `SA_STPL_SA07`，7RM5 没有；
- 7RM5 有 `SA_SCD__SA07`，7M4 没有；
- 成员偏移、容器大小和周边图像处理任务组织不完全相同。

### 3. 当前不能扩大成什么结论

不能仅凭这些差异直接断言：

- 两台相机的照片降噪数学内核不同；
- 7RM5 或 7M4 已经替换成另一套主算法；
- 某台相机因硬件限制不能运行同一算法；
- 两台相机的最终照片必然在相同像素输入上得到不同结果。

要得到这些结论，还需要补齐参数来源、图像缓冲尺寸、任务调度、具体硬件寄存器和端到端输入输出验证。

## 算法移植需要做什么

### A. 先定义移植目标

移植目标必须明确是以下一种：

1. 在另一台 Sony 机身上复现某条照片降噪路径；
2. 在主机/PC 上实现可运行的等效模型；
3. 只移植照片降噪控制策略，不实现像素级滤波器。

这三个目标的交付物不同。不能把“复制一个 `.bin`”当作其中任何一个。

### B. 移植照片降噪控制层

1. 固定目标机身、固件版本和拍照模式；
2. 定位目标机的 `CamNrs` 虚表和方法入口，确认 ARM/Thumb；
3. 对齐 `+0x14c` 使能、`+0x150` 模式、参数复制块和 `writeCamNrsSgcTable`；
4. 枚举合法/非法模式组合，保留失败返回；
5. 验证 `CAM0/CAM1 NRS SGC` 和 `BFNR` 的任务提交与完成条件。

### C. 移植照片 SA/DSP 计算层

1. 提取目标机的 `sa_sabin_cp_cam_post.bin`，按成员边界记录大小和 SHA-256；
2. 对 `DEFDET`、`DNB`、`DTCF`、`SMOD`、`ZPHD` 分别解码启动程序和槽表；
3. 记录每个成员的输入/输出缓冲、步长、位宽、槽数和启动目标；
4. 用固定夹具逐成员比较主机实现和原固件；
5. 未解出的算子、寄存器或等待条件保留为未知，不能用成片反推成已移植公式。

### D. 不能混入移植的部分

- 不能把 `CamNrv` 视频路径当作照片 `CamNrs`；
- 不能把 ZNR 音频路径当作照片降噪；
- 不能把 FDN/PDN 检测任务当作照片降噪主内核；
- 不能假定相机固件像桌面 RAW 软件一样接收任意 RAW；
- 没有硬件手册或运行时 trace 时，不能把主机近似实现标成原厂等价。

### E. 验收顺序

1. **成员门**：完整成员大小、哈希、启动目标和槽程序；
2. **程序门**：逐槽解码和固定输入输出；
3. **控制门**：`CamNrs` 合法/非法模式与字段写入；
4. **参数门**：目标机实际参数块和 SGC 表；
5. **硬件门**：NRS/BFNR 缓冲、DMA、mailbox 和等待条件；
6. **可选成片门**：只有需要回答最终画质时，才做受控输入或同机开关对比。

## 复现

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'

# 7M4 两版照片 SA、MP SA、音频 ZNR 包和参数表
& $py -B tools/compare_sa_bundles.py `
  debug/nr-study/7m4-v105/sabin/sa_sabin_cp_cam_post.bin `
  extracted/7m4/sabin/sa_sabin_cp_cam_post.bin

& $py -B tools/compare_sa_bundles.py `
  debug/nr-study/7m4-v105/sabin/sa_sabin_mp_cam_post.bin `
  extracted/7m4/sabin/sa_sabin_mp_cam_post.bin

& $py -B tools/compare_sa_bundles.py `
  debug/nr-study/7m4-v105/sabin/sa_sabin_cp_zima_post.bin `
  extracted/7m4/sabin/sa_sabin_cp_zima_post.bin

# 三机照片 SA 包成员哈希
& $py -B tools/compare_sa_bundles.py `
  extracted/7m4/sabin/sa_sabin_cp_cam_post.bin `
  extracted/r5/sabin/sa_sabin_cp_cam_post.bin `
  C:/Users/HeMu/Desktop/a7m4-fl23-works/out/models/work/1m2/sabin/sa_sabin_cp_cam_post.bin

# 7M4 两版音频 ZNR 对照，不能作为照片 NR 更新证据
& $py -B tools/compare_sa_znr_member.py `
  debug/nr-study/7m4-v105/sabin/sa_sabin_cp_zima_post.bin `
  extracted/7m4/sabin/sa_sabin_cp_zima_post.bin
```

## 证据边界

已经确认：

- 7M4 两版照片后处理 SA 包逐字节相同；
- 7M4 两版照片 `CamNrs` 控制语义一致；
- 三机照片 SA/DSP 的五个已检查成员逐字节相同；
- 三机照片 `CamNrs` 控制方法语义一致；
- 三机完整照片 SA 容器和部分机型专用外围成员不同；
- 7M4 两版音频/变焦录音 ZNR 成员发生替换。

仍未证明：

- 五个 SA/DSP 成员每个槽位的完整数学含义；
- 三机所有图像缓冲尺寸、任务调度和寄存器时序相同；
- 三机完整照片流水线只有一套执行分支；
- 三机硬件能力的严格差异；
- 三机在受控像素输入上的最终输出是否相同。

这些未知项不能通过“换镜头拍同一场景”替代固件证据，也不能从“完整固件哈希不同”直接推出
“照片降噪主算法不同”。
