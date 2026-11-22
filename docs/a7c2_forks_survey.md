# a7c2-fl23-research 139 个 fork 逐个核查

上游：`QAQLeonard/a7c2-fl23-research`
核查时间：2026-11-01
上游 main：`ef7db11`（2026-10-25，共 133 个提交、526 个文件、9 个分支）

---

## 一、核查方法（可复现）

1. `git clone --bare` 上游，得到完整历史（133 提交）。
2. 对 139 个 fork 逐个 `git ls-remote --heads`，拿到每个 fork 的全部分支头 SHA。
3. 把每个 fork 的 `main` 与上游 `main`（`ef7db11`）比对：
   - SHA 相同 → **无改动**
   - SHA 是上游祖先 → **落后**（只是旧快照）
   - SHA 不是上游祖先 → **独立线**
4. 对独立线，`--filter=blob:none` 拉取树对象，`merge-base` + `rev-list` + `diff --name-status` 得到「领先几个提交、改了哪些文件」。
5. 对非 main 分支，用上游同名分支的 SHA 作基线算领先提交数；上游没有的分支用 `merge-base main` 作基线。
6. 全程只读：只拉取 commit/tree 对象，不写入任何 fork。

数据文件（都在 `work/a7c2_forks/`）：

| 文件 | 内容 |
|---|---|
| `forks.txt` | 139 个 fork 名单（按目标里的日期顺序） |
| `refs_raw.txt` / `refs.csv` | 每个 fork 的全部分支头 SHA（350 行） |
| `branch_status.csv` | 每个分支相对上游同名分支：SAME / DIFF / FORK_ONLY |
| `ancestry.txt` | 22 个不同的 main SHA：BEHIND 还是 DIVERGED |
| `diverged_summary.json` | 8 条独立线的提交信息与改动统计 |
| `branch_summary.json` | 53 条不同分支线的提交信息与改动统计 |
| `fork_table.md` | 139 行逐 fork 表（本文第三节） |

---

## 二、总览

| 分类 | fork 数 | 说明 |
|---|---|---|
| main 与上游完全一致、且无其它分支 | **38** | 纯 fork，没有做任何改动 |
| main 无改动、但在其它分支有领先提交 | **16** | 把工作放在 feature / codex / build 分支 |
| main 停在上游旧提交（14 个不同快照点）、无分支工作 | **67** | 只是 fork 得早，没有自己的提交 |
| main 落后、但在其它分支有领先提交 | **8** | 分支上有工作，main 没合并 |
| main 有独立提交、且无分支工作 | **2** | mplind、qin-ziliang |
| main 有独立提交、且有分支工作 | **6** | oiOxOio、twjohnny42、Gureum-Yeoreum-Tori、609476965、sstyo、kopwei |
| 仓库不可读（私有或已删除） | **2** | `ChambersXDU`、`fumo3xia` |

**关键判断：139 个 fork 里只有约 20 条不同的工作线，其中真正新增相机功能的不到 10 条；其余要么没动，要么是同一个人的多个账号/同一份工作被重复 fork。**

同一份工作被多个账号重复持有（SHA 完全相同）的证据：

| 工作 | 相同 SHA 的账号 |
|---|---|
| `feature/precapture-offline` | KimiLLee、qq5272689、littleben、EVA271828、Thebearcoding、Ady0402、linxingyang、Mu-Qian-5130、RayNSN、waleyGithub、ailee945（各自 SHA 不同，内容同源） |
| `feature/pixel-shift` | qq5272689、RayNSN、zuooo、Mu-Qian-5130、waleyGithub（同上） |
| `feature/double-exposure-average` | a5272689、zuooo、lifewxy、TerETAS（**SHA 完全相同**） |
| `build/*`、`feature/desktop-*`、`refactor/desktop-current-flows` | waleyGithub、RayNSN、LeaMem、ailee945（**SHA 完全相同**） |
| `develop` | Thebearcoding、Ady0402、linxingyang（**SHA 完全相同**） |

---

## 三、真正有内容的改进（按性质分组）

### A. 相机侧研究（对 A7M4 移植最有参考价值）

| 工作 | 谁 | 规模 | 内容 |
|---|---|---|---|
| 预拍摄离线研究 | KimiLLee 116、qq5272689 119、littleben 113、EVA271828 112、Thebearcoding 105、Mu-Qian-5130 87、RayNSN 83、ailee945 72、waleyGithub 86 | 每份约 300–350 个新增文件 | 新增 py/c/h/S，含 RAW 后备分配、AP 采集排序、CP 读出、RPC 生命周期、DMM 配置、包导出与诊断 |
| Pixel Shift 研究 | zuooo 57、qq5272689 51、Mu-Qian-5130 40、waleyGithub 37、RayNSN 35 | 每份 65–95 个文件 | 「区分 Program Shift 与 Pixel Shift」、CAIF 队列、GPIO 写、九点图像坐标、CP 影子读 |
| 双重曝光平均 | a5272689 | 38 提交 / 265 文件（197 py、31 c、20 h） | 原生请求 + SDK 回调 + 媒体注册；ARM64 两帧均值内核；离线分块暂存与显式核心加载 |
| 视频模式 CP 模式选择 | oiOxOio | 10 提交 / 29 文件 | 离线 CP 模式清单、模式查表与 vtable 候选、选择谓词与直接调用者追踪 |

> 注意：这几条研究线的提交信息高度模板化（"Trace native…"、"Qualify original…"），且同一份工作出现在多个账号下，**更像是一份研究笔记/代码骨架被反复 fork，而不是多人独立开发**。价值在于它们把 A7C2 的 AP/CP 内部结构写清楚了，可直接当 A7M4 的对照参考。

### B. 桌面工具与工程化

| 工作 | 谁 | 规模 | 内容 |
|---|---|---|---|
| 独立大改（最大一条） | mplind | 81 提交 | 英文 UI 默认、个人版/公开版分离（个人资产只在个人版打包）、A7CR 支持、macOS Intel 打包、ExifTool 参数注入修复、HALD 导出、PC Remote 影片配方写入、胶片 look 库 |
| 英文界面 | ttuyetdinh | `feature/english-ui` 51 提交 / 224 文件；`backup/english-ui-develop` 17 提交 / 58 文件 | 非侵入式英文界面叠加层、Jiritsu 文件错误与 USB 传输错误区分、AF 显示缩放 |
| 韩文界面 | Gureum-Yeoreum-Tori | main 2 提交 + `feat/korean-ui` 4 提交 / 22 文件 | 韩文 README 设为默认、桌面界面语言选择、Mac 前置条件文档 |
| Windows 服务驱动检测 | **oiOxOio** | 1 提交 / 8 文件 | 见下节，**对 A7M4 移植最直接** |
| A7CR 一键准备 App | twjohnny42 | 5 提交 / 44 文件 | Windows 一键准备、按机身安装、定向恢复、驱动预检 |
| A7CR 工作流文档 | littleben | 1 提交 / 4 文件 | 外部 A7CR 工作流与验证边界 |
| PhotoLUT / HNCS look | jeffrayen | 各 1 提交 | Windows 离线给 JPEG 套 .cube；HNCS 风格 A7Look 候选（来自 Shumon Saito 配置） |
| 预览冻结修复 | F1reD1amond | 1 提交 / 8 文件 | 自定义 look 接管预览时结束原生 LUT 更新 |
| GUI 主题 | ZONGRUICHD-Rd | 1 提交 / 10 文件 | 暖色亮色主题重设计 |
| macOS Intel 打包 | wwengg | 1 提交 / 3 文件 | 支持 macOS x86_64 打包 |
| Linux / Unraid | chenyucheng0503 | 3 提交 / 24 文件 | 默认部件顺序、Linux 实例安装权限校验、Linux 下免驱动安装与恢复（中文提交） |
| 会话戳记 + 重试 | pengyinyu2009-oss | 2 提交 / 7 文件 | 会话戳记 AP 字段按遥测接受、journal replace 重试（#44/#49） |
| 离线集成修复 | nanguoopxiao | 1 提交 / 61 文件 | 服务优先工作流与当前 FL23/LUT 兼容 |
| 开发分支 | Thebearcoding / Ady0402 / linxingyang | 13 提交 / 38 文件 | 从 PR #9 移植旧主机工作流修复；LUT 刷新确认、原生 LUT 上传门控、EVF LUT 门控试运行 |
| codex 分支 | RayNSN / waleyGithub | 各 1 提交 | 恢复原生 LUT 上传门控并迁移旧版；ExifTool 文件名参数注入；混合 look 批量恢复保留标题计划 |
| 编译器缓存 | Thebearcoding | 1 提交 / 4 文件 | 离线编译器缓存按输入内容失效 |

### C. 发布工程（不是相机功能）

| 工作 | 谁 | 内容 |
|---|---|---|
| `build/release-v3.0.3-corrected`、`build/release-v3.0.4`、`-r2` | waleyGithub / RayNSN / LeaMem / ailee945（同 SHA） | 给 CI 附 43 个分发输入（21 个 `.a7look` + 15 个 `.dcp` + json/txt/csv/zip） |
| `build/mac-service-cooldown-10.21` | 同上 | Mac DMG 构建：相机弹回 MSC 时冷却并重试服务进入 |
| `refactor/desktop-current-flows` | 同上 | 删退役桌面引擎、共用同一份 Sony USB 依赖（78 文件） |

### D. 文档 / 许可 / 本地化

| 工作 | 谁 | 内容 |
|---|---|---|
| 禁止付费条款 | qin-ziliang | README 补「禁止付费使用」并链接 A7CR 独立研究 |
| 许可更新 | 上游自己 | 改用未修改的 PolyForm 非商业许可、免责声明加粗 |

### E. 停在旧版本的 fork（75 个）

它们的 main 是上游历史上的某个提交，典型停靠点：

| 停靠提交 | fork 数 |
|---|---|
| 把被接受的发布版本号改为 3.0.3 | 26 |
| README 改用英文并保留中文翻译 | 10 |
| 采用未修改的 PolyForm 非商业许可 | 3 |
| 原生构建改用 UTF-8 | 3 |
| 发布 3.0.2 | 7 |
| 新增「发布已检查草稿 release」工作流 | 6 |
| 发布 v2.1 / v2.2-preview.1 / 声音片段 0.5 秒 / 合并 PR #6 等 | 其余 20 |

**这些不是「改进」，只是 fork 得早。**

---

## 四、对 A7M4 移植最有用的三条

### 1. oiOxOio 的 Windows 服务驱动检测修复 —— 直接命中「服务模式入口」

`src/camera_runtime/transport.py` 增加 `SERVICE_DEVICE_SCAN`：

```python
def _sony_pids():
    backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
    pids = ['%04x' % d.idProduct for d in usb.core.find(find_all=True, id_vendor=0x054c, backend=backend)]
    if SERVICE_DEVICE_SCAN is not None:
        pids.extend(SERVICE_DEVICE_SCAN())
    return list(dict.fromkeys(pids))
```

`src/desktopapp/platforms/usb_drivers.py` 用 Windows SetupAPI 枚举 `USB\VID_054C&PID_xxxx`，把「在设备管理器里可见但 libusb 看不到」的 `0336` 服务设备补进来；未知 PID 仍不作为安装目标；服务接口打不开时仍停止、不重试认证。

**为什么对 A7M4 重要**：A7M4 免 Fastboot 方案的第一步就是把相机切到服务模式并让上位机看到 `0x054C:0x03E2`。在 Windows 上，如果 WinUSB 驱动没绑定，libusb 根本枚举不到这个设备——这正是 oiOxOio 修的那个坑。移植时这一段可以直接拿来用。

### 2. RayNSN / waleyGithub 的 codex 分支 —— 三个小修复

- `codex/restore-native-lut-gate`：恢复原生 LUT 上传门控并迁移旧版（19 文件）
- `codex/fix-exiftool-filename-injection`：修 ExifTool 文件名参数注入（4 文件）
- `codex/fix-mixed-look-name-resume`：混合 look 批量恢复时保留完整标题计划（4 文件）

这三个都是小范围、可验证的 bug 修复，适合在 A7M4 侧原样搬。

### 3. `feature/precapture-offline` / `feature/pixel-shift` 的代码骨架

它们把「原生 CP 模式选择、模式查表、vtable 候选、选择谓词」写成了可读的 Python/C 骨架。A7M4 的 `methods/03` 只给了补丁偏移，没给语义；这些研究分支正好补上语义层。

---

## 五、逐个核查表（139 个 fork）

口径：main 列是 main 相对上游 ef7db11 的状态；其它分支 列只列有领先提交的分支，已并入主线的分支不列。

| # | 日期 | Fork | main 相对原始仓库 | 其它分支（有内容的） |
|---|---|---|---|---|
| 1 | 2026-11-01 | oiOxOio | **独立线**：修 Windows 服务驱动检测（libusb 绑定前，8 文件） | 视频模式 CP 模式选择离线研究（29 文件） — 11 提交；修 Windows 服务驱动检测（8 文件） — 1 提交 |
| 2 | 2026-11-01 | jessonwood | 无改动 | — |
| 3 | 2026-10-31 | mplind | **独立线**：81 提交独立线：英文 UI、个人版/公开版分离、A7CR、macOS Intel 打包、ExifTool 注入修复 | — |
| 4 | 2026-10-31 | ttuyetdinh | 无改动 | 英文界面叠加层（58 文件） — 17 提交；英文界面（224 文件） — 51 提交 |
| 5 | 2026-10-29 | qin-ziliang | **独立线**：README 补「禁止付费使用」条款并链接 A7CR 独立研究 | — |
| 6 | 2026-10-28 | JornahLee | 无改动 | — |
| 7 | 2026-10-28 | NCWU2017 | 无改动 | — |
| 8 | 2026-10-28 | 8g6487 | 无改动 | — |
| 9 | 2026-10-28 | MrDannyWu | 无改动 | — |
| 10 | 2026-10-28 | NBHultraman | 无改动 | — |
| 11 | 2026-10-28 | kyralo | 无改动 | — |
| 12 | 2026-10-28 | agitoxin | 无改动 | — |
| 13 | 2026-10-28 | luoxuehuan | 无改动 | — |
| 14 | 2026-10-28 | hackerc5 | 无改动 | — |
| 15 | 2026-10-28 | KHTdhl | 无改动 | — |
| 16 | 2026-10-28 | Raxcl | 无改动 | — |
| 17 | 2026-10-28 | wisnuprama | 无改动 | — |
| 18 | 2026-10-28 | pengyinyu2009-oss | 无改动 | 会话戳记按遥测接受 + journal replace 重试（7 文件） — 2 提交 |
| 19 | 2026-10-28 | chenyucheng0503 | 无改动 | Linux / Unraid 安装与恢复支持（24 文件） — 3 提交 |
| 20 | 2026-10-28 | fifizy | 无改动 | — |
| 21 | 2026-10-28 | luo744557 | 无改动 | — |
| 22 | 2026-10-28 | wwengg | 无改动 | macOS x86_64 打包（3 文件） — 1 提交 |
| 23 | 2026-10-27 | clownfive | 无改动 | — |
| 24 | 2026-10-27 | F1reD1amond | 无改动 | 修预览冻结：自定义 look 接管预览时结束原生 LUT 更新（8 文件） — 1 提交 |
| 25 | 2026-10-27 | ayanami520 | 无改动 | — |
| 26 | 2026-10-27 | ZONGRUICHD-Rd | 无改动 | GUI 暖色亮色主题重设计（10 文件） — 1 提交 |
| 27 | 2026-10-27 | jeffrayen | 无改动 | HNCS 风格 A7Look 候选（来自 Shumon Saito 的配置，4 文件） — 1 提交 |
| 28 | 2026-10-26 | silov | 无改动 | — |
| 29 | 2026-10-26 | wxy0118 | 无改动 | — |
| 30 | 2026-10-26 | PanYunbo | 无改动 | — |
| 31 | 2026-10-26 | pianshu666 | 无改动 | — |
| 32 | 2026-10-26 | xiaoyewww | 无改动 | — |
| 33 | 2026-10-26 | fayfay123 | 无改动 | — |
| 34 | 2026-10-26 | betteray | 无改动 | — |
| 35 | 2026-10-26 | liushihan97 | 无改动 | — |
| 36 | 2026-10-26 | JrrrZzz | 无改动 | — |
| 37 | 2026-10-26 | xiongzhuozhuo | 无改动 | — |
| 38 | 2026-10-26 | ephowe | 无改动 | — |
| 39 | 2026-10-26 | UltraDream | 无改动 | — |
| 40 | 2026-10-26 | Liam0205 | 无改动 | — |
| 41 | 2026-10-26 | a5272689 | 无改动 | — |
| 42 | 2026-10-26 | Sstrawer | 无改动 | — |
| 43 | 2026-10-26 | snnallmouse | 无改动 | — |
| 44 | 2026-10-26 | AranRedfield | 无改动 | — |
| 45 | 2026-10-26 | LongJingGit | 无改动 | — |
| 46 | 2026-10-26 | xcodeshen | 无改动 | — |
| 47 | 2026-10-26 | MIN-JEONGHO | 无改动 | — |
| 48 | 2026-10-26 | xzhouzhou758 | 无改动 | — |
| 49 | 2026-10-26 | wdw89 | 无改动 | — |
| 50 | 2026-10-25 | zuooo | 无改动 | — |
| 51 | 2026-10-25 | urskdz | 无改动 | — |
| 52 | 2026-10-25 | lifewxy | 无改动 | — |
| 53 | 2026-10-25 | lorrin-x | 无改动 | — |
| 54 | 2026-10-25 | TerETAS | 无改动 | — |
| 55 | 2026-10-25 | twjohnny42 | 落后（停在「采用未修改的 PolyForm 非商业许可」） | A7CR Windows 一键准备 App（44 文件） — 5 提交 |
| 56 | 2026-10-25 | Mu-Qian-5130 | 无改动 | — |
| 57 | 2026-10-25 | SinZhangQ | 无改动 | — |
| 58 | 2026-10-25 | MiChuan | 无改动 | — |
| 59 | 2026-10-25 | Gureum-Yeoreum-Tori | **独立线**：韩文 README 改为默认 + 韩文界面（feat/korean-ui 4 提交） | 韩文界面 + 韩文 README（22 文件） — 4 提交 |
| 60 | 2026-10-25 | twoBxx | 落后（停在「采用未修改的 PolyForm 非商业许可」） | — |
| 61 | 2026-10-25 | zhongjiongsheng | 落后（停在「采用未修改的 PolyForm 非商业许可」） | — |
| 62 | 2026-10-25 | qq5272689 | 落后（停在「免责声明加粗并明确反对商业盈利」） | — |
| 63 | 2026-10-25 | cainebear | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 64 | 2026-10-25 | KimiLLee | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 65 | 2026-10-24 | littleben | 落后（停在「把被接受的发布版本号改为 3.0.3」） | A7CR 外部工作流与验证边界文档（4 文件） — 1 提交 |
| 66 | 2026-10-24 | EVA271828 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 67 | 2026-10-24 | Counkery | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 68 | 2026-10-24 | alecctv | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 69 | 2026-10-24 | timestarv | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 70 | 2026-10-24 | Phoonmando | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 71 | 2026-10-24 | Thebearcoding | 落后（停在「把被接受的发布版本号改为 3.0.3」） | 开发分支 — 13 提交；离线编译器缓存按输入内容失效（4 文件） — 1 提交 |
| 72 | 2026-10-24 | Ady0402 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | 开发分支 — 13 提交 |
| 73 | 2026-10-24 | linxingyang | 落后（停在「把被接受的发布版本号改为 3.0.3」） | 开发分支 — 13 提交 |
| 74 | 2026-10-24 | JueMzi | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 75 | 2026-10-24 | Vincent-Huang-2000 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 76 | 2026-10-24 | katebloom6 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 77 | 2026-10-24 | jason9359 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 78 | 2026-10-23 | liming-byte | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 79 | 2026-10-23 | 7487 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 80 | 2026-10-23 | Lur1N77777 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 81 | 2026-10-23 | aznikline | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 82 | 2026-10-23 | atom30260-jpg | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 83 | 2026-10-23 | waleyGithub | 落后（停在「把被接受的发布版本号改为 3.0.3」） | Mac DMG 构建：服务进入冷却与重试（4 文件） — 2 提交；发布工程：附 3.0.3 分发输入（43 文件） — 1 提交；发布工程：附 3.0.4 原生包分发输入（43 文件） — 1 提交；发布工程：3.0.4 原生包分发输入重打（43 文件） — 1 提交；修 ExifTool 文件名参数注入（4 文件） — 1 提交；混合 look 批量恢复时保留完整标题计划（4 文件） — 1 提交；删除退役桌面引擎、共用同一份 Sony USB 依赖（78 文件） — 2 提交 |
| 84 | 2026-10-23 | kungger888 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 85 | 2026-10-23 | Deejan | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 86 | 2026-10-23 | RayNSN | 落后（停在「把被接受的发布版本号改为 3.0.3」） | Mac DMG 构建：服务进入冷却与重试（4 文件） — 2 提交；发布工程：附 3.0.3 分发输入（43 文件） — 1 提交；发布工程：附 3.0.4 原生包分发输入（43 文件） — 1 提交；发布工程：3.0.4 原生包分发输入重打（43 文件） — 1 提交；恢复原生 LUT 上传门控并精确迁移旧版（19 文件） — 1 提交；删除退役桌面引擎、共用同一份 Sony USB 依赖（78 文件） — 2 提交 |
| 87 | 2026-10-23 | kedzlcy | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 88 | 2026-10-23 | LeaMem | 落后（停在「把被接受的发布版本号改为 3.0.3」） | Mac DMG 构建：服务进入冷却与重试（4 文件） — 2 提交；发布工程：附 3.0.3 分发输入（43 文件） — 1 提交；发布工程：附 3.0.4 原生包分发输入（43 文件） — 1 提交；发布工程：3.0.4 原生包分发输入重打（43 文件） — 1 提交；删除退役桌面引擎、共用同一份 Sony USB 依赖（78 文件） — 2 提交 |
| 89 | 2026-10-23 | melonbluZhujc | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 90 | 2026-10-23 | hujunqiang | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 91 | 2026-10-23 | fansong1998 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 92 | 2026-10-23 | Tianyakes | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 93 | 2026-10-23 | mzxuan08 | 落后（停在「把被接受的发布版本号改为 3.0.3」） | — |
| 94 | 2026-10-23 | ailee945 | 落后（停在「原生构建改用 UTF-8，替换被移除的 build-input URL」） | Mac DMG 构建：服务进入冷却与重试（4 文件） — 2 提交 |
| 95 | 2026-10-22 | Spacecowboy19 | 落后（停在「原生构建改用 UTF-8，替换被移除的 build-input URL」） | — |
| 96 | 2026-10-22 | klaus-a-hub | 落后（停在「原生构建改用 UTF-8，替换被移除的 build-input URL」） | — |
| 97 | 2026-10-22 | triplemarvelous | 落后（停在「自检导入改为从自身目录加载声音规划器」） | — |
| 98 | 2026-10-22 | Moe-City | 落后（停在「发布 3.0.2：3.0.1 之后报告的 bug 修复」） | — |
| 99 | 2026-10-22 | Leihb | 落后（停在「发布 3.0.2：3.0.1 之后报告的 bug 修复」） | — |
| 100 | 2026-10-22 | JohnXLeung | 落后（停在「发布 3.0.2：3.0.1 之后报告的 bug 修复」） | — |
| 101 | 2026-10-22 | Butterballl | 落后（停在「发布 3.0.2：3.0.1 之后报告的 bug 修复」） | — |
| 102 | 2026-10-22 | long112358 | 落后（停在「发布 3.0.2：3.0.1 之后报告的 bug 修复」） | — |
| 103 | 2026-10-22 | zs69 | 落后（停在「发布 3.0.2：3.0.1 之后报告的 bug 修复」） | — |
| 104 | 2026-10-22 | ckunkun | 落后（停在「声音片段长度默认 0.5 秒」） | — |
| 105 | 2026-10-22 | q1067972249 | 落后（停在「声音片段长度默认 0.5 秒」） | — |
| 106 | 2026-10-21 | xyhmnb | 落后（停在「新增「发布已检查草稿 release」工作流」） | — |
| 107 | 2026-10-21 | fengyn | 落后（停在「新增「发布已检查草稿 release」工作流」） | — |
| 108 | 2026-10-21 | ZrSamuel | 落后（停在「新增「发布已检查草稿 release」工作流」） | — |
| 109 | 2026-10-21 | Purepalm | 落后（停在「新增「发布已检查草稿 release」工作流」） | — |
| 110 | 2026-10-21 | daylin-gao | 落后（停在「新增「发布已检查草稿 release」工作流」） | — |
| 111 | 2026-10-21 | letea | 落后（停在「新增「发布已检查草稿 release」工作流」） | — |
| 112 | 2026-10-21 | RamboCao | 落后（停在「合并 PR #6（来自 UzLung1435）」） | — |
| 113 | 2026-10-21 | nanguoopxiao | 落后（停在「合并 PR #6（来自 UzLung1435）」） | 服务优先工作流与当前 FL23/LUT 兼容修复（61 文件） — 1 提交 |
| 114 | 2026-10-21 | Tacx | 落后（停在「合并 PR #6（来自 UzLung1435）」） | — |
| 115 | 2026-10-20 | Zhihan-Wang-UT | 落后（停在「README 增加 Telegram 社区与作者社交账号」） | — |
| 116 | 2026-10-20 | UzLung1435 | 落后（停在「修 usb list 报错；修视频模式下 FL2/FL3 曲线调用导致的高光裁切」） | — |
| 117 | 2026-10-20 | li-fangjie | 落后（停在「README 增加 Telegram 社区与作者社交账号」） | — |
| 118 | 2026-10-20 | 4965898 | 落后（停在「README 增加 Telegram 社区与作者社交账号」） | — |
| 119 | 2026-10-20 | xiaojiahui0311 | 落后（停在「发布 v2.2-preview.1：恢复与服务优先 USB 修复」） | — |
| 120 | 2026-10-20 | qmh1210 | 落后（停在「发布 v2.2-preview.1：恢复与服务优先 USB 修复」） | — |
| 121 | 2026-10-19 | lin-hongkuan | 落后（停在「发布 v2.1：已验证 LUT 导入器、对比图与 DCP 署名」） | — |
| 122 | 2026-10-18 | ChambersXDU | 仓库不可读（私有 / 已删除） | — |
| 123 | 2026-10-18 | wanpeng19 | 落后（停在「发布 v2.1：已验证 LUT 导入器、对比图与 DCP 署名」） | — |
| 124 | 2026-10-18 | SunicYosen | 落后（停在「发布 v2.1：已验证 LUT 导入器、对比图与 DCP 署名」） | — |
| 125 | 2026-10-18 | xzy1107 | 落后（停在「发布 v2.1：已验证 LUT 导入器、对比图与 DCP 署名」） | — |
| 126 | 2026-10-18 | kobayashiming | **独立线**：v2.1 重写：FL23、自定义 LUT 导入器、六场景对比（386 删 / 47 改 / 26 增） | — |
| 127 | 2026-10-18 | 609476965 | **独立线**：v2.0→v2.1 LUT 独立线，16 提交（独立 LUT 设备导入与精确恢复） | — |
| 128 | 2026-10-18 | sstyo | **独立线**：v2.0 LUT 线，6 提交（LUT 与 FL23 工作流拆分） | — |
| 129 | 2026-10-18 | kopwei | **独立线**：v2.0 LUT 线，5 提交（LUT 与 FL23 工作流拆分） | — |
| 130 | 2026-10-17 | thc817w | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 131 | 2026-10-17 | AbsoluteRay | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 132 | 2026-10-17 | fumo3xia | 仓库不可读（私有 / 已删除） | — |
| 133 | 2026-10-17 | i-Greece | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 134 | 2026-10-17 | Zhoujie-SONG | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 135 | 2026-10-17 | cissoid | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 136 | 2026-10-17 | EricFan2002 | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 137 | 2026-10-17 | luoyulong9588 | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 138 | 2026-10-17 | yourpapayouknow | 落后（停在「README 改用英文并保留中文翻译」） | — |
| 139 | 2026-10-17 | Hiddenl | 落后（停在「README 改用英文并保留中文翻译」） | — |
