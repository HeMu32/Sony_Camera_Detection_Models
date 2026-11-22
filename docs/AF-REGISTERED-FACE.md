# Sony AF：注册人脸、身份匹配与对焦优先级

核对对象：A7 IV 6.02、A7R V 4.00、A1 II 4.00 本地 CP。Sony 网页是对应机型的公开操作说明，不用于推定功能首次加入的固件版本。本轮是静态检查，没有连接机身、登记真人或回放匹配前向。

## 公开行为

| 机型 | 人脸登记与优先设置 | 启用条件 |
|---|---|---|
| A7 IV | `Face Memory` 最多登记 8 张；`Order Exchanging` 改变多个登记项的优先顺序。`Regist. Faces Priority` 决定是否优先登记人脸。 | `Face/Eye Prior. in AF = On`、`Face/Eye Subject = Human`。 |
| A7R V、A1 II | `Face Memory` 最多登记 7 张；从列表选 **一个**当前识别目标。`Regist. Face Priority` 开启时优先当前选择；关闭时仍可对脸对焦，但不给该目标此优先项。`Switch Ident. Target` 可分配到自定义键。 | 手册给出的配置是 `Subject Recog in AF = On`、`Recognition Target = Human`；不把这个配置外推到 `Auto`。 |

登记和拍摄是两个阶段。登记时用画面中央的登记框，官方推荐明亮、正面、不遮挡。三机手册都指出单项删除后注册数据仍可能留在机身，并指示用 `Delete All` / `Delete All Faces` 删除。优先不是只允许对某一人对焦，也不能从菜单推定它在任何模式下一定覆盖手动指定、跟踪锁定或 AF 区域。

## 本地固件证据

CP 是既有的 ARM/Thumb 原始镜像，不是可直接枚举 ELF 节的文件；复核启动头并按加载基址 `0x108000` 解码 Thumb 代码。以下为文件偏移；`movw/movt` 组合出的绝对地址减基址定位字符串。这里只分析相机功能。

| 镜像 | SHA-256 前缀 | 原始指令与可作出的结论 |
|---|---|---|
| `extracted/7m4/cp/cpapp-b.bin` | `7b9b137d696e5972` | `+0x175a2e/+0x175a3e` 形成 `ImportAttributeResultPriorityFace` 签名地址；随后读脸记录列表、复制属性并比较多个字段。另有 `FaceMemory`、`CheckSetFacePriorityTrackingMode` 与删除登记人脸相关的诊断。证据支持旧版有注册设置和优先脸属性路径，**不把属性导入例程认成注册匹配算法**；未定位匹配器的学习模型、输入尺度或完整调用关系。 |
| `extracted/r5/cp/cpapp-b.bin` | `3269bf6242633979` | `getAdamParamPoseFr`、`getAdamCmdInfoPoseFr`、`getAdamResultPoseFr` 对应 `+0x185940/+0x185a08/+0x185dac` 的诊断。`+0x1a6130` 引用 `MATCH finished`，随后 `+0x1a6144/+0x1a6148` 写状态字节；`+0x1a6330/+0x1a6334` 引用 `frc_ok`、`frc_status`、`icon_valid`、`yaw_score`。显示有单独的 FR 作业和匹配后结果状态；这些状态字节不能直接命名为最终 AF 选择。 |
| `debug/deepseek_pose_dispatch/fw/a1ii_cp/cpapp-b.bin` | `96b41427958c1b41` | `+0x19b156..+0x19b17c` 为同一个诊断分别传入 `ReID:0x2860`、`FR:0x1434`，是**参数大小而非模型或特征维数**。`+0x737c90/+0x737c9c` 引用 `AdamFrManager::setParametersFrMatchCommon`，所在函数构造并复制参数。`+0x76b416..+0x76b420` 比较匹配记录的索引/标识，失败路径 `+0x76b424/+0x76b430` 引用按 `human_tracklet.id` 缺失 `FrMatchResultRaw` 的诊断。`+0x1d6f04/+0x1d6f1c` 引用 `judgeSwapFaceRecord`，代码从两个记录读有符号字段并比较；`+0x1da490/+0x1da496` 引用优先脸状态变化日志。 |

两台新机还有 `RegistFaceJudger::importFrResult`、`calcFrcScore`、`forceSetParametersFrMatchTracked` 等名称/引用；它们把 FR 结果放进人物跟踪与优先处理的证据链，但本轮未串成单一实机轨迹，也没有恢复打分权重、角度门限、失败回退、登记特征的实际编码或匹配作业调度。两机 NBG 槽 15 名称 `sfr_tf_resnet_454-80-160-320-512-nose_NHWC` 可作继续追查的候选；**名字不是 FR 作业到槽号的绑定证据**。

## 在 AF 流程中的位置

1. **登记侧（独立于每帧 AF）**：用户确认中央框内的脸，机身保存登记资料；A7 IV 设置顺序，A7R V／A1 II 选择一个当前目标。公开证据表明保存了人脸数据，不确定内部存的是图像、特征还是两者。
2. **拍摄侧（⑤ 与 ⑥ 之间的支路）**：视觉检测先提供人脸/人物候选；注册匹配把现场可能是该人的线索交给 **⑥ 主体优先仲裁**。已验证的 FR 作业与状态不是 FDN 脸检出分数、XHD 候选分数或镜头合焦标志。候选/裁片到 FR 的生产者及 FR 到最终 AF 输出的完整连接保留为虚线。
3. **与 ReID 并列**：ReID 比较当前目标与轨迹外观，问「是不是上几帧那个人」；FR 问「是否匹配事先登记/当前选中的脸」。A1 II 的独立参数与 FR 结果消费为这种区分提供了固件证据。FR 可以与轨迹交互，但不应用它代替 ReID，也不应把 A7 IV 画成共享两台新机的 FR 作业。

建议实机验证时固定 AF 区域与焦距，用获得同意的两名参与者分别测试优先开/关、两人同时入区、交换当前识别目标、目标转身遮挡以及已锁定另一人。分别记录识别框、实际 AF 框和稳定合焦，不用单一画框说明整个链路通过。

## Sony 原厂资料

- A7 IV：[Face Memory](https://helpguide.sony.net/ilc/2110/v1/en/contents/TP1000616737.html)、[Regist. Faces Priority](https://helpguide.sony.net/ilc/2110/v1/en/contents/TP1000655382.html)。
- A7R V：[Face Memory](https://helpguide.sony.net/ilc/2230/v1/en/contents/TP1000669397.html)、[Regist. Face Priority](https://helpguide.sony.net/ilc/2230/v1/en/contents/TP1000669398.html)。
- A1 II：[Face Memory](https://helpguide.sony.net/ilc/2440/v1/en/contents/211h_face_registration.html)、[Regist. Face Priority](https://helpguide.sony.net/ilc/2440/v1/en/contents/211h_face_identification.html)。
