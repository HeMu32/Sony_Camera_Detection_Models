# A7R V / A1 II：固件置信度门限

两机 CP 虚表的 RTTI 都是 `Camera::FC::Keypoint::TuningParamPostResult`。
虚表首个方法的文件偏移为 A7R V `+0xc2fb90`、A1 II `+0xd1f9a8`。
从两张原虚表的 `+8/+c/+10/+14` 调用实际 getter：

| 原厂 getter | A7R V 文件偏移 | A1 II 文件偏移 | 基值无效时的回退 |
|---|---|---|---|
| `getFaceConfidenceMidThreshold` | `0x1bf654` | `0x1f64cc` | `0.60` |
| `getFaceConfidenceHighThreshold` | `0x1bf784` | `0x1f6678` | `0.98` |
| `getXhdConfidenceMidThreshold` | `0x6e7fee` | `0x786eb0` | `0.50` |
| `getXhdConfidenceHighThreshold` | `0x6e8010` | `0x786ef8` | `0.50` |

原厂名称来自 A1 II 实际进入诊断分支时引用的符号字符串。只截获
该日志调用，保留原指令执行的选表、读取、范围检查和 VFP 运算。
A7R V 用相同的类、虚表位置、字段读取与返回位模式交叉确认。

- `DetectedTarget` `0..8` 经原表选块 `[0,0,1,2,2,2,3,3,4]`，块间距
  `0x600`。首块地址 A7R V `0x0172424c`、A1 II `0x01739110`，
  各自比 [区域缩放配置](CP-REGION-SCALE.md) 首块高 `0x1d0`。
  这是 CP 配置索引，未绑定到 NBG 槽号。
- 每项由两个原 u16 字段组成。基值 `1..10000` 除以 `10000`，其他值
  走上表的回退值。校正值 `1..9999` 除以 `100000000` 后加到基值，
  否则加零。原程序做 FP32 除法及加法，不在末尾钳制到 `1.0`。
- 人脸灵敏度 `0..6` 取字段组 `[2,0,1,2,3,4,2]`；实测 `7` 走无字段
  读取的返回零分支。XHD 优先用机内 `XhdType` 去索引参数组，再用
  部位和灵敏度从两张原表查字段位移。在已验证的类型索引 `0..8` 中，
  组为 `[0,0,0,1,2,0,1,0,1]`，每组四个部位索引；灵敏度 `6/7`
  走 `3` 的配置。数值枚举不是已验证的用户界面名称。

**这些是原固件的回退与解码规则，不是检测时实际装载的配置。**
五个配置块位于文件映射之外；未从发布包找到经验证的写入来源，
也未把 CP 比较使用的分数与 NBG 张量的类别、变换和归一化逐项绑定。
不能凭这些值直接替换主机的实验性画框阈值。

```powershell
$py = 'E:/anaconda3/envs/test311/python.exe'
& $py -B tools/probe_sony_cp_confidence.py extracted/r5/cp/cpapp-b.bin
& $py -B tools/probe_sony_cp_confidence.py debug/deepseek_pose_dispatch/fw/a1ii_cp/cpapp-b.bin --target 8 --sensitivity 6 --xhd-type 4 --part 3 --fixture-fields 0 1000
& $py -B -m unittest discover -s tools -p 'test_sony_cp_*.py' -v
```
