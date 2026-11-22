# A1 II: original Auto subject-recognition schedule and region dispatch

## Executed Region Dispatch

The original Thumb code distinguishes two arrays:

| Record | Address or pointer | Relevant fields |
|---|---|---|
| Schedule entry (`0x24` bytes) | VA `0x0170348c + 0x24 * index` | IP byte at `+0x1c`, enable byte at `+0x20` |
| Region (`0x20` bytes) | Pointer returned by cached-region helper | `u16 cx, cy, w, h` at `+8/+0xa/+0xc/+0xe`; IP byte at `+0x1c` |

Original `NEXT_IP` (`0x1c1b98`, file offset) reads the current index from
`state+0x8a24`, skips disabled entries, and wraps to the first enabled entry.
From `0x1c19bc`, the original dispatch reads one chosen schedule entry.
For the tested IPs, `0x757e44` enters `0x1c594c`; with the cached region
branch enabled by the fixture state, `0x1c59d0` stamps the chosen IP into
every region. The original caller invokes virtual slot `+0x40` once at
`0x1c1a68` with the `(count, region_pointer)` pair and the translated selector.
The loop at `0x1c1aa8` is diagnostics only; it does not submit a job per region.

The bounded replay supplies two cached regions and a three-entry schedule
with IP 3, one disabled entry, and IP 4. Original code dispatches
schedule index 0 with selector 4 and index 2 with selector 5, in separate
calls, then wraps to index 0. Both calls preserve both region geometries
and tag all regions with the same current IP. A one-entry control with IP
2 uses selector 1 and wraps to itself.

```powershell
& 'E:/anaconda3/envs/test311/python.exe' -B -m unittest discover -s debug/deepseek_dispatch/a1ii_auto -p test_region_dispatch.py -v
```

This establishes a rotating schedule and one-IP-per-call region dispatch,
not one-IP-per-frame: the replay does not execute the outer frame driver,
the virtual backend, or region generation. The IP-to-NBG mapping and
whether distinct IP results are submitted or published together in a
single frame remain unverified. References to a “per-frame list” below
describe the earlier static hypothesis, not an executed mixed-class frame.

## Executed XHD Backend Routes

The `AutoRecogParamManager` RTTI (`0xc6efe0`) points to the vtable at
file offset `0xd1c9d0`. The original constructor (`0x755e6e`) installs
this vtable and initializes the driver pointer array at `this+0xa20`.
The original selector context initializer `0x75583e` stores the result
of `0x75615a` into its first word. That getter returns the same global
manager (`0x013cfb88`) initialized at `0x756536..0x756538`.
Thus the selector's virtual `+0x40` resolves to `0x1c3454`.

In a second bounded replay, each captured dispatch is passed into that
native backend. It reads the ROI IP byte at `+0x1c`, builds the region
group and selects `this[0xa20 + 4*IP]`. The replay stops at the actual
virtual parameter-view call (`0x1c3788`). Then the original selected
driver initializes an explicit host allocation and writes the NPU
graph field at `block+0xa8` (`0x1f04e8..0x1f04ea`) and the original
three descriptors `(0,2,+0)`, `(1,2,+0xa8)`, `(0x10,3,+0x4bc)`.

| IP | Selector | Object offset | Driver and NPU graph | Constructor target | Config VA |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | `+0x5c0` | 8 | 1 | `0x01738f40` |
| 3 | 4 | `+0x6d0` | 11 | 4 | `0x01739b40` |
| 4 | 5 | `+0x8f0` | 9 | 5 | `0x0173a140` |
| 5 | 6 | `+0x7e0` | 10 | 6 | `0x0173a740` |

This table is an executed mapping, not a filename or class-name inference.
It supersedes the earlier suggestion below that an IP number could be
used as a graph ID. The BK4 loader associates the four VAs with
`0x60011`, `0x60013`, `0x60014`, `0x60015`, respectively. In particular,
the stage delivery's explicit graph-10/target-4 experiment does not use
graph 10's constructor-selected Auto configuration. The replay neither
sets the image parameters nor submits a job; one-frame cross-class
execution and UI publication remain unverified.

Offline analysis only. Inputs are the pinned local images listed in §1 and a host-side
Unicorn replay of original Thumb slices on explicit fixtures. No camera body, no NPU/VIP9000,
no alternative network, no FP64 CUDA, no remote target, no network exploit. New artifacts
live only in `debug/deepseek_dispatch/a1ii_auto/`.

## 0 Verdict

1. **The two halfwords are not a SA job and engine.** `docs/ADAM-XHD.md` is right:
   the CP short descriptor is `(proc_id = 0x10, operation = 3, block+0x4bc)`, and
   the Adam application CPU executes original XHD operation `3` for `proc_id 0x10`.
   The CP-side 8-byte record layout `<HHI>` is `(u16 op, u16 selector, u32 va)`; the
   middle field selects a *controller* (`engine`), it is not an SA job member number (§3.1).
2. **The Auto producer is in CP, not in the host merge.** One Auto frame is decided by
   `Camera::FC::Api::GetAdamModeRequest`/`GetAdamProcessRequest` on a mode value, then
   built into a region work list by `[AUTORECOG]SelectMode` + `SetParam ip_cnt/ip_num`
   (§3.2–§3.4). The host "run slot 8 and slot 10, merge the boxes" flow is a
   different thing: it is two explicit `(driver, target)` pairs (§4.4).
3. **The per-target published configuration is real BK4 data.** BK4 namespace 6 holds 27
   `TuningParamIP` records of `0x600` bytes; the XHD driver resolves a detection target to
   one key through the original binding tables, and target 3 and target 4 load *different*
   NMS bytes and a different `minimum_side` (§4.1–§4.3). Those are the actual settings,
   not strings and not probe fixtures.
4. **A manual-only selection is one target → one mode → one IP.** The original selector maps
   a single `Api::SelectedTarget` to a per-IP mode; targets 0/1/2 take the no-store branch,
   targets 3/4/5/6 store modes 4/5/6/0 (§3.5, replayed).
5. **The Auto decision is a 9-way switch, not "always multi".** Modes 0/1/2 return 0 and log
   `AutoRecog->SingleRecog`; modes 3/4/5 run the `validate` + `only_one_target` path; mode 6
   runs the keypoint-validate path; modes 7/8 run a `0x77d32c`/`0x77d464` check (§3.3,
   replayed).

Everything below is separated into **byte-verified**, **executed**, and **not established**.

## 1 Images and address convention

| image | path | bytes | SHA-256 |
|---|---|---:|---|
| A1 II `cpapp-b.bin` | `debug/deepseek_pose_dispatch/fw/a1ii_cp/cpapp-b.bin` | 19,190,024 | `96b41427958c1b412b3e15c5bd9b51e34657ae1e1246344438b8368031ca4859` |
| A1 II published backup | `debug/a1ii_fw/firmware.tar` → `0110_backup/SYSIPSX-DSLR/HC/CX89900_ALLLANG.bin` | 2,614,216 | (member) |
| A1 II BK4 second image | same member, `published_pair()[1]` at member offset `0x100000` | 829 namespaces / 33,218 items | — |

CP `cpapp-b.bin` is a flat ARM image: the startup stub at file `0x0` is
`e59f1004 / e2811004 / e1a0f001` with the linked literal `0x10800c`, so
**VA = file offset + 0x108000**. Every CP offset below is a **file offset**.
`debug/deepseek_dispatch/a1ii_auto/scan.py` builds a full reference index
(21,159,947 bytes of refs, ADR / MOVW+MOVT / literal-pool / BL) over the image.

## 2 The Adam XHD CPU job (restated, byte-checked)

`tools/probe_sony_adam_xhd.py` already runs the original Adam CPU decoder for
`proc_id 0x10`, operation `3`. The CP side that feeds it:

| item | A1 II file offset | bytes |
|---|---:|---|
| `AdamXhdDriver` vtable | `0xd1f2b4` | 9 Thumb slots, RTTI name at `0xc70c20` |
| `AdamXhdDriver` constructor | `0x1ef3d4` | `str.w r4,[r5,#0xc4]` (block), `bl 0x1ef450`, `str.w r0,[r5,#0x108]` |
| `AdamXhdDriver::Initialize` | `0x1ef450` | writes the shared block fields |
| `get_procs` | `0x1ef4ed` | writes `count` + N 8-byte `<HHI>` records |
| `set_parameters` | `0x1ef5fd` | — |
| `parameter_view` / `result_view` | `0x1f0091` / `0x1f00f1` | `(block+0, 0x520)` / `(block+0x520, 0x260)` |

Executed with the existing probe (A1 II, driver 8, target 3, published backup):

```
short_schedule  [16, 3, 0x4bc]        # (op=0x10, selector=3, va=block+0x4bc)
long_schedule   [0, 2, 0], [1, 2, 0xa8], [16, 3, 0x4bc]
nms_id          0                      # driver 8; 9/10/11 -> 2/3/1
+0x510           0x6504b030 = input1 + 0x32030
```

So `0x10` is the XHD procedure id, `3` is the *controller* selector, and the
descriptor's `va` is the shared block. This matches `docs/ADAM-XHD.md`; the host
"two halfwords = SA job + engine" reading is not supported by these bytes.

**Native shared-buffer dispatch (byte-verified):** the block is written by
`Initialize`/`set_parameters` and read by the DSP member through `parameter_view`
(`block+0`..`block+0x520`), with the result window `block+0x520`..`block+0x780`.
The per-frame native list is a *separate* 0x24-byte record array written by
`SetParam ip_cnt/ip_num` (§3.4), not the `<HHI>` array.

## 3.1 Why `(0x10, 3)` is not a SA job

`Adam::cmd_exec_common` (`0xdf00c0`), `Adam::RequestQueueManager`
(RTTI `0xb83380`), `Adam::SequenceQueueManager` (RTTI `0xb83470`),
`Adam::CmdInfo` (`0xeedfb3`), `Adam::REQUEST_ID` (`0xeed249`) are a different
namespace from the CP `get_procs` `<HHI>` array. The only shared contact point is
UIPC service `0x00930100` (`ddl_saCalcStart`, A7R V `0x197780`). The `engine`
field of the `<HHI>` record is consumed by `ddl_saGetExecUnit` (`0x167ec0`,
`0xb6` request), which returns one byte; nothing in the pinned images maps
`engine == 3` to an SA package member.

## 3.2 CP mode producer: `Adam::AdamMode` and the `mode == 5` gate

`Camera::FC::Api::GetAdamProcessRequest` (file `0x1374e4`) and
`Camera::FC::Api::InitAdamProcess` (file `0x137774`) both gate on the same
`Adam::AdamMode` value:

```
0x1375b2  ldr r0,[r4]        ; r4 = &AdamProcessRequestInfo
0x1375b4  cmp r0,#5
0x1375b6  bne 0x137636      ; not AdamMode 5 -> *out=0x1000, return 0
```

`InitAdamProcess` then copies a 16-byte record from global `0x1676450` into
`0x139a810` and stores `[r4+0x10]` back into the global:

| offset | field | log string |
|---|---|---|
| `0x00` | `SW` | `CC: InitAdamProcess():SW=%x, mode=%x, seq_mode=%x, frc=%x, fps=%x, exe=%x, img.addr=%x` (`0xf18568`) |
| `0x04` | `mode` | `CC: GetAdamModeRequest():SW=%x, cam_mode=%x, ezoom=%x, adam_addr=%p` (`0xf18425`) |
| `0x08` | `seq_mode` | `CC: GetAdamProcessRequest():SW=%x, mode=%x, seq_mode=%x, pre_seq_mode=%d, …` (`0xf18469`) |
| `0x0c` | `pre_seq_mode` | `CC: GetAdamProcessRequest():REQUEST_YES,…` (`0xf184e2`) |
| `0x10` | `pre_frc` | — |
| `0x14` | `new_frc` | — |
| `0x18` | `pre_fps` | — |
| `0x1c` | `new_fps` | — |
| `0x20` | `exe` (executable id) | `CC: InitAdamProcess():… exe=%x, img.addr=%x` (`0xf18568`) |

`exe` is the concrete executable id in the per-frame Adam request; it is an argument of
`InitAdamProcess`, not a graph index and not a menu item.

The command count per mode is bounded by `GetAdamCmdID` (`0x1c2b62`-`0x1c2bce`):

```
[GetAdamCmdID] seq=%d                       (0x1c2c7c)
ERROR: GetAdamCmdID sequence_num:%d > max:%d at AdamMode:%d   (0xf3b143)
```

and `[AUTORECOG]SetSequenceNumber seq:%d, exe:%d, is_skip:%d, ret:%d`
(`0x1c2b84`, string `0x1c2b84`) is the per-frame per-executable record: `seq`,
`exe`, `is_skip`, `ret`. That is the closest observed producer of a *per-executable
work list entry*.

## 3.3 The Auto switch (executed)

`[AUTORECOG] mode status auto:%d keypoint:%d` (`0x19ec20`) is logged by the wrapper
at `0x19ea88`, which calls the original 9-way switch at `0x19e9c4`:

```
0x19e9c4  push {r4,r6,r7,lr}
0x19e9ca  ldr r0,[r0]        ; r0 = *(int *)param
0x19e9cc  cmp r0,#8
0x19e9ce  bhi 0x19ea5c      ; > 8 -> default
0x19e9d0  tbb [pc,r0]        ; 9-entry table at 0x19e9d4
```

Table bytes at `0x19e9d4`: `4a 4a 4a 05 05 05 1c 26 39 00`
→ targets `0x19ea68 0x19ea68 0x19ea68 0x19e9de 0x19e9de 0x19e9de 0x19ea0c 0x19ea20 0x19ea46`.

Replayed (`replay.py`, executed on the pinned image with fixed stubs); the `case`
column is executed, the path and return columns are byte-verified in §3.3's case bodies:

| mode | case (executed) | path (byte-verified) | returns |
|---:|---|---|---:|
| 0,1,2 | `0x19ea68` | `*0x11e8bd4 = 3`, `*0x19b5892 = 0`, return | 0 |
| 3,4,5 | `0x19e9de` | `SonyLog` + `0x19e5c8` validate + virtual `+0x14` with 6, then `0x19e64c` + virtual `+0x14` with 7 | 1 |
| 6 | `0x19ea0c` | `0x19e64c` keypoint validate + virtual `+0x14` with 7 | 1 |
| 7 | `0x19ea20` | `0x77d32c` (returns `0x0143ebc1`) + `0x77d464` | 1 |
| 8 | `0x19ea46` | virtual `+0x2c` + `*0x11e8bd4 = 3` | 0 |

The wrapper then writes global byte `0x19b5892` and logs
`[AUTORECOG] ** AutoRecog->SingleRecog!! **` (`0x19ec78`) when the switch
returns 0. So modes 0/1/2 (and 8 in this fixture) collapse to single-target
recognition; modes 3–7 stay on the Auto path. `[AUTORECOG]set only_one_target:%d`
(`0x19ec54`) is logged from `0x19eb0e` in the same wrapper, with the value that
`0x75492a` returns; that value is then used to index a 6-entry table at VA
`0x00d74b60` (`0x2c16a 0x1f51f1a 0x2c16b 0x1f51f1c 0x2c16c 0x1f51f1e`), i.e. the
wrapper does have a "one target vs several" branch, but which modes reach it and
what the six entries mean is **not established** (the replay stubs `0x75492a` to 0).

**Not established:** what the *numeric* mode means at the menu level, and whether the
Auto flag reaches this switch as `auto == 1` from the settings. The switch input is a
field of an object, not a BK4 key (§4.4).

## 3.4 The per-frame region work list (byte-verified)

`[AUTORECOG]INFO: SetParam ip_cnt: %d, ip_num:%d` (`0x1c1b44`) is logged from
`0x1c1a6a`-`0x1c1a88`, and the loop that follows builds the per-region
records:

```
0x1c1a6a  ldr r0,[r4]                 ; ip_num (count)
0x1c1a6c  ldr.w r1,[sb]                ; sb = &state
0x1c1a74  adr r2,#0xcc                   ; "[AUTORECOG]INFO: SetParam ip_cnt: %d, ip_num:%d"
0x1c1a80  bl 0x1b7488                      ; SonyLog(2, 0xd, 0x129, file, fmt, ip_cnt, ip_num)
0x1c1a8c  ldr r0,[sp,#0x18]                  ; guard: region array non-null
0x1c1a90  ldrb.w r0,[sp,#0x28]                 ; guard: count non-zero
0x1c1a96  movw r8,#0x2b81                     ; log file "auto_recog_mode_selector.cpp"
0x1c1a9a  movw r4,#0x128f                       ; log file 0x0104128f
0x1c1aa8  ldr.w r0,[sb]                            ; sb = state
0x1c1aac  ldr r2,[sp,#0x2c]                          ; region array base
0x1c1aae  uxtb r1,r6                                   ; index
0x1c1ab4  add.w r1,r2,r1,lsl #5                            ; stride 0x20
0x1c1abc  ldrh r2,[r1,#8]                                   ; cx
0x1c1abe  ldrh r3,[r1,#0xa]                                   ; cy
0x1c1ac0  ldrh r5,[r1,#0xc]                                    ; w
0x1c1ac2  ldrh r1,[r1,#0xe]                                     ; h
0x1c1ac4  ldrb r0,[r0,#0x1c]                                      ; ip
0x1c1ad2  mov.w r2,#0x132                                        ; line 0x132
0x1c1ada  bl 0x1b7488                     ; "[AUTORECOG] ip:%d region cx:%d cy:%d w:%d h:%d"
0x1c1ae2  adds r6,#1 / cmp r0,r1 / blo 0x1c1aa8   ; next region
```

The region record is **0x20 bytes**, and the loop reads its
`+0x08 cx`, `+0x0a cy`, `+0x0c w`, `+0x0e h` (`u16`). The log's `ip`
comes from the **schedule entry**, not the region; `0x1c59d0` also
independently writes that IP to the region byte at `+0x1c`.
The 4-entry class table at VA `0x00e2378c` (file `0xd1b78c`) is `[4, 5, 6, 0]`, i.e.
`ip 3 -> class 4`, `ip 4 -> class 5`, `ip 5 -> class 6`, `ip 6 -> class 0`.

**The file is `auto_recog_mode_selector.cpp`.** The same function passes
`0xf3928f = "auto_recog_mode_selector.cpp"` as the log file name at `0x1c1a70`,
and `0xf3ab81 = "[AUTORECOG] ip:%d region cx:%d cy:%d w:%d h:%d"` is a **live**
string referenced at `0x1c1a96`. So the region list is produced in the Auto mode
selector, not in the XHD driver and not in the host merge.

A second 7-entry selector follows at `0x1c16d8` (table bytes
`04 30 34 36 38 3a 3c 3e 00`, targets `0x1c16e4 0x1c173c 0x1c1744 0x1c1748
0x1c174c 0x1c1750 0x1c1754 0x1c1758`), switching on a byte at `[fp+0xc]` and
writing one byte at `[r6+0x32]` per mode. That is the mode→mode translation the
`getModeInfo5SA` string describes; its target bodies were not replayed.

The list is then committed per IP by `[AUTORECOG]SetParam_region(id:%x)`
(`0x1c3b28`) and `[AUTORECOG]SetParam_region(id:%x) in ADAM_MNG_STATUS_END`
(`0x1c3b4c`), and the per-IP Adam parameters are
`[AUTORECOG][IP%d] ipdrv:0x%X param_adam:0x%X res_adam:0x%X`
(`0xf3cfda`, dead copy) — the live copy is the same message family in
`0x19dxxx`-`0x1c5xxx`.

**Observed region consumer:** `0x1c19c0` begins preparation for a single
virtual `+0x40` call, using the selected `0x24`-byte schedule entry.
The later `0x1c1aa8` loop logs the `0x20`-byte regions. Neither code
slice establishes that IP is a graph or driver ID.

## 3.5 Manual-only selection (executed)

`[AUTORECOG]SelectMode targets:%d` (`0x1c54e8`) is logged from `0x1c54a8` in the
selector at `0x1c55e4`:

```
0x1c55ec  mov sb,r3         ; sb = out2
0x1c55ee  mov r4,r2          ; r4 = target
0x1c55f0  mov r8,r1           ; r8 = out
0x1c55f2  bl 0x7574f6          ; per-frame object
0x1c560c  cmp r4,#6
0x1c560e  strb.w r4,[r8,#4]       ; *(out+4) = target
0x1c5618  tbb [pc,r4]               ; 7-entry table at 0x1c561c
```

Table bytes at `0x1c561c`: `04 04 04 2f 37 32 35 00`
→ targets `0x1c5624 0x1c5624 0x1c5624 0x1c567a 0x1c568a 0x1c5680 0x1c5686`.

Replayed (executed):

| target | case | mode written | returns |
|---:|---|---:|---:|
| 0,1,2 | `0x1c5624` | no store (exits at `0x1c568e`) | 0 |
| 3 | `0x1c567a` | 4 | 1 |
| 4 | `0x1c568a` | 5 | 1 |
| 5 | `0x1c5680` | 6 | 1 |
| 6 | `0x1c5686` | 0 | 1 |

So a manual selection of one target produces exactly one per-IP mode, and targets
0/1/2 produce no mode at all in this fixture (their case body iterates a region
array and exits when it is empty). This is the "manual-only" shape: one target → one
mode → one IP, whereas the Auto path (§3.3 modes 3–7) is the multi-IP path.

**Comparison with the host two-job merge:** the host flow ran driver 8 + target 3 and
driver 10 + target 4 as two independent `AdamXhdDriver` jobs and merged the boxes.
That is two manual selections, not one Auto frame: the original selector would have
produced one mode per target, and the per-frame region list would carry the `ip` values
of the enabled categories.

## 4 BK4: the actual published settings

### 4.1 Namespace and keys

`parsers/sony_bk4.py` parses the published backup. The A1 II second image has
**829 namespaces and 33,218 items**; namespace `0x6` holds **27 items**
(`0x60000`…`0x6001a`). The original binding tables in the CP image resolve a
detection target to one of those keys:

| table | A1 II VA | file offset | content |
|---|---:|---:|---|
| `TuningParamIP` destinations | `0x11e90c0` | `0xe90c0` | 15 pointers, 4 sentinels at the end |
| `TuningParamIP` identifiers | `0x11e910c` | `0xe910c` | `0x60000 0x60001 0x6000f 0x60010 0x60011 0x60012 0x60013 0x60014 0x60015 0x60016 0x60017 0x60018 0x60019 0x6001a 0x60003 0x60004 0x60005 0` |

`TuningParamIP::TuningParamIP(DetectedTarget)` (`0x1f6318`) takes the target in
`1..6`, rejects anything else with `invalid detection target=%d`, and stores
`table[target-1]` at `this[4]`; the six table entries are
`0x0001e608 0x01f5204f 0x0002e609 0x01f52050 0x0001e60a 0x01f52052`
(file `0xd74c70`).

### 4.2 Executed published load

`tools/probe_sony_cp_xhd_nms.py --published-backup` was run on the A1 II image with
the published member. It loads the record through the original BK4 reader
(`0xc7ffe6`, pointers `0x01964080`) into the original destination and checks the copy:

| target | driver | configuration id | destination VA | BK4 record offset | record SHA-256 |
|---:|---:|---|---:|---:|---|
| 3 | 8 | `0x60011` | `0x1738f40` | `0x38c64` | `1d5d7bfd19c09139fb8656629a83d00d25b1257f7a887d98e52949075046dec5` |
| 4 | 10 | `0x60013` | `0x1739b40` | `0x3986c` | `8e30cbe8a1accb86ca1a5e822d55d77e484d93b5ff8e4cc78e4ce672e79889e0` |

Both records are `0x600` bytes. Their NMS-relevant slice `+0x171..+0x18b`:

```
target 3: 80 4c 80 4c 00 4c 00 4c 00 00 80 4c 80 4c 00 4c 00 4c 00 00 e5 81 12 3c 00 01   minimum_side 0x12
target 4: 80 4c 80 4c 00 00 80 4c 80 4c 00 00 80 4c 80 4c 00 00 00 00 e5 81 09 3c 00 00   minimum_side 0x09
```

So the two targets carry **different** `thr`/`thr_per` pairs and a different
`minimum_side`: the difference is real BK4 data, not a probe fixture. `thr_glbl` is
`0xe5 = 229` and `bgfg_thr` is `0x81 = 129` in both, and `skip_max_decode` is
`0x01` for target 3 and `0x00` for target 4.

### 4.3 What BK4 does *not* provide

The published backup contains the **per-target** `TuningParamIP` records; it does not
contain a per-frame region list, a mode value, or a category→slot table. Those live in the
CP image (§3.3, §3.4) and in the `.rsc` slot tables (§4.4). The host cannot therefore
recover an Auto frame from BK4 alone; it can recover the per-target settings that an Auto
frame would apply to each `ip` it selects.

### 4.4 Category → slots, and why the host merge is not Auto

`extracted/1m2/ai_dict/*.rsc` (and `rsc_configs.json`) map `ai_model_category` to slots:

| category | subject | A1 II slots |
|---:|---|---|
| 0 | human | `{}`, 5+6, 14+15 |
| 1 | animal | 18, 19 |
| 2 | bird | 7, 8 |
| 3 | insect | 7, 8 |
| 4 | car | 7, 8 |
| 5 | train | 11, 17 |
| 6 | airplane | 9 |
| 7 | (A1 II extra) | 10, 16 |

The CP driver ids are `(8, 9, 10, 11, 16, 17, 18)` (probe `LAYOUT`), i.e. the same
numbers as the `.rsc` slots: `driver id == slot id` on this path. So a single Auto frame
that enables several categories must produce a **union** of slots (e.g. bird+insect+car = 7, 8;
airplane = 9; category 7 = 10, 16; train = 11, 17), which is exactly the multi-`ip` region
list of §3.4.

The host flow ran slot 8 (driver 8) and slot 10 (driver 10) as two independent jobs and
merged their boxes in one 320×320 coordinate space. That is two manual selections, not the
union: it does not produce the per-IP `region` records, does not go through
`SelectMode`/`SetParam ip_cnt/ip_num`, and does not use the `ip` class table. It also
re-uses the same input frame for both graphs instead of the per-region crops the firmware
would build.

## 5 Bounded replay

`debug/deepseek_dispatch/a1ii_auto/replay.py` executes two original slices and nothing else:

```
& 'E:/anaconda3/envs/test311/python.exe' -B replay.py     # writes replay.json
& 'E:/anaconda3/envs/test311/python.exe' -B scan.py build  # reference index (~80 s)
& 'E:/anaconda3/envs/test311/python.exe' -B scan.py refs 0x1c54e8 0x1c1b44
```

It maps the pinned image, gives only the `0x11e8000` settings page write access, stubs
every external callee with a fixed fixture, and stops at the first instruction of each
boundary. It never submits a job, never touches a queue, and never runs the NPU path.
Observed output (`replay.json`):

```
auto_modes:      0,1,2 -> 0x19ea68 ; 3,4,5 -> 0x19e9de ; 6 -> 0x19ea0c ; 7 -> 0x19ea20 ; 8 -> 0x19ea46
selected_targets: 0,1,2 -> case 0x1c5624 (no store, returns 0)
                 3 -> case 0x1c567a, mode 4, returns 1
                 4 -> case 0x1c568a, mode 5, returns 1
                 5 -> case 0x1c5680, mode 6, returns 1
                 6 -> case 0x1c5686, mode 0, returns 1
```

The fixtures are explicit: the per-frame object's virtual `+0x14`/`+0x2c`/`+0x54`/`+0x58`
return 0, the keypoint pointer returns `0x0143ebc1`, and the logs return 0. The replay
therefore proves the *control flow*, not the numeric values.

## 6 Established vs not established

| item | status | evidence |
|---|---|---|
| `(0x10, 3, block+0x4bc)` is `(proc_id, operation, va)` | **executed + byte-verified** | §2, `probe_sony_cp_xhd_nms.py` |
| Adam CPU decoder for `proc_id 0x10`, op `3` | **executed** | `probe_sony_adam_xhd.py` |
| `AdamMode == 5` gate in `GetAdamProcessRequest`/`InitAdamProcess` | **byte-verified** | §3.2 |
| 9-way Auto switch and its targets | **byte-verified + executed** | §3.3, `replay.py` |
| `only_one_target` sits on modes 3/4/5 | **byte-verified** | §3.3 |
| `SetParam ip_cnt/ip_num` region list (0x24-byte records) | **byte-verified** | §3.4 |
| `ip` class table `[4,5,6,0]` | **byte-verified** | §3.4 |
| target→mode table (7 entries) | **byte-verified + executed** | §3.5 |
| BK4 namespace 6 keys and per-target records | **executed** | §4.1–§4.2 |
| target 3 vs target 4 NMS bytes differ | **executed** | §4.2 |
| driver id == `.rsc` slot id on this path | **inferred** (same number set) | §4.4 |
| Auto frame's *numeric* mode and per-frame `ip` list | **not established** | needs the settings writer and the region producer's caller |
| `AutoRecog`/`SearchModeSwitcher` → slot, ROI, frame interval | **not established** | unchanged from `docs/AF-RESEARCH-AUDIT.md` |
| `engine == 3` → SA package member | **not established** | `ddl_saGetExecUnit` only returns a byte |
| BK4 key that stores the Auto/menu setting | **not established** | namespace `0x3`/`0x5` not enumerated to a setting name |

## 7 Open items

1. **The region list producer's caller.** `0x1c19c0` builds the list from a state object
   at `sl`; the code that fills `ip_cnt` and the 0x24-byte records was not traced. That is
   the next concrete producer to close.
2. **The Auto mode's numeric value.** Modes 3–7 keep Auto alive; which mode the menu's
   `Auto` selects is not established.
3. **The BK4 setting key.** Namespace `0x3` (2,562 items) and `0x5` (1,643 items) are the
   candidates for the subject-recognition setting; no key was tied to a named setting.
4. **The `exe` id.** `InitAdamProcess(... exe=%x ...)` takes an executable id; its mapping to
   the XHD driver id or to the `ip` of §3.4 was not observed.
5. **Dead string copies are regional.** `debug/deepseek_pose_dispatch/REPORT.md` found the
   `0xf43000`–`0xf45000` block unreferenced. That is not the whole `0xf3xxxx`
   range: `0xf3928f`, `0xf3ab81` and `0xf39260`/`0xf39268` are **live** and referenced
   from the `0x1c1xxx` selector functions. Only quote a `0xf3xxxx` string after a refs
   query in `debug/deepseek_dispatch/a1ii_auto/scan.py`.
