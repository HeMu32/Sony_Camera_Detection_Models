# A7R V / A1 II: original Adam JNP interface

JNP is not the Adam XHD box decoder. The original ARM controller and driver
feed a hardware register window, and a later CPU routine repacks the result.
This probe runs the original instructions against synthetic registers and
records. It does not perform pose extraction or infer a box from a photo.

| Adam file offset | A7R V | A1 II |
|---|---:|---:|
| Controller method and command switch | `0x164ac`, `0x16500` | `0x16854`, `0x168a8` |
| Low-level parameter register writer | `0x34a18` | `0x41bc4` |
| Read 24 peak counts | `0x34e80` | `0x4202c` |
| Read pose count | `0x34ff0` | `0x4219c` |
| CPU result packing | `0x161b4` | `0x1655c` |

The diagnostics `JNP_CTRL_E` and `JNP_DRV_E` are referenced in the original
images: for example, R5 `0x165b8`, `0x3456c`, `0x347e0` and `0x34828`
materialize the original strings with ARM `MOVW/MOVT`. Both driver tables
select JNP base `0x03003000` and the same six preceding poll registers.

- The writer validates the high nibble `9` at `+0x47`, two 7-bit
  dimension-minus-one fields (width `8..80`, height `2..80`, area divisible
  by 16), part indexes, and seven 8-byte-aligned
  addresses. Its 47 word writes copy matching offsets from the input into
  JNP MMIO. Tests use `64x42` (both slot-1 output grids) and `52x36` (R5
  slot-12 output grid) as fixtures. The fixtures alone do not identify
  which model feeds this block.
- The peak reader extracts 24 low five-bit fields from `+0xa8..+0xbf`.
  The pose reader extracts bits `23:16` of MMIO `+0x0c`. A held-busy fixture
  returns status `2` and clears the two outputs.
- The CPU packer emits a four-byte header and one `0x68`-byte record per
  pose when both its tag and count are nonzero. Each record starts with four
  bytes derived from the third trailing source word, four 7-bit values
  extracted from the preceding two words, then up to 24 raw joint words
  (zero-padded). Both source record stride and the joint count come from
  controller fields. The source words in these tests are synthetic.

The CP `Jnp Result dump` consumer reads the four 7-bit values in
`(left, top, right, bottom)` order. The original ARM/Thumb slices
(`0x1ad06c..0x1ad0ce` R5, `0x1e093c..0x1e099e` A1 II; **file offsets**)
take two canvas dimensions from the geometry block at `+0xa` and `+0xc`.
They scale the horizontal origin and span by `width / 64` and the vertical
origin and span by `height / 42`, flooring the origin and span separately.
The probe converts native `x,y,width,height` to half-open `xyxy` and replays
the geometry for up to the smaller of the first two header bytes. The
tests connect the original Adam packer directly to both original CP
projections and compare three distinct boxes, four canvas sizes and empty
results. This is canvas geometry, not yet full CP publication or a
JPEG-to-camera-canvas transform.

## Firmware Defaults

The CP `PostureEstimationParam` path writes the actual `0xec`-byte operation-6
payload at configuration buffer `+0x530`. R5 `0x1ac71c` calls initializer
`0x6d8a44`; A1 II `0x1dffec` calls `0x774a68` (all file offsets). Original
pointer translation and descriptor stores write three `(address, operation)`
records at `+0x61c`: `(+0x74, 2)`, `(+0x530, 6)`, `(+0x634, 2)`. The
probe supplies the allocation bases, extracts the bytes pointed to by the
second record, and calls the original Adam command-6 preparation
(`0x16098` / `0x16440`) and JNP register writer. It never starts the device.

The two images generate byte-identical operation-6 data and 47 word writes.
The dimensions are `64x42`, the configured part count is 19, and the pose
limit used by CPU output packing is 10. The following are original raw
values, not thresholds fitted to images:

| JNP offset | Bytes or words actually written |
|---|---|
| `+0x28` | Four bytes `25,25,25,25`; changing each CP byte at `+0x67..+0x6a` in isolation changes exactly the corresponding register byte |
| `+0x2c` | Four bytes `0,5,5,5` |
| `+0x30..+0x37` | Four halfwords `460,921,921,921` |
| `+0x3b` | Output limit `10`, forwarded into the CPU result header |
| `+0x40`, `+0x44` | `0x00660002`, `0x902b00c8` |

This establishes numerical values for this specific posture branch, not the
meaning and scaling of every threshold field, or the tuning for cat/animal.
The matching 4/4/4 diagnostic fields are named `heat_th`, `vec_th` and
`cost_th`; this correspondence does not establish their comparators or units.

## Slot-1 Buffer Handoff

The probe executes the original CP procedure builder (14 descriptors) and
reads the NPU parameters at configuration `+0x11c`. Its first four
`(procedure, operation)` pairs are `(0,2)`, `(1,2)`, `(2,6)` and `(0x10,0)`;
the slot-1 descriptor precedes the JNP descriptor. It runs the original Adam
descriptor reader at R5 `0x1bfe4` or A1 II `0x212b4`, stopping immediately
before the backend call. Both hand off graph ID `1`, the on-disk slot ID of
the byte-identical `default.nab` graph (SHA-256
`a5d5ead9478e3b069598483bda95702283d7b0c3d825de02c67250eeea4394b6`).

| Argument | Address | Descriptor capacity | Saved SDK tensor bytes |
|---|---:|---:|---:|
| Input 0 | `0x0ff00000` | 516096 | |
| Output 0 | `0x13100000` | 158592 | 107520 (`64x42x40`) |
| Output 1 | `0x13180000` | 158592 | 51072 (`64x42x19`) |

The R5 backend visits the output list in order (`0x68068..0x680cc`) and
copies each address and capacity into the buffer creation arguments
(`0x681dc..0x68204`). The index passed at `0x680bc..0x680cc` selects the
backend output descriptor at `0x217a8`. All offsets refer to Adam files.
The capacity is not the tensor footprint. The two stored SDK tensors total
158592 bytes, but each original descriptor independently declares 158592.

For both cameras the original JNP register writer uses output 0 at `+0xcc`
and output 1 at `+0xc8`. `+0xd0` is output 0 plus `0xa80` bytes, and
`+0xdc` is `0xa80`, the bytes in one `64x42` plane. Flipping bit 3 in each
command-6 word at `+0x3c..+0x5c` preserves address alignment. Seven such
changes each flip exactly one corresponding register bit; `+0x54` and
`+0x5c` have no effect on the 47 writes in this slice. A bit-0 perturbation
can instead make the alignment validator reject the request, and is not
evidence of a mode flag. The replay establishes address aliasing and a
one-plane offset, not channel ownership, hardware signedness, internal
normalization or part association. No JNP arithmetic was executed. The
heatmap grouping remains experimental; this replay does not generate a
detection.

```powershell
$env:PYTHONPATH = '.'
& 'E:/anaconda3/envs/test311/python.exe' -B -m unittest discover -s tools -p 'test_sony_adam_jnp.py' -v
& 'E:/anaconda3/envs/test311/python.exe' -B tools/probe_sony_adam_jnp.py extracted/r5/adam/adam_app.bin --cp-defaults extracted/r5/cp/cpapp-b.bin
```
