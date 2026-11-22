"""Structural parse of the Minna detection dictionaries (`/usr/sabin/*_dic*.bin`).

Facts recovered from the CP firmware (Thumb-2, Ghidra):
  - `Camera::Minna::{Fdn,Pdn,Scd,Aed,Bed}::*ParamManager::buildParam2`
    copies per-layer u16 fields (w,h,channels,kernel,stride,pad) from
    `*DrvParam` into the MMU parameter block; the dictionaries themselves
    hold quantised tensors only.
  - `fdn_*_dict.bin` is the only file that also carries the per-scale
    descriptor: 4 sections, each `[w, h, 6, 3, 4, 1, 1, a, a, h, w/6, 3, 4]`
    with windows 12x24 / 18x30 / 24x48 / 36x60.
  - `*_s2f.bin` carries float32 per-layer dequantisation scales.

This tool emits `dic_layout.json` so the host-side forward pass has the exact
tensor boundaries it needs.
"""
import hashlib
import json
import os
import struct

SABIN = {
    '7M4': os.path.join('out', 'models', 'work', '7m4', 'sabin'),
    '7RM5': os.path.join('out', 'models', 'work', 'r5', 'sabin'),
    '1M2': os.path.join('out', 'models', 'work', '1m2', 'sabin'),
}
FDN_HEADER_LEN = 13 * 4


def fdn_sections(d):
    """Return the 4 per-scale descriptors of an fdn dictionary."""
    out = []
    off = 0
    while off + FDN_HEADER_LEN <= len(d):
        v = struct.unpack_from('<13I', d, off)
        if 8 <= v[0] <= 128 and 8 <= v[1] <= 256 and v[2] == 6 and v[3] == 3 and v[4] == 4:
            out.append({'offset': off, 'w': v[0], 'h': v[1], 'layers': v[2],
                        'kernel': v[3], 'stride': v[4], 'fields': list(v)})
            off += FDN_HEADER_LEN
        else:
            off += 4
    return out


def s2f_scales(d):
    if len(d) % 4:
        return []
    vals = struct.unpack_from('<%df' % (len(d) // 4), d, 0)
    return [round(v, 6) for v in vals]


def describe(path):
    d = open(path, 'rb').read()
    rec = {'file': os.path.basename(path), 'size': len(d),
           'sha256': hashlib.sha256(d).hexdigest()}
    if 'fdn' in rec['file']:
        rec['sections'] = fdn_sections(d)
    if rec['file'].endswith('_s2f.bin'):
        rec['scales'] = s2f_scales(d)
    return rec


def main():
    layout = {}
    for cam, root in SABIN.items():
        items = []
        for name in sorted(os.listdir(root)):
            if name.endswith('.bin') and ('dic' in name or 'dict' in name):
                items.append(describe(os.path.join(root, name)))
        layout[cam] = items
    dst = os.path.join('out', 'models', 'dic_layout.json')
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(layout, f, indent=1, ensure_ascii=False)
    for cam, items in layout.items():
        print(cam, len(items), 'files')
        for it in items:
            if 'sections' in it:
                print(' ', it['file'], [s['fields'] for s in it['sections']])
            if 'scales' in it:
                print(' ', it['file'], it['scales'][:8])
    print('wrote', dst)


if __name__ == '__main__':
    main()
