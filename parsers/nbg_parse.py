"""NBG (VeriSilicon VPMN) 解析器 —— 目前只解到首层，作为可复现的证据。

用法: python nbg_parse.py <file.nb>
输出: NBG 头 + 首层 8 组 (size, offset) + 各段前 32 字节
"""
import json
import os
import struct
import sys

SEGS = ['nn table', 'tp table', 'ICD', 'coeff header',
        'coeff data', 'ppu instr', 'load states', 'dyna io']


def u32(d, o):
    return struct.unpack_from('<I', d, o)[0]


def parse(path):
    d = open(path, 'rb').read()
    out = {'file': os.path.basename(path), 'size': len(d)}
    if d[:4] != b'VPMN':
        out['error'] = 'not VPMN'
        return out
    out['version'] = '0x%08x' % u32(d, 4)
    out['header_d4'] = u32(d, 8)
    out['name'] = d[0xc:0x4c].split(b'\0')[0].decode('latin1')
    out['layer_count'] = u32(d, 0x4c)
    out['operation_count'] = u32(d, 0x50)
    out['input_count'] = u32(d, 0x54)
    out['output_count'] = u32(d, 0x58)
    out['field_5c'] = u32(d, 0x5c)
    out['core_count'] = u32(d, 0x60)
    out['field_64'] = u32(d, 0x64)
    out['mem_pool_align'] = u32(d, 0x160)
    out['axi_sram'] = u32(d, 0x16c)
    out['mem_pool_size'] = u32(d, 0x174)
    out['vip_sram'] = u32(d, 0x178)

    base = 0x17c
    if base + 64 > len(d):
        out['error'] = 'too short'
        return out
    pairs = []
    for i in range(8):
        size = u32(d, base + 8 * i)
        off = u32(d, base + 8 * i + 4)
        seg = {'segment': SEGS[i], 'size': size, 'offset': off}
        if off + size <= len(d) and size:
            seg['head32'] = d[off:off + 32].hex()
        pairs.append(seg)
    out['layer0'] = pairs
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nab_graphs')
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.nb'))
    results = [parse(p) for p in paths]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nbg_header.json')
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\nwrote', dst, file=sys.stderr)


if __name__ == '__main__':
    main()
