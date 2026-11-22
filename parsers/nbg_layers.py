"""导出 15 个 NBG 图的全部层类型表 -> nbg_layers.json"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = (rb'ConvolutionReluPoolingLayer2', rb'PoolingLayer2',
         rb'FullyConnectedReluLayer', rb'TensorCopy', rb'TensorMul',
         rb'TensorSub', rb'TensorReduceSum', rb'ActivationLayer',
         rb'Concat', rb'Reshape', rb'TensorAdd')
PAT = rb'(' + b'|'.join(TYPES) + rb')'


def u32(d, o):
    return int.from_bytes(d[o:o + 4], 'little')


def parse_graph(path):
    d = open(path, 'rb').read()
    out = {'file': os.path.basename(path), 'size': len(d)}
    if d[:4] != b'VPMN':
        out['error'] = 'not VPMN'
        return out
    out['version'] = '0x%08x' % u32(d, 4)
    out['name'] = d[0xc:0x4c].split(b'\0')[0].decode('latin1')
    out['layer_count'] = u32(d, 0x4c)
    out['operation_count'] = u32(d, 0x50)
    out['input_count'] = u32(d, 0x54)
    out['output_count'] = u32(d, 0x58)
    out['core_count'] = u32(d, 0x60)
    out['mem_pool_align'] = u32(d, 0x160)
    out['axi_sram'] = u32(d, 0x16c)
    out['mem_pool_size'] = u32(d, 0x174)
    out['vip_sram'] = u32(d, 0x178)

    segs = ['nn table', 'tp table', 'ICD', 'coeff header',
            'coeff data', 'ppu instr', 'load states', 'dyna io']
    base = 0x17c
    l0 = []
    for i in range(8):
        size = u32(d, base + 8 * i)
        off = u32(d, base + 8 * i + 4)
        l0.append({'segment': segs[i], 'size': size, 'offset': off})
    out['layer0'] = l0

    layers = []
    for m in re.finditer(PAT, d):
        o = m.start()
        p = o + 64
        vals = [u32(d, p + 4 * i) for i in range(3)]
        layers.append({'offset': o, 'type': m.group(1).decode(),
                       'id': vals[0], 'kernel': vals[1], 'stride': vals[2]})
    out['layers'] = layers
    out['layer_types_found'] = len(layers)
    return out


def main():
    gdir = os.path.join(HERE, 'nab_graphs')
    files = sorted(f for f in os.listdir(gdir) if f.endswith('.nb'))
    res = [parse_graph(os.path.join(gdir, f)) for f in files]
    dst = os.path.join(HERE, 'nbg_layers.json')
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print('%-28s %8s %6s %6s %6s' % ('file', 'size', 'layers', 'ops', 'found'))
    for r in res:
        print('%-28s %8d %6d %6d %6d' % (
            r.get('name', r['file'])[:28], r['size'],
            r.get('layer_count', -1), r.get('operation_count', -1),
            r.get('layer_types_found', -1)))
    print('\nwrote', dst)


if __name__ == '__main__':
    main()
