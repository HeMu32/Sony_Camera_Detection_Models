"""Parse the self-describing `.nab` scene-recognition model (NAB container).

The `.nab` file (only on ILCE-7RM5 / ILCE-1M2) is a serialized graph:
  - 8-byte header: 'NA\\0\\0' + 'I\\0\\0\\0'
  - u32 @0x08, u32 count @0x0c, then `count` 8-byte (offset,size) entries
  - node table of 64-byte name records, some followed by a 64-byte layer-type
    record ('ConvolutionReluPoolingLayer2', 'FullyConnectedReluLayer', ...)
  - the remaining bytes are tensor data / weights.

This tool emits `nab_graph.json` with every node record and the layer-type
records it finds.  It is a *structural* parse: it proves the container is
self-describing and gives the node/layer inventory needed for a host-side forward
pass.
"""
import json
import os
import struct
import sys

NAME_LEN = 64


def is_text(b):
    s = b.rstrip(b'\0')
    return bool(s) and all(32 <= c < 127 for c in s)


def parse(path):
    d = open(path, 'rb').read()
    magic = d[:4]
    version = d[4:8]
    n0 = struct.unpack_from('<I', d, 8)[0]
    count = struct.unpack_from('<I', d, 0x0c)[0]
    table = []
    for i in range(min(count, 64)):
        off = 0x10 + i * 8
        if off + 8 > len(d):
            break
        a, b = struct.unpack_from('<II', d, off)
        table.append((a, b))

    nodes = []
    off = 0x100
    while off + NAME_LEN <= len(d):
        chunk = d[off:off + NAME_LEN]
        if is_text(chunk):
            name = chunk.rstrip(b'\0').decode('latin1')
            # a following 64-byte text record is a layer type
            nxt = d[off + NAME_LEN:off + 2 * NAME_LEN]
            kind = 'node'
            if is_text(nxt):
                kind = 'layer'
                typ = nxt.rstrip(b'\0').decode('latin1')
            else:
                typ = None
            nodes.append({'offset': hex(off), 'name': name, 'kind': kind, 'type': typ})
            off += NAME_LEN
        else:
            off += 4
    return {
        'file': os.path.basename(path),
        'size': len(d),
        'magic': magic.hex(),
        'version': version.hex(),
        'u32_08': n0,
        'count': count,
        'table': [{'offset': hex(0x10 + i * 8), 'a': hex(a), 'b': hex(b)} for i, (a, b) in enumerate(table)],
        'nodes': nodes,
        'layer_types': sorted({n['type'] for n in nodes if n['type']}),
    }


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
    out = {}
    for cam, sub in (('7RM5', 'r5'), ('1M2', '1m2')):
        p = os.path.join(root, sub, 'ai_dict', 'default.nab')
        if os.path.exists(p):
            out[cam] = parse(p)
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nab_graph.json')
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for cam, g in out.items():
        print(cam, g['size'], 'nodes', len(g['nodes']), 'layers', len(g['layer_types']))
        print('  types:', g['layer_types'])
    print('wrote', dst)


if __name__ == '__main__':
    main()
