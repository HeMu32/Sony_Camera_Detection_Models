"""尝试直接跑 .nab 图：152 个 conv(pad,kernel,stride) + 195 个张量。

张量记录 (id, count, offset, size, dtype, flag)；
权重在 0x5c48 + offset，长度 size。
通道数按 in=out 反推：in*out*k*k = size。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAB = os.path.join(HERE, 'work', 'r5', 'ai_dict', 'default.nab')


def parse_nodes(d):
    def printable(b):
        s = b.rstrip(b'\0')
        return len(s) > 0 and all(32 <= c < 127 for c in s)
    off = 0x7dc
    nodes = []
    while off < 0x4a00:
        t = d[off:off + 64]
        if not printable(t):
            off += 4
            continue
        p = off + 64
        while p < 0x4a00 and not printable(d[p:p + 64]):
            p += 4
        params = [int.from_bytes(d[off + 64 + 4 * i:off + 64 + 4 * i + 4], 'little')
                  for i in range((p - off - 64) // 4)]
        nodes.append((t.rstrip(b'\0').decode('latin1'), params, off))
        off = p
    return nodes


def parse_tensors(d):
    off = 0x4a00
    out = []
    while off + 24 <= 0x5c48:
        v = [int.from_bytes(d[off + 4 * j:off + 4 * j + 4], 'little') for j in range(6)]
        out.append(v)
        off += 24
    return out


def main():
    d = open(NAB, 'rb').read()
    nodes = parse_nodes(d)
    tensors = parse_tensors(d)
    convs = [n for n in nodes if len(n[1]) == 3]
    print('convs', len(convs), 'tensors', len(tensors))
    print('first conv params', convs[0][1], 'last', convs[-1][1])
    print('first tensor', tensors[0], 'last', tensors[-1])
    sizes = [t[3] for t in tensors]
    print('size min/max/sum', min(sizes), max(sizes), sum(sizes))
    print('weights region bytes', len(d) - 0x5c48)


if __name__ == '__main__':
    main()
