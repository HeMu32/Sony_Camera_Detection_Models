"""Execute original FDN stage-2 control code in an isolated host memory model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from probe_sa_fdn_dispatch import MEMBER_SHA, U16, check, executable_banks


MASK = (1 << 32) - 1
BLOCKS = {
    "A": (0x100, 0x1134, 0x1148, 0x1162),
    "B": (0x1a800, 0xfb4, 0xfc4, 0xfd6),
}


def signed(value, bits):
    return (value ^ (1 << (bits - 1))) - (1 << (bits - 1))


class Stop(Exception):
    def __init__(self, reason, **fields):
        self.detail = {"reason": reason, **fields}


class DispatchMachine:
    """A bounded SA-DSP subset; devices can receive writes but supply no reads."""

    float32_enabled = False

    def __init__(self, addresses, image):
        self.pc = int(addresses["stage_dsp_address"], 16)
        self.mailbox = int(addresses["mailbox_selector"], 16)
        self.reg = [0] * 16
        self.link = 0
        self.flag = False
        dst = int(addresses["first_transfer_destination"], 16)
        self.regions = ((dst, image),)
        if self.pc != dst:
            self.regions += ((self.pc, image[:0x34]),)
        self.stack_low = dst + len(image)
        self.stack_high = int(addresses["stack"], 16)
        self.stack = bytearray(self.stack_high - self.stack_low)
        self.mailbox_data = b""
        self.mmio_writes = []
        self.mailbox_reads = []
        self.pcs = []

    def read(self, address, size, *, fetch=False, sign=False):
        address &= MASK
        for base, data in self.regions:
            offset = address - base
            if 0 <= offset <= len(data) - size:
                return int.from_bytes(data[offset:offset + size], "little", signed=sign)
        if not fetch and self.stack_low <= address <= self.stack_high - size:
            offset = address - self.stack_low
            return int.from_bytes(self.stack[offset:offset + size], "little", signed=sign)
        offset = address - self.mailbox
        if not fetch and 0 <= offset <= len(self.mailbox_data) - size:
            raw = self.mailbox_data[offset:offset + size]
            value = int.from_bytes(raw, "little", signed=sign)
            self.mailbox_reads.append({"pc": self.pc, "address": address, "size": size, "value": value})
            return value
        raise Stop("unmapped fetch" if fetch else "unmodeled read", address=address, size=size)

    def write(self, address, size, value):
        address &= MASK
        value &= (1 << (size * 8)) - 1
        if self.stack_low <= address <= self.stack_high - size:
            offset = address - self.stack_low
            self.stack[offset:offset + size] = value.to_bytes(size, "little")
        elif 0x40000 <= address <= 0x4ffff - size + 1:
            self.mmio_writes.append({"pc": self.pc, "address": address, "size": size, "value": value})
        else:
            raise Stop("unmodeled write", address=address, size=size, value=value)

    @staticmethod
    def float32_result(op, dst, src):
        import numpy as np

        if op == 0x39:
            return int(np.float32(signed(src, 32)).view(np.uint32))
        rhs = np.uint32(src).view(np.float32)
        if op == 0x38:
            if not np.isfinite(rhs) or not np.float32(-(1 << 31)) <= rhs < np.float32(1 << 31):
                raise Stop("unmodeled float conversion", source=src)
            return int(rhs)
        lhs = np.uint32(dst).view(np.float32)
        operation = (np.add, np.subtract, np.multiply, np.divide)[op - 0x34]
        with np.errstate(all="ignore"):
            result = operation(lhs, rhs)
        if not np.isfinite(result):
            raise Stop("unmodeled float result", left=dst, right=src, operation=op)
        return int(result.view(np.uint32))

    def step(self):
        p, r = self.pc, self.reg
        if p & 1:
            raise Stop("unaligned instruction", address=p)
        code = self.read(p, U16.size, fetch=True)
        d, s, group = code & 15, (code >> 4) & 15, code >> 12
        op, sub = code >> 8, code >> 4
        a, b = r[d], r[s]
        target = p + 2
        immediate = signed((code >> 4) & 255, 8)
        if code == 0x0004:
            raise Stop("wfi")
        if code in (0, 1, 2):
            if code:
                self.flag = code == 2
        elif code == 3:
            target = self.link
        elif sub == 0x002:
            r[d] = int(self.flag)
        elif sub == 0x003:
            r[d] = p + 4
        elif sub == 0x007:
            self.flag = bool(a >> 31)
            r[d] = a << 1
        elif sub == 0x008:
            self.flag = bool(a & 1)
            r[d] = signed(a, 32) >> 1
        elif sub == 0x00a:
            target = a
        elif sub == 0x00b:
            self.link, target = target, a
        elif sub == 0x00c:
            self.link = self.read(a, 4)
            r[d] = a + 4
        elif sub == 0x00d:
            self.write(a - 4, 4, self.link)
            r[d] = a - 4
        elif 1 <= op <= 0x0c:
            kind, mode = divmod(op - 1, 3)
            size = (1, 1, 2, 4)[kind]
            r[d] = self.read(b + (r[0] if mode == 1 else 0), size, sign=kind in (0, 2))
            if mode == 2:
                r[s] += size
        elif 0x0d <= op <= 0x18:
            kind, mode = divmod(op - 0x0d, 4)
            size = 1 << kind
            self.write(a + (r[0] if mode == 1 else -size if mode == 3 else 0), size, b)
            if mode in (2, 3):
                r[d] += size if mode == 2 else -size
        elif op in (0x19, 0x1a):
            bits = 8 if op == 0x19 else 16
            r[d] = signed(b & ((1 << bits) - 1), bits)
        elif op in (0x1b, 0x1c, 0x1d, 0x1e, 0x20, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27):
            r[d] = {0x1b: b, 0x1c: b & 255, 0x1d: b & 65535, 0x1e: a + b,
                    0x20: a - b, 0x22: a * b, 0x23: a & b, 0x24: a | b,
                    0x25: ~b, 0x26: a ^ b, 0x27: -b}[op]
        elif op == 0x21:
            r[d] = a - b - int(self.flag)
            self.flag = r[d] < 0
        elif op == 0x28:
            r[d] = (-b - int(self.flag)) & MASK
            self.flag = self.flag or r[d] != 0
        elif op in (0x29, 0x2a, 0x2b):
            shift = min(b, 32)
            r[d] = a << shift if op == 0x29 else (
                a if op == 0x2a else signed(a, 32)) >> shift
        elif op in (0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33):
            self.flag = {0x2e: a == b, 0x2f: a >= b, 0x30: signed(a, 32) >= signed(b, 32),
                         0x31: a > b, 0x32: signed(a, 32) > signed(b, 32), 0x33: not a & b}[op]
        elif self.float32_enabled and 0x34 <= op <= 0x39:
            r[d] = self.float32_result(op, a, b)
        elif group == 4:
            r[d] = self.read(p + 4 + 2 * (immediate & 255), 2, sign=True)
        elif group == 5:
            r[d] = self.read((p & ~3) + 4 + 4 * (immediate & 255), 4)
        elif group == 6:
            r[d] = self.read(b + 4 * (code >> 8 & 15), 4)
        elif group == 7:
            self.write(a + 4 * (code >> 8 & 15), 4, b)
        elif group == 8:
            self.flag = a == immediate & MASK
        elif group in (9, 10):
            r[d] = (0 if group == 9 else a) + immediate
        elif group in (11, 12, 13):
            if group == 11 or (group == 12 and self.flag) or (group == 13 and not self.flag):
                target = p + 4 + 2 * signed(code & 0xfff, 12)
        elif code >> 9 in (0x78, 0x79, 0x7a):
            shift, kind = (code >> 4) & 31, code >> 9
            r[d] = a << shift if kind == 0x78 else (
                a if kind == 0x79 else signed(a, 32)) >> shift
        else:
            raise Stop("unsupported opcode", opcode=code)
        r[:] = [value & MASK for value in r]
        self.pc = target & MASK

    def run(self, limit, *, stop_at=None):
        for _ in range(limit):
            if self.pc == stop_at:
                return {"pc": self.pc, "reason": "slice boundary"}
            self.pcs.append(self.pc)
            try:
                self.step()
            except Stop as exc:
                return dict(pc=self.pc, **exc.detail)
        return {"pc": self.pc, "reason": "step limit"}


def front_end(blob, block, selector=None, limit=256):
    addresses = check(blob, *BLOCKS[block])
    src = int(addresses["stage_file_offset"], 16)
    image = blob[src:src + addresses["first_transfer_bytes"]]
    return run_dispatch(addresses, image, block, selector, limit)


def bank_entry(blob, block, bank_index, selector=None, limit=256):
    banks = list(executable_banks(blob, BLOCKS[block][0]))
    bank = banks[bank_index]
    addresses = check(blob, *BLOCKS[block])
    src, dst, size = bank["file_offset"], bank["destination"], bank["length"]
    shift = dst - 0x800
    addresses.update(
        stage_dsp_address=hex(dst), first_transfer_destination=hex(dst),
        stack=hex(bank["stack"]), entry_shift=shift,
        selector_targets=[] if bank_index else addresses["selector_targets"],
    )
    result = run_dispatch(addresses, blob[src:src + size], block, selector, limit)
    result["bank"] = bank
    return result


def run_dispatch(addresses, image, block, selector, limit):
    machine = DispatchMachine(addresses, image)
    first = machine.run(limit)
    result = {"block": block, "first_stop": first, "fixture_selector": selector,
              "mmio_writes": machine.mmio_writes, "mailbox_reads": machine.mailbox_reads}
    result["r8_after_entry"] = machine.reg[8]
    if selector is not None:
        initial_wait = {"A": 0x1150, "B": 0xfc8}[block] + addresses.get("entry_shift", 0)
        if first != {"pc": initial_wait, "reason": "wfi"}:
            raise ValueError(f"unexpected stop before mailbox fixture: {first}")
        machine.pc += 2  # Explicit synthetic wake, not a simulated peripheral response.
        request = machine.run(limit)
        result["first_read_after_wake"] = request
        address = request.get("address", 0)
        # A/4 reads three more fixture words at +0x64, +0x68, and +0x6c.
        fixture_size = 0x70 if (block, selector) == ("A", 4) else 0x64
        if (request["reason"] == "unmodeled read" and request.get("size") == 4 and
                not address & 3 and 0x17800 <= address <= 0x17d00 - fixture_size):
            machine.mailbox = address
            words = [selector] + [0x51000000 + offset for offset in range(4, fixture_size, 4)]
            machine.mailbox_data = b"".join(word.to_bytes(4, "little") for word in words)
            result["after_fixture"] = machine.run(limit)
        seen = set(machine.pcs)
        result["reached_targets"] = [target for target in addresses["selector_targets"]
                                     if int(target, 16) in seen]
    result["instructions_seen"] = len(machine.pcs)
    result["registers_at_stop"] = {f"r{i}": v for i, v in enumerate(machine.reg)}
    result["registers_at_stop"]["lr"] = machine.link
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("member", type=Path)
    cli.add_argument("--block", choices=BLOCKS, default="A")
    cli.add_argument("--bank", type=int, choices=range(5), help="Enter a copied local bank directly")
    cli.add_argument("--fixture-selector", type=int, choices=range(6))
    cli.add_argument("--max-steps", type=int, default=256)
    args = cli.parse_args()
    if args.max_steps <= 0:
        cli.error("--max-steps must be positive")
    data = args.member.read_bytes()
    if sha256(data).hexdigest() != MEMBER_SHA:
        cli.error("expected pinned A7 IV FDN member")
    if args.bank is None:
        result = front_end(data, args.block, args.fixture_selector, args.max_steps)
    else:
        result = bank_entry(data, args.block, args.bank, args.fixture_selector, args.max_steps)
    print(json.dumps(result, indent=2))
