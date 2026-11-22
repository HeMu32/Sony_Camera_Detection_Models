"""Check A7 IV FDN's two stage-2 entry maps and mailbox jump tables."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct


MEMBER_SHA = "1d146af1db2b188dcaf1bdbf41ec9dee95b6fe6bfc8d76b2295abe3f9a77eead"
U16 = Struct("<H")
U32 = Struct("<I")
ROW = Struct("<4I")
BANK_ROWS = (0, 1, 3, 6, 8)
STUB_START = bytes.fromhex("06 b0") + bytes(14)


def word(blob, offset):
    return U32.unpack_from(blob, offset)[0]


def pc_literal(read, pc, register):
    opcode, = U16.unpack(read(pc, 2))
    if opcode >> 12 != 5 or opcode & 15 != register:
        raise ValueError(f"expected ldr r{register},[pc,#imm] at {pc:#x}")
    literal = (pc & ~3) + 4 + 4 * ((opcode >> 4) & 255)
    return word(read(literal, 4), 0)


def executable_banks(blob, block):
    stage = block + word(blob, block + 0x10)
    descriptor = block + word(blob, block + 0x20)
    if word(blob, block + 0x24) != 19 * ROW.size:
        raise ValueError("unexpected FDN descriptor table size")
    for index in BANK_ROWS:
        source, dest, w2, w3 = ROW.unpack_from(blob, descriptor + ROW.size * index)
        start, length = stage + source, (w2 >> 16) + 1
        if (w2 & 0xffff != 0x4000 or w3 != w2 & 0xffff0000 or
                dest & 0xff or length > 0x1000 or start + length > len(blob)):
            raise ValueError(f"unsupported bank descriptor {index}")
        data = blob[start:start + length]
        stack, entry, doorbell = Struct("<3I").unpack_from(data, 0x28)
        if (not data.startswith(STUB_START) or not dest <= entry < dest + length or
                stack != dest + 0x1000 or doorbell != 0x48208):
            raise ValueError(f"unexpected stub in bank descriptor {index}")
        yield {"file_offset": start, "destination": dest, "length": length,
               "entry": entry, "stack": stack, "descriptor_index": index}


def check(blob, block, mailbox_pc, doorbell_pc, table_pc):
    stage_offset = block + word(blob, block + 0x10)
    stage_address = word(blob, block + 0x54)
    descriptor = block + word(blob, block + 0x20)
    table_size = word(blob, block + 0x24)
    source, dest, w2, w3 = ROW.unpack_from(blob, descriptor)
    if (table_size != 19 * ROW.size or source != 0 or dest != 0x800 or
            w2 & 0xffff != 0x4000 or w3 != w2 & 0xffff0000):
        raise ValueError("unsupported first transfer descriptor")
    length = (w2 >> 16) + 1

    def local_read(address, size):
        offset = address - dest
        if not 0 <= offset <= length - size:
            raise ValueError("read outside first proposed DMA span")
        start = stage_offset + source + offset
        return blob[start:start + size]

    stack = pc_literal(local_read, dest + 0x10, 15)
    entry = pc_literal(local_read, dest + 0x14, 1)
    if local_read(entry, 8) != bytes.fromhex("8f 18 9f 18 af 18 bf 18"):
        raise ValueError("mapped entry does not start by saving r8..r11")
    mailbox = pc_literal(local_read, mailbox_pc, 8)
    doorbell = pc_literal(local_read, doorbell_pc, 0)
    table = pc_literal(local_read, table_pc, 0)
    targets = [table + offset for offset in local_read(table, 5)]
    if (stack, mailbox, doorbell) != (0x1800, 0x17800, 0x48208):
        raise ValueError("unexpected stage-2 stack or dispatch addresses")
    if any(target & 1 or not dest <= target < dest + length for target in targets):
        raise ValueError("dispatch target outside mapped instructions")
    return {
        "block_file_offset": hex(block), "stage_file_offset": hex(stage_offset),
        "stage_dsp_address": hex(dest), "descriptor_file_offset": hex(descriptor),
        "boot_external_entry": hex(stage_address),
        "first_transfer_bytes": length, "first_transfer_destination": hex(dest),
        "stack": hex(stack), "entry_dsp_address": hex(entry),
        "entry_file_offset": hex(stage_offset + source + entry - dest),
        "mailbox_selector": hex(mailbox), "doorbell_candidate": hex(doorbell),
        "jump_table": hex(table), "selector_targets": [hex(target) for target in targets],
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("member", type=Path)
    blob = cli.parse_args().member.read_bytes()
    if sha256(blob).hexdigest() != MEMBER_SHA:
        cli.error("expected pinned A7 IV FDN member")
    print(json.dumps([
        check(blob, 0x100, 0x1134, 0x1148, 0x1162),
        check(blob, 0x1a800, 0xfb4, 0xfc4, 0xfd6),
    ], indent=2))
