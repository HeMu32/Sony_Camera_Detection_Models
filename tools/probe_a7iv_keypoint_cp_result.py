"""Replay the original common AED/AED2/BED ARM parameter and result methods."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from struct import pack, unpack_from
import tarfile

from unicorn.arm_const import (
    UC_ARM_REG_C1_C0_2, UC_ARM_REG_FPEXC, UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_R8,
)

from parsers.sony_bk4 import published_pair
from probe_a7iv_fdn_cp_request import BACKUP_MEMBER, REGS, U32, execute
from probe_a7iv_fdn_cp_result import (
    GLOBALS, MANAGER, MINNA_CONFIG, OUTPUT, RECORD, cp_machine, probe_minna_backup,
)


CONSTRUCTOR = 0x6fd0b4
RETURN = 0x2656e4
CANDIDATE, PARAMETER, TABLE_PTRS = (0x30000000 + off for off in (0x300, 0x400, 0x500))
TABLE_A, TABLE_B = 0x30000800, 0x30000a50
CANDIDATE_BYTES = 0x64
RECORD_BYTES = 0x58
LUT_BYTES = 0x244
LUT_HASHES = (
    "e88f7cd4c382182b1d2a2ec38536ac51761c465f8cb4bd32e1297174dd9af61b",
    "90d97c016c79516041667f5a4a3b9708050389700bd7b481568e5f6ac019b4cd",
)


def published_config(blob, member):
    _, image = published_pair(member)[1]
    status, _, logged, blocks = probe_minna_backup(blob, image)
    if status != len(MINNA_CONFIG) or logged:
        raise ValueError("published MINNA configuration did not load")
    return blocks[0]


def score_resource_paths(blob):
    """Read native AED2 filenames immediately before the two loader calls."""
    m = cp_machine(blob, (0x260000,))
    for start, stop in ((0x260290, 0x260298), (0x260322, 0x26032a)):
        execute(m, start, stop, ((UC_ARM_REG_R8, MANAGER), (UC_ARM_REG_R0, PARAMETER)))
        owner, buffer, path = (m.reg_read(r) for r in REGS)
        if (owner, buffer) != (MANAGER, PARAMETER):
            raise ValueError("unexpected AED2 resource loader arguments")
        yield PurePosixPath(bytes(m.mem_read(path, 0x20)).split(b"\0", 1)[0].decode("ascii"))


def published_score_tables(blob, directory):
    tables = tuple((directory / path.name).read_bytes() for path in score_resource_paths(blob))
    if tuple(sha256(table).hexdigest() for table in tables) != LUT_HASHES:
        raise ValueError("expected the exact published AED2 score resources")
    return tables


def probe_keypoint(blob, config, candidate, record=None, *, fixture_tables=None):
    """Observe firmware on an explicit candidate and synthetic SA writeback."""
    if (len(config) != 0x100 or len(candidate) != CANDIDATE_BYTES or
            (record is not None and len(record) != RECORD_BYTES)):
        raise ValueError("unexpected MINNA, candidate or result fixture size")
    pages = (0x265000, 0x266000, 0x275000, 0x585000, 0x586000, 0x6fa000, 0x6fd000,
             0x713000, 0xb9b000, 0xc17000)
    m = cp_machine(blob, pages)
    for page in (GLOBALS, 0x01243000, 0x014b6000):
        m.mem_map(page, 0x1000)
    m.mem_write(MINNA_CONFIG[0][1], bytes(config))
    m.mem_write(CANDIDATE, bytes(candidate))
    if record is not None:
        m.mem_write(RECORD, bytes(record))
        m.mem_write(OUTPUT, b"\xa5" * 0xc4)
    tables = 0
    if fixture_tables is not None:
        if len(fixture_tables) != 2 or any(len(table) != LUT_BYTES for table in fixture_tables):
            raise ValueError("expected two complete score table fixtures")
        tables = TABLE_PTRS
        m.mem_write(tables, pack("<2I", TABLE_A, TABLE_B))
        for address, table in zip((TABLE_A, TABLE_B), fixture_tables):
            m.mem_write(address, bytes(table))
        m.reg_write(UC_ARM_REG_C1_C0_2, 0xf00000)
        m.reg_write(UC_ARM_REG_FPEXC, 1 << 30)

    def call(entry, *args):
        execute(m, entry, RETURN, (*zip(REGS, args), (UC_ARM_REG_LR, RETURN | 1)))

    call(CONSTRUCTOR, MANAGER, CANDIDATE, tables)
    vptr, = U32.unpack(m.mem_read(MANAGER, 4))
    installed_tables, = U32.unpack(m.mem_read(MANAGER + 0xc8, 4))
    if installed_tables != tables:
        raise ValueError("original keypoint constructor did not bind the resource pointers")
    prepare, read_result = unpack_from("<2I", m.mem_read(vptr + 0x20, 8))
    m.mem_write(MANAGER + 8, pack("<2I", PARAMETER, RECORD))
    # Fixtures for the allocator's two resource bases and three offsets.
    m.mem_write(MANAGER + 0x20, pack("<2Q", 0x500010000, 0x500020000))
    m.mem_write(MANAGER + 0x48, pack("<4I", 0x100, 0x200, 0, 0x300))
    call(prepare, MANAGER, CANDIDATE)
    if record is not None:
        call(read_result, MANAGER, OUTPUT)
    return {
        "vtable": vptr, "methods": (prepare & ~1, read_result & ~1),
        "input_route_flag": m.mem_read(MANAGER + 0x59, 1)[0],
        "parameter": bytes(m.mem_read(PARAMETER, 0x80)),
        "result": bytes(m.mem_read(OUTPUT, 0xc4)) if record is not None else None,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("firmware_tar", type=Path)
    cli.add_argument("candidate", type=Path, help="Explicit 0x64-byte CP candidate")
    cli.add_argument("result", type=Path, help="Explicit 0x58-byte SA output")
    cli.add_argument("--score-dir", type=Path, help="Bind the published AED2 score resources")
    args = cli.parse_args()
    cp = args.cp.read_bytes()
    with tarfile.open(args.firmware_tar, "r") as archive:
        config = published_config(cp, archive.extractfile(BACKUP_MEMBER).read())
    tables = published_score_tables(cp, args.score_dir) if args.score_dir else None
    witness = probe_keypoint(cp, config, args.candidate.read_bytes(), args.result.read_bytes(),
                             fixture_tables=tables)
    print(json.dumps({k: v.hex(" ") if isinstance(v, bytes) else v for k, v in witness.items()},
                     indent=2))
