"""Run the pinned CP FDN consumer, with explicitly synthetic records and thresholds."""

from hashlib import sha256
from struct import pack, unpack_from

from unicorn import UC_ARCH_ARM, UC_HOOK_CODE, UC_MODE_ARM, UC_PROT_EXEC, UC_PROT_READ, Uc
from unicorn.arm_const import (UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R3,
                               UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_SB, UC_ARM_REG_SP)

from parsers.fdn_face_frame import check_geometry
from parsers.sony_bk4 import Backup, published_pair
from probe_a7iv_fdn_cp_request import CP_BASE, CP_SHA256, REGS, execute


ENTRY, FIXTURE_RETURN = 0x264d30, 0x264f60
DRIVER_RESULT_ENTRY, MANAGER_VTABLE = 0x2646ac, 0xc17e68
GLOBALS = 0x0123e000
THRESHOLDS = (0x0123e5c8, 0x0123e5cc)
MANAGER, RECORD, OUTPUT, SP = (0x30000000 + i * 0x100 for i in (0, 1, 2, 15))
MINNA_KEYS, MINNA_DESTINATIONS = 0xf608cc, 0xf608b8
MINNA_CONFIG = (
    (0xb0000, 0x0123e528), (0xb0001, 0x014b6023),
    (0xb0002, 0x014b5f23), (0xb0003, 0x01486a8c),
)


def cp_machine(blob, pages):
    if sha256(blob).hexdigest() != CP_SHA256:
        raise ValueError("expected pinned A7 IV 6.02 CP image")
    machine = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    for page in pages:
        machine.mem_map(page, 0x1000, UC_PROT_READ | UC_PROT_EXEC)
        machine.mem_write(page, blob[page - CP_BASE:page - CP_BASE + 0x1000])
    machine.mem_map(MANAGER, 0x1000)
    machine.reg_write(UC_ARM_REG_SP, SP)
    return machine


def probe_result_pointer(blob, physical_base, offset, section_descriptor):
    """Run the CP result-pointer assignment with a fixture ARM section table."""
    machine = cp_machine(blob, (0x264000, 0xb9b000, 0x566000, 0x579000, 0x582000))
    table, control = 0x30004000, 0x0147a2a8
    machine.mem_map(table, 0x4000)
    machine.mem_map(control & ~0xfff, 0x1000)
    machine.mem_write(control, pack("<I", table))
    machine.mem_write(MANAGER + 0x0c, pack("<I", 0xdeadbeef))
    machine.mem_write(MANAGER + 0x20, pack("<Q", physical_base))
    machine.mem_write(MANAGER + 0x48, pack("<I", offset))
    # The descriptor is for the synthetic 0x82000000 section only.
    machine.mem_write(table + (0x82000000 >> 20) * 4, pack("<I", section_descriptor))
    execute(machine, 0x264afe, 0x264b5c,
            ((UC_ARM_REG_R8, MANAGER), (UC_ARM_REG_R9, MANAGER + 0x20)))
    return unpack_from("<I", machine.mem_read(MANAGER + 0x0c, 4))[0]


def prepare_result(machine, manager, record, thresholds, *, search_type=9):
    if thresholds is not None:
        machine.mem_map(GLOBALS, 0x1000)
        machine.mem_write(THRESHOLDS[0], pack("<2i", *thresholds))
    machine.mem_write(manager + 0x0c, pack("<I", RECORD))
    machine.mem_write(manager + 0x5b,
                      pack("<B4I", search_type, 0x13579bdf, 0x2468ace0, *THRESHOLDS))
    machine.mem_write(RECORD, pack("<i8I", *record))
    machine.mem_write(OUTPUT, bytes([0xa5]) * 0x40)


def read_result(machine):
    data = bytes(machine.mem_read(OUTPUT, 0x40))
    return {
        "search_type": data[1], "exec_id": unpack_from("<I", data, 4)[0],
        "user_data": unpack_from("<I", data, 8)[0], "passed": data[0xc],
        "left_eye_raw": unpack_from("<2I", data, 0x14),
        "right_eye_raw": unpack_from("<2I", data, 0x20),
        "nose_root_raw": unpack_from("<2I", data, 0x2c),
        "score": unpack_from("<i", data, 0x34)[0], "other_direction": data[0x38],
        "output_hex": data.hex(" "),
    }


def probe(blob, record, thresholds):
    machine = cp_machine(blob, (ENTRY & ~0xfff,))
    prepare_result(machine, MANAGER, record, thresholds)
    # Leave logging disabled. Execute the original consumer's return as well.
    execute(machine, ENTRY, FIXTURE_RETURN, (*zip(REGS, (MANAGER, OUTPUT)),
                                            (UC_ARM_REG_LR, FIXTURE_RETURN | 1)))
    return read_result(machine)


def probe_dispatch(blob, record, thresholds, *, manager_id, status, search_type=9):
    if manager_id not in range(10) or status not in range(3):
        raise ValueError("expected one of ten managers and a normal response status 0..2")
    driver, notice = 0x30000400, 0x30000300
    machine = cp_machine(blob, (0x264000, 0x713000, MANAGER_VTABLE & ~0xfff))
    for i in range(10):
        manager = driver + 0xc8 + i * 0x70
        machine.mem_write(manager, pack("<IB", MANAGER_VTABLE, i))
        machine.mem_write(manager + 0x5a, b"\x01")
    prepare_result(machine, driver + 0xc8 + manager_id * 0x70, record, thresholds,
                   search_type=search_type)
    machine.mem_write(notice, bytes((manager_id, status)))
    execute(machine, DRIVER_RESULT_ENTRY, FIXTURE_RETURN,
            (*zip(REGS, (driver, notice, OUTPUT)), (UC_ARM_REG_LR, FIXTURE_RETURN | 1)))
    return {"ready": machine.mem_read(OUTPUT, 1)[0],
            "manager_flags": [machine.mem_read(driver + 0xc8 + i * 0x70 + 0x5a, 1)[0]
                              for i in range(10)],
            "result": read_result(machine)}


def probe_listener(blob, dispatched):
    """Replay the FC listener on a ready CP result, stopping before enqueue."""
    if len(dispatched) != 0x40 or dispatched[0] != 1 or dispatched[1] not in (0, 1):
        raise ValueError("expected a ready CP fixture with FC route 0 or 1")
    route = dispatched[1]
    listener, owner = 0x30000300, 0x30000400
    vtable, stop = ((0xc194d4, 0x72ff8a), (0xc19524, 0x7305b6))[route]
    listener_table = 0xc19760
    pages = (0x28a000, 0x6fa000, 0x72f000, 0x730000, 0xc19000)
    machine = cp_machine(blob, pages)
    machine.mem_write(OUTPUT, dispatched)
    machine.mem_write(listener, pack("<I", listener_table))
    machine.mem_write(listener + 4 + route * 4, pack("<I", owner + 4))
    machine.mem_write(owner + 4, pack("<I", vtable))
    machine.mem_write(owner + 0xc, bytes([route]))
    entry, = unpack_from("<I", machine.mem_read(listener_table + 8, 4))
    execute(machine, entry, stop, zip(REGS, (listener, OUTPUT)))
    # Both routes construct the same FDNResult at SP+0x3c before enqueue.
    data = machine.mem_read(machine.reg_read(UC_ARM_REG_SP) + 0x3c, 0x38)
    return {
        "exec_id": unpack_from("<I", data)[0], "passed": data[4],
        "score": unpack_from("<i", data, 8)[0],
        "landmarks": tuple(unpack_from("<2I", data, offset) for offset in (0x10, 0x1c, 0x28)),
        "other_direction": data[0x30], "user_data": unpack_from("<I", data, 0x34)[0],
    }


def probe_face_frame(blob, xywh, frame, *, transform=None, square=True, minimum=0):
    """Execute FC's native rectangle getter and output, with no lens map."""
    transform = frame if transform is None else transform
    check_geometry(xywh, frame, transform, minimum)
    machine = cp_machine(blob, (0x27f000, 0x280000, 0x6fa000, 0x715000, 0x736000,
                                0x9b7000, 0xc1a000))
    machine.mem_map(0x0140d000, 0x1000)
    face, projection, node = 0x30000300, 0x30000400, 0x30000500
    machine.mem_write(node + 8, pack("<I", face))
    machine.mem_write(face, pack("<I", 0xc1a0d4))
    machine.mem_write(face + 0x18, pack("<I4H", 0xc16bbc, *xywh))
    machine.mem_write(SP + 0x48, pack("<BB3HI", int(square), minimum, *frame, 0, projection))
    machine.mem_write(projection, pack("<8H2I", 0, 0, *transform, *transform, 0, 0, 0, 0))
    execute(machine, 0x27f7f2, 0x27f82a, ((UC_ARM_REG_SB, node),))
    return unpack_from("<4H", machine.mem_read(SP + 0x40, 8))


def probe_face_pair(blob, first, second, *, weights, sources):
    """Execute FC's stock same-face predicate and pair merger on explicit candidates."""
    machine = cp_machine(blob, (0x6fa000, 0x715000, 0x727000, 0x736000, 0x738000, 0xc1a000))
    owner, a, b, merged = 0x30000300, 0x30000400, 0x30000500, 0x30000600
    for address, rect, weight, source in zip((a, b), (first, second), weights, sources, strict=True):
        machine.mem_write(address, pack("<I4xBBH", 0xc1a0d4, 0, source, weight))
        machine.mem_write(address + 0x18, pack("<I4H", 0xc16bbc, *rect))
    machine.mem_write(merged, pack("<I", 0xc1a0d4))

    def call(entry, args):
        execute(machine, entry, FIXTURE_RETURN, (
            *zip((*REGS, UC_ARM_REG_R3), args),
            (UC_ARM_REG_LR, FIXTURE_RETURN | 1),
        ))

    call(0x72744e, (owner, 0))
    threshold, = unpack_from("<H", machine.mem_read(owner + 0x10, 2))
    call(0x727afc, (owner + 0xc, a + 0x18, b + 0x18))
    same = machine.reg_read(UC_ARM_REG_R0)
    if same == 0:
        return threshold, None
    if same != 1:
        raise ValueError("unexpected FC same-face predicate output")
    call(0x727482, (owner, a, b, merged))
    return threshold, unpack_from("<4H", machine.mem_read(merged + 0x1c, 8))


def probe_minna_backup(blob, backup):
    """Replay the bulk loader with its original tables, omitting locks and logging."""
    _, status, attempted, logged, blocks = _load_minna_backup(blob, backup)
    return status, attempted, logged, blocks


def _load_minna_backup(blob, backup):
    keys, destinations = (unpack_from("<5I", blob, va - CP_BASE)
                          for va in (MINNA_KEYS, MINNA_DESTINATIONS))
    if tuple(zip(keys, destinations)) != (*MINNA_CONFIG, (0, 0)):
        raise ValueError("unexpected MINNA configuration bindings")
    source = Backup(backup)
    for key, _ in MINNA_CONFIG:
        item = source.lookup(key)
        if item is not None and len(item[1]) > 0x100:
            raise ValueError("MINNA record exceeds replay size limit")
    base = 0x32000000
    pages = (FIXTURE_RETURN & ~0xfff, 0x266000, 0x4c3000, 0xac9000, 0x585000, 0x586000, 0xf60000,
             0x01012000)
    machine = cp_machine(blob, pages)
    for page in (GLOBALS, 0x01464000, 0x014b5000, 0x014b6000, 0x01486000):
        machine.mem_map(page, 0x1000)
    machine.mem_map(base, (len(backup) + 0xfff) & ~0xfff, UC_PROT_READ)
    machine.mem_write(base, backup)
    machine.mem_write(0x0146492c, pack("<3I", base, base + 0x100, base + source.items_offset))
    for _, address in MINNA_CONFIG:
        machine.mem_write(address, b"\xa5" * 0x100)
    attempted, logged = [], []

    def trace(m, pc, size, _user):
        if pc == 0x4c30a6:
            attempted.append(m.reg_read(UC_ARM_REG_R0))
        elif pc == 0x4c30d0:
            logged.append(m.reg_read(UC_ARM_REG_R3))
            m.reg_write(UC_ARM_REG_PC, (pc + size) | 1)
        elif pc in (0x4c3066, 0x4c3082, 0x4c3092, 0x4c30e0):
            m.reg_write(UC_ARM_REG_PC, (pc + size) | 1)

    machine.hook_add(UC_HOOK_CODE, trace, begin=0x4c3040, end=0x4c30ec)
    execute(machine, 0x4c3040, FIXTURE_RETURN,
            (*zip(REGS, (MINNA_KEYS, MINNA_DESTINATIONS)),
             (UC_ARM_REG_LR, FIXTURE_RETURN | 1)), limit=4096)
    status, = unpack_from("<i", pack("<I", machine.reg_read(UC_ARM_REG_R0)))
    return machine, status, tuple(attempted), tuple(logged), tuple(
        bytes(machine.mem_read(address, 0x100)) for _, address in MINNA_CONFIG)


def probe_with_published_minna(blob, member, record, *, pdn_scores=None):
    """Read the released MINNA defaults in CP and apply them to fixture scores."""
    offset, image = published_pair(member)[1]
    machine, status, attempted, logged, blocks = _load_minna_backup(blob, image)
    if status != len(MINNA_CONFIG) or logged:
        raise ValueError("published MINNA configuration did not load")
    prepare_result(machine, MANAGER, record, None)
    execute(machine, ENTRY, FIXTURE_RETURN, (*zip(REGS, (MANAGER, OUTPUT)),
                                            (UC_ARM_REG_LR, FIXTURE_RETURN | 1)))
    result = {"member_offset": offset, "loaded_keys": attempted, "config_blocks": blocks,
              "thresholds": tuple(unpack_from("<i", blocks[0], i)[0] for i in (0xa0, 0xa4, 0xb0)),
              "result": read_result(machine)}
    if pdn_scores is not None:
        machine.mem_write(MANAGER + 0x58, pack("<2I", 0, 0x0123e5d8))
        machine.mem_write(RECORD, pack("<2iI", *pdn_scores, 0))
        execute(machine, 0x266aa0, 0x266ad8, zip(REGS, (MANAGER, OUTPUT)))
        native = machine.mem_read(OUTPUT + 8, 0xc)
        result["pdn_passes"] = tuple(native[:2])
        result["pdn_scores"] = unpack_from("<2i", native, 4)
    return result


def probe_minna_update(blob, payload, *, key=0xb0000):
    """Replay one inline BK4 notification with the original MINNA callback."""
    if not 0 < len(payload) <= 0x100 or not 0 <= key <= 0xffffffff:
        raise ValueError("expected one short update fixture")
    msg, subscriber = 0x30000300, 0x30000500
    registered, target = MINNA_CONFIG[0]
    machine = cp_machine(blob, (0xaca000, 0x6f7000, 0x521000, 0x585000, 0x586000))
    machine.mem_map(GLOBALS, 0x1000)
    machine.mem_write(target, b"\xa5" * 0x100)
    machine.mem_write(msg, pack("<3I", key, len(payload), 1) + payload)
    machine.mem_write(subscriber + 8, pack("<3I", registered, 0x6f778d, target))
    stop = 0xaca334 if key == registered else 0xaca39a
    execute(machine, 0xaca2ba, stop, zip(REGS, (msg, subscriber)))
    return bytes(machine.mem_read(target, 0x100))
