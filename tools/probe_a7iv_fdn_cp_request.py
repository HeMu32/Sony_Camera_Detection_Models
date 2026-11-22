"""Execute pinned CP request builders locally, stopping before the queue call."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct
import tarfile

from unicorn import (
    UC_ARCH_ARM, UC_HOOK_MEM_WRITE, UC_MODE_ARM, UC_PROT_ALL, UC_PROT_READ, UC_PROT_EXEC, Uc,
)
from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
    UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R7, UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10,
    UC_ARM_REG_SP,
)


CP_BASE = 0x108000
CP_SHA256 = "7b9b137d696e5972d20db7f9119076172ed8e0730805d7f0b4c31b87689e2c2b"
BACKUP_MEMBER = "0110_backup/SYSIPSX-DSLR/LS/CH89100_UC2.bin"
BACKUP_SHA256 = "298dda092a4e948c1a1235f6db83c69ef51b4b703dfd7e2ef41a5197e86418dd"
SP_SHA256 = "6ea5cda332e49b0270006349ea687905e26766dd2aa6d1750a07a28711968136"
DRIVER, MANAGER, INPUT, DESCRIPTOR = (0x30000000 + i * 0x1000 for i in range(4))
STACK = 0x3000ffc0
PARAM_VA = 0x80001000
WINDOW = (640, 480, 704, 32, 64, 480, 320, 0, 2)
U32 = Struct("<I")
REGS = (UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2)
PYRAMID_RECORD = Struct("<Q3H2x")


def execute(machine, start, stop, registers=(), *, limit=1024):
    for register, value in registers:
        machine.reg_write(register, value)
    machine.emu_start(start | 1, stop, timeout=1_000_000, count=limit)
    pc = machine.reg_read(UC_ARM_REG_PC)
    if pc != stop:
        raise RuntimeError(f"slice {start:#x} stopped at {pc:#x}, expected {stop:#x}")


def pyramid_slice(machine, levels, extent):
    """Carry an explicit FC frame table through the original FDN request prefix."""
    if not 0 <= extent <= 0xffff or len(levels) != 5:
        raise ValueError("expected a 16-bit candidate extent and five frame records")
    producer, selected, metadata, level_out = (DRIVER + off for off in (0x6000, 0x6100, 0x6200, 0x6300))
    raw = b"".join(PYRAMID_RECORD.pack(*entry) for entry in levels)
    machine.mem_write(producer, raw)
    machine.mem_write(selected, b"\xa5" * len(raw))
    machine.mem_write(level_out, U32.pack(5))
    machine.mem_write(STACK + 0xc, Struct("<H").pack(extent))

    # ProcessFdnImpl's copy loop moves exactly 14 bytes of each record.
    execute(machine, 0x72f362, 0x72f378, (
        (UC_ARM_REG_R1, selected), (UC_ARM_REG_R10, producer),
    ))
    copied = bytes(machine.mem_read(selected, len(raw)))
    if copied != b"".join(raw[i:i + 14] + b"\xa5\xa5" for i in range(0, len(raw), 16)):
        raise ValueError("FDN did not copy the expected frame records")

    # The candidate-geometry virtual getter has already returned in this slice.
    execute(machine, 0x72f8aa, 0x72f8ca, ((UC_ARM_REG_R4, level_out),))
    ok = machine.reg_read(UC_ARM_REG_R4)
    index, = U32.unpack(machine.mem_read(level_out, 4))
    if (ok, index) == (0, 5):
        raise ValueError("original FDN scale selector rejected the candidate extent")
    if ok != 1 or not 0 <= index < len(levels):
        raise ValueError("unexpected FDN scale selector result")

    machine.mem_write(metadata + 4, U32.pack(0x13579bdf))
    machine.mem_write(STACK, U32.pack(0))
    execute(machine, 0x28ac1c, 0x28ac66, (
        *zip(REGS, (INPUT, metadata, selected)), (UC_ARM_REG_R3, index),
    ))
    return {"candidate_extent": extent, "level": index,
            "copied_frame_record_hex": copied[index * 16:(index + 1) * 16].hex(" ")}


def backup_load(machine, key, destination, size, *, expected_size=None):
    stop = 0xac95dc
    # The original, unlocked backup reader; do not enter OS locking or queue code.
    execute(machine, 0xac95de, stop, (*zip(REGS, (key, destination, size)),
            (UC_ARM_REG_R3, 1), (UC_ARM_REG_LR, stop | 1)))
    loaded = machine.reg_read(UC_ARM_REG_R0)
    if loaded != (size if expected_size is None else expected_size):
        raise ValueError(f"firmware backup reader returned {loaded:#x} bytes for {key:#x}")
    return bytes(machine.mem_read(destination, loaded))


def factory_lookup(machine, backup, backup_args, job, lookup_output, frame):
    if sha256(backup).hexdigest() != BACKUP_SHA256:
        raise ValueError("expected pinned A7 IV 6.02 UC2 BK4 archive member")
    base, stop = 0x32000000, 0xac95dc
    groups = U32.unpack_from(backup, 0x10)[0]
    machine.mem_map(base, (len(backup) + 0xfff) & ~0xfff, UC_PROT_READ)
    machine.mem_write(base, backup)
    machine.mem_map(0x01464000, 0x1000, UC_PROT_ALL & ~UC_PROT_EXEC)
    machine.mem_map(0x014f8000, 0x2000, UC_PROT_ALL & ~UC_PROT_EXEC)
    machine.mem_write(0x0146492c, Struct("<3I").pack(base, base + 0x100, base + 0x100 + groups * 6))

    raw = backup_load(machine, *backup_args)
    execute(machine, 0xb02238, stop, ((UC_ARM_REG_R0, job), (UC_ARM_REG_R1, lookup_output),
                                     (UC_ARM_REG_LR, stop | 1)))
    value = machine.mem_read(lookup_output, 1)[0]
    execute(machine, 0x50bc04, 0xb05ed2, ((UC_ARM_REG_R7, frame),))
    execute(machine, 0xb05ed2, 0xb05eda)
    return {"table_sha256": sha256(raw).hexdigest(), "job_byte": hex(value),
            "next_predicate_pc": hex(0xb05ed2),
            "runtime_counter_address": hex(machine.reg_read(UC_ARM_REG_R0))}


def resource_completion_slice(machine, resource_prefix, member_base, member_len, worker, job):
    # One synthetic allocation result and one ARM section descriptor; no OS or DMA.
    reply, table, destination = 0x30009800, 0x30004000, 0x82000000
    reply_phys = 0x500000000 + destination - 0x80000000
    slot = Struct("<H").unpack_from(resource_prefix, 0x14)[0]
    machine.mem_map(0x0147a000, 0x1000, UC_PROT_ALL & ~UC_PROT_EXEC)
    machine.mem_write(0x0147a2a8, U32.pack(table))
    machine.mem_write(table + (destination >> 20) * 4, U32.pack(destination | 2))
    message = Struct("<HH2xH4xI8xQI4x").pack(0x8020, 0, slot, 0x11223344, reply_phys,
                                              0x55667788)
    machine.mem_write(reply, message)
    machine.reg_write(UC_ARM_REG_SP, 0x3000b000)
    execute(machine, 0x50b214, 0x50c7b0, ((UC_ARM_REG_R0, reply),))
    normalized = machine.reg_read(UC_ARM_REG_R0)
    completion = bytes(machine.mem_read(normalized, 0x14))
    if U32.unpack_from(completion, 0x10)[0] != destination:
        raise ValueError("resource handler did not translate the fixture address")

    execute(machine, 0x50c7b0, 0x4e63a4)
    descriptor = machine.reg_read(UC_ARM_REG_R0)
    args = bytes(machine.mem_read(descriptor, 0x10))
    transfers = U32.unpack_from(args, 4)[0]
    if transfers != 1:
        raise ValueError(f"expected one transfer, saw {transfers}")
    transfer_addr = U32.unpack_from(args, 8)[0]
    size, source, target = Struct("<3I").unpack(machine.mem_read(transfer_addr, 0xc))
    if (size, source, target) != (member_len, member_base, destination):
        raise ValueError(f"DMA input does not describe the selected member: "
                         f"{(hex(size), hex(source), hex(target))}")

    # Supply the descriptor's byte copy in RAM to exercise only the CP header lookup.
    machine.mem_map(destination, (size + 0xfff) & ~0xfff, UC_PROT_READ)
    machine.mem_write(destination, bytes(machine.mem_read(source, size)))
    entry_out, stop = 0x30009880, 0xac95dc
    entry_lookup = []
    for selector in (0, 0xff):
        machine.reg_write(UC_ARM_REG_SP, 0x3000b000)
        machine.mem_write(0x3000b000, U32.pack(0))  # No optional header fields requested.
        machine.mem_write(entry_out, U32.pack(0))
        execute(machine, 0xb03008, stop,
                (*zip(REGS, (worker, job, selector)), (UC_ARM_REG_R3, entry_out),
                 (UC_ARM_REG_LR, stop | 1)))
        entry_lookup.append({"selector": selector, "status": machine.reg_read(UC_ARM_REG_R0),
                             "returned_entry": hex(U32.unpack(machine.mem_read(entry_out, 4))[0])})

    return {"fixture_reply_hex": message.hex(" "), "normalized_reply_hex": completion.hex(" "),
            "stop_pc": hex(0x4e63a4), "descriptor_address": hex(descriptor),
            "descriptor_hex": args.hex(" "),
            "transfer": {"size": size, "source": hex(source), "destination": hex(target)},
            "entry_lookup_with_fixture_copy": entry_lookup}


def sa_argument_write_slice(machine, subrecord, worker):
    # The CP's register window is RAM in Unicorn, with no modeled device or start pulse.
    base = U32.unpack(machine.mem_read(0xd5fbbc + worker * 4, 4))[0]
    if base != 0xf5440000 + worker * 0x10000:
        raise ValueError("unexpected SA register window in the pinned CP image")
    machine.mem_map(base, 0x1000, UC_PROT_ALL & ~UC_PROT_EXEC)
    writes = []

    def record_write(_uc, _access, address, size, value, _data):
        writes.append({"offset": hex(address - base), "size": size, "value": hex(value)})

    hook = machine.hook_add(UC_HOOK_MEM_WRITE, record_write, begin=base, end=base + 0xfff)
    try:
        machine.reg_write(UC_ARM_REG_SP, 0x3000b000)
        execute(machine, 0x50cb76, 0x50cbba,
                ((UC_ARM_REG_R10, subrecord), (UC_ARM_REG_R9, worker), (UC_ARM_REG_R5, 0)))
    finally:
        machine.hook_del(hook)
    return {"stop_pc": hex(0x50cbba), "register_base": hex(base), "writes": writes}


def scheduler_slices(machine, subrecord, job, bundle, receiver_frame):
    if sha256(bundle).hexdigest() != SP_SHA256:
        raise ValueError("expected pinned A7 IV 6.02 camera-post SP bundle")
    stop, sp_base = 0xac95dc, 0x33000000
    machine.reg_write(UC_ARM_REG_SP, 0x3000b000)
    for page, length in ((0x011ec000, 0x1000), (0x01472000, 0x3000),
                         (0x014b1000, 0x1000), (0x01563000, 0x1000)):
        machine.mem_map(page, length, UC_PROT_ALL & ~UC_PROT_EXEC)
    for key, destination, size in ((0x930000, 0x014f8fe4, 0x200), (0x930009, 0x014f91e4, 0x200),
                                   (0x93000b, 0x01472cd8, 0x80)):
        backup_load(machine, key, destination, size)

    # Trace cold scheduler setup, then insert a host-only SP registry entry.
    execute(machine, 0xb05714, 0xb05762)
    execute(machine, 0xb02a9c, 0xb02afe)
    machine.mem_map(sp_base, (len(bundle) + 0xfff) & ~0xfff, UC_PROT_READ)
    machine.mem_write(sp_base, bundle)
    machine.mem_write(0x011ece90, U32.pack(sp_base))

    original_subrecord = bytes(machine.mem_read(subrecord, 0x20))
    usecase = original_subrecord[4]
    mask_out = 0x3000a000
    execute(machine, 0xb0567c, stop, (*zip(REGS, (mask_out, job, usecase)),
                                     (UC_ARM_REG_LR, stop | 1)))
    mask, error = Struct("<II").unpack(machine.mem_read(mask_out, 8))
    if error:
        raise ValueError("firmware could not resolve the use case")
    execute(machine, 0xb052f0, stop, ((UC_ARM_REG_R0, subrecord), (UC_ARM_REG_LR, stop | 1)))
    selection = machine.reg_read(UC_ARM_REG_R0)
    if selection not in range(0x1000, 0x1005):
        raise ValueError(f"unexpected worker selection for the cold fixture: {selection:#x}")
    worker = selection - 0x1000
    token = machine.mem_read(subrecord + 2, 1)[0]

    execute(machine, 0xb059f0, 0xb02a20, zip(REGS, (worker, job, 4)))
    load_args = tuple(machine.reg_read(register) for register in REGS)
    member_base, member_len = Struct("<II").unpack(machine.mem_read(0x01473b6c, 8))
    file_offset = member_base - sp_base
    if not 0 <= file_offset <= len(bundle) - member_len or member_len == 0:
        raise ValueError("firmware registry did not select a member of the mapped bundle")
    execute(machine, 0xb02a20, 0x4cc3ac)
    queue = machine.reg_read(UC_ARM_REG_R0)
    allocation_size = machine.reg_read(UC_ARM_REG_R2)
    # Provide a host scratch pointer for the isolated post-allocation slice.
    message = 0x30009020
    machine.mem_write(machine.reg_read(UC_ARM_REG_R1), U32.pack(message))
    execute(machine, 0xb02a4a, 0x4cbce0)
    if machine.reg_read(UC_ARM_REG_R1) != message or machine.reg_read(UC_ARM_REG_R0) != queue:
        raise ValueError("unexpected destination for the original resource request")
    resource_prefix = bytes(machine.mem_read(message, 0x18))
    completion = resource_completion_slice(machine, resource_prefix, member_base, member_len,
                                           worker, job)

    preload = backup_load(machine, 0x930006, 0x014f9464, 0x20, expected_size=0x10)
    machine.reg_write(UC_ARM_REG_SP, 0x3000b000)
    execute(machine, 0xb05e4c, 0xb02610)
    dest, source, length = (machine.reg_read(register) for register in REGS)
    if length != 0x18:
        raise ValueError(f"unexpected preload request size: {length:#x}")
    preload_message = bytes(machine.mem_read(source, length))
    counter = 0x01473968
    preload_count = U32.unpack(machine.mem_read(counter, 4))[0]

    # Test the nonzero branch separately, without claiming the preload ran.
    staged, ring, ring_state = 0x3000a100, 0x011ec660, 0x014f9493
    slot = original_subrecord.ljust(0x48, b"\0")
    machine.mem_write(staged, slot)
    machine.mem_write(ring_state, bytes(3))
    machine.mem_write(counter, U32.pack(1))
    machine.reg_write(UC_ARM_REG_SP, 0x3000b000)
    execute(machine, 0x50bc04, 0x50bca0,
            ((UC_ARM_REG_R5, staged), (UC_ARM_REG_R7, receiver_frame)))
    if bytes(machine.mem_read(ring, len(slot))) != slot:
        raise ValueError("the nonzero predicate did not preserve the staged record")
    ring_result = machine.reg_read(UC_ARM_REG_R0)
    argument_writes = sa_argument_write_slice(machine, subrecord, worker)

    return {
        "fixture": "cold allocator RAM and one manually mapped archive SP bundle",
        "usecase_mask": hex(mask), "worker_selection": hex(selection),
        "worker": worker, "allocated_request_byte_2": token,
        "sp_member": {"offset": hex(file_offset), "size": member_len,
                      "name": bundle[file_offset + 20:file_offset + 32].decode("ascii"),
                      "sha256": sha256(bundle[file_offset:file_offset + member_len]).hexdigest()},
        "load_boundary": {"pc": hex(0xb02a20), "arguments": list(load_args),
                          "allocation_queue": hex(queue), "allocation_size": allocation_size,
                          "initialized_prefix_hex": resource_prefix.hex(" ")},
        "resource_completion_fixture": completion,
        "sa_argument_write_slice": argument_writes,
        "startup_preload": {"factory_table_hex": preload.hex(" "),
                            "destination": hex(dest), "message_hex": preload_message.hex(" "),
                            "runtime_count_before_send": preload_count},
        "nonzero_counter_fixture": {
            "pc": hex(0x50bca0), "return_code": ring_result,
            "ring_address": hex(ring), "ring_state": list(machine.mem_read(ring_state, 3)),
            "record_prefix_hex": bytes(machine.mem_read(ring, len(original_subrecord))).hex(" "),
        },
    }


def receiver_slices(machine, request_va, backup, bundle):
    # Re-enter offline at the known receive sites with the actual generated bytes.
    machine.reg_write(UC_ARM_REG_SP, 0x3000c000)
    execute(machine, 0x50dd34, 0x50b594, ((UC_ARM_REG_R4, request_va),))
    bulk = [machine.reg_read(register) for register in REGS]
    subrecord = machine.reg_read(UC_ARM_REG_R4)
    length = machine.reg_read(UC_ARM_REG_R10)

    execute(machine, 0x50e080, 0xb03e14, ((UC_ARM_REG_R0, subrecord),))
    forwarded = machine.reg_read(UC_ARM_REG_R0)
    execute(machine, 0xb03e48, 0x50bbd0, ((UC_ARM_REG_R0, forwarded),))
    execute(machine, 0x50bbd0, 0xb02242)
    job = machine.reg_read(UC_ARM_REG_R0)
    row = machine.reg_read(UC_ARM_REG_R2) + job
    mode_byte = machine.reg_read(UC_ARM_REG_R4)
    lookup_output, frame = (machine.reg_read(register) for register in (UC_ARM_REG_R1, UC_ARM_REG_R7))
    execute(machine, 0xb01ff0, 0xb02006)
    execute(machine, 0xb0201a, 0xac9698)
    backup_args = tuple(machine.reg_read(register) for register in REGS)
    backup_id, backup_dst, backup_size = backup_args
    result = {
        "bulk_classifier": {"destination_pc": hex(0x50b594), "arguments": bulk},
        "subrecord_offset": subrecord - request_va, "subrecord_length": length,
        "forwarded_offset": forwarded - request_va,
        "subrecord_handler": hex(0xb03e14), "start_handler": hex(0x50bbd0),
        "lookup_boundary": {"pc": hex(0xb02242), "job": job,
                            "request_byte_7": mode_byte, "address": hex(row)},
        "backup_read": {"pc": hex(0xac9698), "key": hex(backup_id),
                        "destination": hex(backup_dst), "length": backup_size},
    }
    if backup is not None:
        result["factory_lookup"] = factory_lookup(machine, backup, backup_args, job, lookup_output, frame)
        if bundle is not None:
            result["scheduler_slices"] = scheduler_slices(machine, subrecord, job, bundle, frame)
    return result


def probe(blob, *, manager_id, search_type, usecase, window=WINDOW, receiver=False, backup=None,
          bundle=None, side_dictionary_offset=0x19400, pyramid=None, candidate_extent=None):
    if sha256(blob).hexdigest() != CP_SHA256:
        raise ValueError("expected pinned A7 IV 6.02 CP image")
    if manager_id not in range(10) or search_type not in range(256) or usecase not in range(2):
        raise ValueError("expected manager 0..9, a byte search type, and use case 0 or 1")
    if bundle is not None and backup is None:
        raise ValueError("the scheduler slice requires both factory backup and SP bundle")
    if (pyramid is None) != (candidate_extent is None):
        raise ValueError("pyramid records and candidate extent must be supplied together")
    machine = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    pinned = (len(blob) + 0xfff) & ~0xfff
    machine.mem_map(CP_BASE, pinned, UC_PROT_READ | UC_PROT_EXEC)
    machine.mem_write(CP_BASE, blob)
    machine.mem_map(DRIVER, 0x10000, UC_PROT_ALL & ~UC_PROT_EXEC)
    machine.mem_map(PARAM_VA, 0x1000, UC_PROT_ALL & ~UC_PROT_EXEC)
    machine.reg_write(UC_ARM_REG_SP, STACK)

    # Supply only synthetic RAM inputs. No interrupt, I/O, or hardware reads.
    machine.mem_write(MANAGER + 4, bytes((manager_id,)))
    machine.mem_write(MANAGER + 8, U32.pack(PARAM_VA))
    machine.mem_write(MANAGER + 0x18, Struct("<3Q").pack(0x500004000, 0x500010000, 0x500100000))
    machine.mem_write(MANAGER + 0x44, Struct("<4I").pack(0, 0x100, 0, side_dictionary_offset))
    machine.mem_write(INPUT + 4, Struct("<BB2xIIQ").pack(search_type, usecase, 0x13579bdf, 0, 0x500020000))
    machine.mem_write(INPUT + 0x20, Struct("<3H6x6H").pack(*window))
    if pyramid is not None:
        frame_selection = pyramid_slice(machine, pyramid, candidate_extent)
        machine.reg_write(UC_ARM_REG_SP, STACK)

    execute(machine, 0x6fcc8e, 0x6fcd3c, ((UC_ARM_REG_R4, DRIVER),))
    execute(machine, 0x264b5c, 0x264bb4,
            ((UC_ARM_REG_R8, MANAGER), (UC_ARM_REG_R9, MANAGER + 0x20)))
    execute(machine, 0x264f60, 0x264fc8, zip(REGS, (MANAGER, INPUT)))
    execute(machine, 0x713b92, 0x713ba8, zip(REGS, (MANAGER, DESCRIPTOR)))

    machine.reg_write(UC_ARM_REG_SP, STACK)
    execute(machine, 0x27073c, 0x2708d2, zip(REGS, (DRIVER, DESCRIPTOR, 1)))
    request_va = machine.reg_read(UC_ARM_REG_SP) + 0x48
    request = bytes(machine.mem_read(request_va, 0x30))
    execute(machine, 0xb01cc4, 0xb02610, ((UC_ARM_REG_R0, request_va),))
    queue, payload, length = (machine.reg_read(register) for register in REGS)
    if payload != request_va or length != len(request):
        raise ValueError("unexpected pointer or length at queue boundary")

    param = bytes(machine.mem_read(PARAM_VA, 0x28))
    descriptor = bytes(machine.mem_read(DESCRIPTOR, 0x18))
    result = {
        "fixture": {"manager_id": manager_id, "search_type": search_type, "usecase": usecase},
        "manager_search_type_after_build": machine.mem_read(MANAGER + 0x5b, 1)[0],
        "parameter_block_hex": param.hex(" "),
        "cp_descriptor_hex": descriptor.hex(" "),
        "cp_request_hex": request.hex(" "),
        "job": request[0x13], "manager_id_on_wire": request[0x20],
        "sammu_parameter_address": hex(U32.unpack_from(request, 0x24)[0]),
        "sammu_auxiliary_address": hex(U32.unpack_from(request, 0x28)[0]),
        "queue_boundary": {"pc": hex(0xb02610), "destination": hex(queue), "length": length},
    }
    if pyramid is not None:
        result["frame_selection"] = frame_selection
    if receiver or backup is not None:
        result["receiver_slices"] = receiver_slices(machine, request_va, backup, bundle)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("--manager-id", type=int, default=3)
    cli.add_argument("--search-type", type=lambda text: int(text, 0), default=0)
    cli.add_argument("--usecase", type=int, choices=(0, 1), default=1)
    cli.add_argument("--receiver", action="store_true", help="Also run isolated receive-path slices")
    cli.add_argument("--backup-tar", type=Path, help="Read the factory BK4 member for the receiver")
    cli.add_argument("--sp-bundle", type=Path, help="Trace worker and SP lookup with --backup-tar")
    args = cli.parse_args()
    backup = None
    if args.backup_tar:
        with tarfile.open(args.backup_tar, "r:") as tar:
            with tar.extractfile(BACKUP_MEMBER) as file:
                backup = file.read()
    print(json.dumps(probe(args.cp.read_bytes(), manager_id=args.manager_id,
                           search_type=args.search_type, usecase=args.usecase,
                           receiver=args.receiver, backup=backup,
                           bundle=args.sp_bundle.read_bytes() if args.sp_bundle else None), indent=2))
