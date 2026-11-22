"""Run the original FDN A entry on a parameter block built by the original CP."""

import argparse
from collections import deque
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from struct import unpack

from probe_a7iv_fdn_cp_request import WINDOW, probe as cp_probe
from probe_sa_fdn_boot import (
    FIXTURE_BASE, FIXTURE_READS, DmaTraceMachine, consume_a_mailboxes, fixture_wakes,
    probe as boot_probe,
)
from probe_sa_fdn_dispatch import MEMBER_SHA, ROW, word


ENTRY = 0x50000000
PARAMETER_SIZE = 0x28
VALIDATOR = 0x500009bc
VALIDATOR_RETURN = 0x50000af6
NEXT_CALL = 0x50000b54
FIXTURE_PARAMETER_VA = 0xa0000400
SQUARE_WINDOW = (640, 480, 704, 32, 32, 64, 64, 0, 2)
A_TO_B_DICTIONARY_SUBMIT = 0x50002766
A_COMPUTE = {"pc": 0x500008e4, "reason": "unsupported opcode", "opcode": 0xa}
POST_CONV_RECORDS = (0x50000208, 0x50000328, 0x50000468, 0x500004f0,
                     0x50000600, 0x50000694, 0x500007d4)
DICTIONARY_DESTINATIONS = (0x3800, 0x4000, 0x4800, 0x5000, 0xa000, 0xa800)
POST_CONV_REGISTERS = (0x40190, 0x40194, 0x40198, 0x4019c, 0x401a0, 0x401a8,
                       0x401c4, 0x401d0, 0x40218, 0x40320, 0x40330)
DICTIONARY_HASHES = (
    "cbf0ee891b1ec2ce75e4334dca292553004cbc963756688cb7b0b80856844840",
    "26c8577dfeb08e4091b961024013e3607263587fb8259a74b087987c17340f30",
)


def trace_dictionary_request(machine, addresses, dictionaries):
    # A single wake for the preceding image request. No image or weights are
    # copied and no completion status is fabricated.
    machine.pc += 2
    machine.scratch_end = 0x800
    start = len(machine.scratch_trace)
    stop = machine.run(8192)
    address = stop.get("address", -1)
    if ((stop["pc"], stop["reason"], stop.get("size")) != (0x50001d5e, "unmodeled read", 4)
            or not 0x17d00 <= address <= 0x17e1c or address % 4):
        raise ValueError(f"unexpected stop after image request: {stop}")
    writes = machine.scratch_trace[start:]
    if [(e["address"], e["size"]) for e in writes] != [(i * 4, 4) for i in range(len(writes))]:
        raise ValueError("expected sequential descriptor writes")
    values = [e["value"] for e in writes]
    if values[-2:] != [1, 0x40000000] or (len(values) - 2) % 4:
        raise ValueError("unexpected dictionary descriptor terminator")
    if len(dictionaries) != 2 or tuple(sha256(d).hexdigest() for d in dictionaries) != DICTIONARY_HASHES:
        raise ValueError("expected both original A7 IV FDN dictionaries")
    candidates = []
    for name, address, data in zip(("front", "side"), addresses[1:3], dictionaries):
        base = address & 0x3fffffff
        rows = []
        for i in range(0, len(values) - 2, 4):
            source, target, w2, w3 = values[i:i + 4]
            offset, size = source - base, (w2 >> 16) + 1
            if w2 != w3 or w2 & 0xffff or not 0 <= offset <= len(data) - size:
                break
            rows.append((offset, target, size))
        else:
            candidates.append((name, data, rows))
    if len(candidates) != 1:
        raise ValueError("expected requests to select exactly one pinned dictionary")
    name, data, rows = candidates[0]
    slices = b"".join(data[offset:offset + size] for offset, _, size in rows)
    return {"stop": stop, "dictionary": name, "sha256_in_request_order": sha256(slices).hexdigest(),
            "requests": [{"offset": offset, "destination": target, "size": size}
                         for offset, target, size in rows]}


def trace_linked_b(machine, fdn, block_va):
    # Start after A's unresolved register-dependent read, not after a simulated
    # output or handshake. The bootstrap reads a relative link from the member.
    post = deepcopy(machine)
    block = 0x100
    link_table = block + word(fdn, block + 0x1c)
    linked_offset = word(fdn, link_table + 4)
    linked_block, linked_va = block + linked_offset, block_va + linked_offset
    if linked_block != 0x1a800:
        raise ValueError("expected the linked FDN B block")
    post.pc = 0x50000b70
    post.regions = post.regions[:-1] + ((block_va, fdn[block:]),)
    post.fixture_reads = {0x48160: deque((block_va,))}
    entry = post.run(16, stop_at=linked_va)
    if entry != {"pc": linked_va, "reason": "slice boundary"}:
        raise ValueError(f"expected the linked bootstrap: {entry}")
    call_return = post.link
    first = post.run(1024)
    if first != {"pc": linked_va + 0x94, "reason": "unmodeled read",
                 "address": 0x48004, "size": 4}:
        raise ValueError(f"unexpected linked bootstrap device boundary: {first}")
    post.fixture_reads = {addr: deque(values) for addr, values in FIXTURE_READS.items()}
    first_write = len(post.mmio_writes)
    waits = fixture_wakes(post, 2)
    fetch = post.run(1024)
    writes = {w["address"]: w["value"] for w in post.mmio_writes[first_write:]}
    external_va, external_end = writes[0x49d04], writes[0x49d08]
    external_offset = linked_block + word(fdn, linked_block + 0x30)
    external_length = external_end - external_va + 1
    stage_offset = linked_block + word(fdn, linked_block + 0x10)
    catalog_offset = linked_block + word(fdn, linked_block + 0xc)
    if (fetch != {"pc": external_va, "reason": "unmapped fetch", "address": external_va, "size": 2}
            or external_offset + external_length != stage_offset
            or writes[0x48168] != block_va + catalog_offset - block):
        raise ValueError("linked bootstrap does not match the FDN B member layout")
    post.regions += ((external_va, fdn[external_offset:stage_offset]),)
    external_stop = post.run(1024)
    return {"link_table_member_offset": link_table, "linked_block": linked_block, "entry": entry,
            "a_call_return_pc": call_return, "first_device_read": first, "bootstrap_waits": waits,
            "b_catalog": {"member_offset": catalog_offset, "fixture_address": writes[0x48168]},
            "external_mapping": {"member_offset": external_offset, "address": external_va,
                                 "size": external_length}, "external_stop": external_stop}


def mount_catalog(post, fdn):
    block = 0x100
    table = block + word(fdn, block + 0xc)
    boot = boot_probe(fdn, "A", fixture="header-only")
    writes = {w["address"]: w["value"] for w in boot["mmio_writes"]}
    va, block_va = writes[0x48168], writes[0x48160]
    if block_va != FIXTURE_BASE or va - block_va != table - block:
        raise ValueError("boot catalog pointer does not resolve to the original member")
    post.regions += ((va, fdn[table:0x1a800]),)
    catalog_writes = [w for w in boot["mmio_writes"] if w["address"] == 0x48168]
    post.fixture_reads[0x48168] = deque([va] * 128)
    return {"member_offset": table, "fixture_address": va, "boot_writes": catalog_writes}, block_va


def trace_program_overlays(post, fdn):
    """Follow only the original control code with the boot-produced catalog pointer."""
    catalog, block_va = mount_catalog(post, fdn)
    table, va = (catalog[k] for k in ("member_offset", "fixture_address"))
    post.extra_scratch = ((0, 0x800),)
    requests, calls, waits = [], [], []
    for _ in range(48):
        first = len(post.mmio_writes)
        stop = post.run(8192)
        for entry in post.mmio_writes[first:]:
            if entry["address"] != 0x48234:
                continue
            descriptor = entry["value"]
            src, dst, a, b = (post.read(descriptor + i, 4) for i in range(0, ROW.size, 4))
            if not (0x04000000 <= dst < 0x04006000 and dst % 0x1000 == 0):
                continue
            if post.read(descriptor + 0x10, 4) != 1 or post.read(descriptor + 0x14, 4) != 0x40000000:
                raise ValueError("expected one program transfer before the terminator")
            offset = src - va + table
            size = ((a >> 16) & 0x3fff) + 1
            if not table + 0x180 <= offset <= 0x1a800 - size:
                raise ValueError("program transfer source outside FDN A")
            requests.append({"wait_pc": stop["pc"], "member_offset": offset,
                             "destination": dst, "words": [src, dst, a, b],
                             "source_sha256": sha256(fdn[offset:offset + size]).hexdigest()})
        if stop["reason"] == "wfi":
            waits.append(stop["pc"])
            post.pc += 2  # A wait boundary only, not an executed transfer.
            continue
        if stop == {"pc": 0x500008e4, "reason": "unsupported opcode", "opcode": 0x000a}:
            calls.append({"record_r4": post.reg[4], "return_pc": post.link,
                          "program_requests_before_call": len(requests)})
            # Start another independent caller slice; no compute state is supplied.
            post.pc, post.reg[0] = post.link, 0
            continue
        break
    else:
        raise ValueError("did not reach a new device boundary while tracing the overlays")
    result = {"catalog": catalog, "program_requests": requests, "isolated_calls": calls,
              "wait_pcs": waits, "stop": stop}
    if stop == {"pc": 0x50000b5c, "reason": "unmodeled read", "address": 0x48100, "size": 4}:
        transfer, = post.dma_submissions
        result["a_to_b_dictionary_submit"] = transfer
        result["isolated_a_to_b_handoff"] = trace_linked_b(post, fdn, block_va)
    return result


def trace_outer_continuation(machine, fdn, *, trace_catalog=False):
    """Isolate the original caller after the first unknown wrapper."""
    variants = []
    for value in (0, 0xdeadbeef):
        post = deepcopy(machine)
        # Start a separate slice at the saved return PC. No unknown instruction
        # is executed and no computed feature buffer is supplied.
        post.pc, post.reg[0] = post.link, value
        post.scratch_base, post.scratch_end = 0, 0x800
        waits = []
        for expected_pc in (0x50001eac, 0x5000232c, 0x50002810):
            stop = post.run(8192)
            if stop != {"pc": expected_pc, "reason": "wfi"}:
                raise ValueError(f"unexpected FDN continuation stop: {stop}")
            waits.append(stop["pc"])
            post.pc += 2  # Synthetic wake; do not perform the requested transfer.
        descriptor = [post.read(i, 4) for i in range(0, 0x18, 4)]
        post.scratch_base, post.scratch_end = machine.scratch_base, machine.scratch_end
        stop = post.run(8192)
        if stop != {"pc": 0x50000f28, "reason": "wfi"}:
            raise ValueError(f"unexpected FDN follow-up mailbox stop: {stop}")
        waits.append(stop["pc"])
        mailboxes = [{"address": addr, "selector": post.read(addr, 4)}
                     for addr in (0x17800 + bank * 0xe4 for bank in range(5))]
        workers = consume_a_mailboxes(fdn, post.scratch, len(mailboxes))
        workers = [{"bank": w["bank"], "stop": w["stop"], "mailbox_reads": len(w["reads"]),
                    "nonzero_payload_reads": sum(e["value"] != 0 for e in w["reads"][1:])}
                   for w in workers]
        post.pc += 2
        stop = post.run(8192)
        variants.append({"wait_pcs": waits, "intermediate_descriptor_words": descriptor,
                         "followup_mailbox_heads": mailboxes, "followup_workers": workers,
                         "stop": stop})
    if variants[0] != variants[1]:
        raise ValueError("chosen return register values changed the FDN continuation")
    result = {**variants[0], "fixture_return_pc": machine.link,
              "fixture_r0_values": [0, 0xdeadbeef]}
    if trace_catalog:
        if post.pc != 0x500010c8:
            raise ValueError("did not reach the catalog lookup")
        result["program_catalog_continuation"] = trace_program_overlays(post, fdn)
    return result


def seed_boot_control(machine, fdn, block, bank_src, request, dictionaries):
    table = block + word(fdn, block + 0x20)
    offset, target, a, b = ROW.unpack_from(fdn, table + 5 * ROW.size)
    source, size = bank_src + offset, (b >> 16) + 1
    if (source, target, size, a, b) != (0x6b00, 0x7800, 0x460, 0x045f4000, 0x045f0000):
        raise ValueError("unexpected A boot control-table descriptor")
    machine.extra_scratch = ((0, 0x800), (target, target + size), (0x3800, target), (0xa000, 0x17800))
    machine.scratch.update(enumerate(fdn[source:source + size], target))
    data = dictionaries[request["dictionary"] == "side"]
    for transfer in request["requests"]:
        start, dst, count = (transfer[k] for k in ("offset", "destination", "size"))
        if not machine.in_scratch(dst, count):
            raise ValueError("A dictionary transfer is outside known local memory")
        machine.scratch.update(enumerate(data[start:start + count], dst))
    return {"member_offset": source, "destination": target, "size": size}


def map_tail_dictionary(post, rows, name, dictionaries):
    side = name == "side"
    coeff_base = 0x20080000 if side else 0x20040000
    data = dictionaries[side]
    if len(rows) != 48:
        raise ValueError("expected 48 A tail dictionary requests")
    for src, dst, a, b in rows:
        offset, size = src - coeff_base, (a >> 16) + 1
        if a != b or a & 0xffff or not 0 <= offset <= len(data) - size or not post.in_scratch(dst, size):
            raise ValueError(f"unexpected A tail dictionary row {(src, dst, a, b)}")
        post.scratch.update(enumerate(data[offset:offset + size], dst))


def trace_a_later_stages(post, fdn, name, dictionaries, fixture_r0):
    catalog, _ = mount_catalog(post, fdn)
    post.capture_dma, post.dma_submit_pcs = True, ()
    post.pc, post.reg[0] = post.link, fixture_r0
    calls, programs, waits = [], [], []
    slots = None
    tail_submit, final_workers = None, None
    for _ in range(96):
        submitted = len(post.dma_submissions)
        stop = post.run(8192)
        for request in post.dma_submissions[submitted:]:
            site, rows = (request[k] for k in ("submit_pc", "rows"))
            if site == A_TO_B_DICTIONARY_SUBMIT:
                map_tail_dictionary(post, rows, name, dictionaries)
                tail_submit = request
            if len(rows) == 1 and 0x04000000 <= rows[0][1] < 0x04006000:
                src, dest, a, b = rows[0]
                plane_bytes = ((b >> 16) & 0x3fff) + 1
                source_bytes = ((a >> 16) & 0x3fff) + 1
                if (dest % 0x1000 or a >> 30 != 2 or b >> 30 != 1 or
                        a & 0xffff or b & 0xffff != 0x200 or
                        plane_bytes % 8 or source_bytes != 3 * plane_bytes):
                    raise ValueError("unexpected A program-plane request")
                count = plane_bytes // 8
                if dest == 0x04000000:
                    slots = count
                programs.append({"slots": count, "destination": dest,
                                 "member_offset": src - catalog["fixture_address"] + catalog["member_offset"]})
        if stop == A_COMPUTE and post.reg[4] in POST_CONV_RECORDS:
            state = {w["address"]: w["value"] for w in post.mmio_writes}
            calls.append({"registers_r4_to_r7": post.reg[4:8], "return_pc": post.link,
                          "program_slots": slots, "register_snapshot": {
                              hex(addr): state[addr] for addr in POST_CONV_REGISTERS if addr in state}})
            # Independent control continuation; no compute or DMA of unknown memory.
            post.pc, post.reg[0] = post.link, fixture_r0
        elif stop["reason"] == "wfi":
            waits.append(stop["pc"])
            if stop["pc"] == 0x50000f28 and calls and calls[-1]["registers_r4_to_r7"][0] == 0x500007d4:
                regions = tuple((a, bytes(post.scratch[i] for i in range(a, a + 0x538)))
                                for a in DICTIONARY_DESTINATIONS)
                final_workers = consume_a_mailboxes(fdn, post.scratch, 5, data_regions=regions, call_count=3)
            post.pc += 2
        else:
            break
    else:
        raise ValueError("A later control did not reach a new boundary")
    return {"stop": stop, "waits": waits, "calls": calls, "program_requests": programs,
            "dictionary_submit": tail_submit, "final_workers": final_workers,
            "dma_submissions": post.dma_submissions}


def trace_boot_control_continuation(machine, fdn, *, later_dictionary=None, dictionaries=None):
    variants = []
    for value in (0, 0xdeadbeef):
        post = deepcopy(machine)
        # Begin at the original return address; no first-compute output is provided.
        post.pc, post.reg[0] = post.link, value
        waits = fixture_wakes(post, 3)
        descriptor = [post.read(i, 4) for i in range(0, 0x18, 4)]
        first_write = len(post.mmio_writes)
        stop = post.run(8192)
        if stop != A_COMPUTE:
            raise ValueError(f"unexpected second A compute boundary: {stop}")
        # The first dictionary batch placed six disjoint coefficient spans here.
        regions = tuple((start, bytes(post.scratch[i] for i in range(start, start + 0x538)))
                        for start in DICTIONARY_DESTINATIONS)
        workers = consume_a_mailboxes(fdn, post.scratch, 5, data_regions=regions, trace_compute=True)
        variants.append({"waits": waits, "stop": stop, "return_pc": post.link,
                         "registers_r4_to_r7": post.reg[4:8], "preceding_transfer_words": descriptor,
                         "aperture_writes": [w for w in post.mmio_writes[first_write:]
                                             if 0x40000 <= w["address"] < 0x40400],
                         "worker_calls": workers})
        if later_dictionary:
            variants[-1]["later_control"] = trace_a_later_stages(post, fdn, later_dictionary, dictionaries, value)
    if variants[0] != variants[1]:
        raise ValueError("chosen return register values changed the second A compute boundary")
    return {**variants[0], "fixture_r0_values": [0, 0xdeadbeef]}


def trace_first_workload(machine, fdn, block, bank_src, *, trace_catalog=False, boot_table=None,
                         later_dictionary=None, dictionaries=None):
    # Boot descriptor 10 is a stored local table, independent of the requested
    # image and dictionary pixels. Map only its original bytes.
    descriptor = block + word(fdn, block + 0x20) + 10 * ROW.size
    offset, target, w2, w3 = ROW.unpack_from(fdn, descriptor)
    src, size = bank_src + offset, (w2 >> 16) + 1
    if (target, size, w2 & 0xffff, w3) != (0x17d00, 0x120, 0x4000, w2 & 0xffff0000):
        raise ValueError("unexpected A/10 boot table descriptor")
    table = fdn[src:src + size]
    machine.regions += ((target, table),)
    machine.scratch_base, machine.scratch_end = 0x17800, 0x17d00
    first_scratch, first_mmio = len(machine.scratch_trace), len(machine.mmio_writes)
    stop = machine.run(8192)
    if stop != {"pc": 0x500008e4, "reason": "unsupported opcode", "opcode": 0x000a}:
        raise ValueError(f"unexpected first workload stop: {stop}")
    entries = []
    for bank in range(5):
        addr = 0x17800 + 0xe4 * bank
        entries.append({"address": addr, "selector": machine.read(addr, 4)})
    pointer = machine.reg[4]
    external_src = block + word(fdn, block + 0x30)
    command_start, wrapper = (external_src + va - ENTRY for va in (pointer, stop["pc"]))
    commands = fdn[command_start:wrapper]
    local_start = bank_src + 0x34
    if not commands or commands != fdn[local_start:local_start + len(commands)]:
        raise ValueError("expected the same command bytes in the outer and first bank code")
    result = {"table": {"file_offset": src, "address": target, "size": size,
                        "sha256": sha256(table).hexdigest()},
              "stop": stop, "command_pointer_r4": pointer, "return_address": machine.link,
              "matching_command_bytes": {"outer_file_offset": command_start,
                                         "bank_file_offset": local_start, "size": len(commands),
                                         "sha256": sha256(commands).hexdigest()},
              "mailbox_heads": entries, "mailbox_writes": machine.scratch_trace[first_scratch:],
              "register_writes": machine.mmio_writes[first_mmio:],
              "banks_after_fixture_wake": consume_a_mailboxes(fdn, machine.scratch, 5,
                                                             trace_compute=boot_table is not None)}
    if boot_table is None:
        result["caller_continuation"] = trace_outer_continuation(machine, fdn, trace_catalog=trace_catalog)
    else:
        result["boot_control_fixture"] = {
            "table": boot_table, "first_registers_r4_to_r7": machine.reg[4:8],
            "isolated_after_first_compute": trace_boot_control_continuation(
                machine, fdn, later_dictionary=later_dictionary, dictionaries=dictionaries),
        }
    return result


def run_fdn_parameters(fdn, parameter_address, parameter_bytes, address_table, *, trace_setup=False,
                       dictionaries=None, trace_catalog=False, boot_control=False, trace_later_stages=False):
    if sha256(fdn).hexdigest() != MEMBER_SHA:
        raise ValueError("expected pinned A7 IV FDN member")
    if len(parameter_bytes) != PARAMETER_SIZE or len(address_table) != 0x160:
        raise ValueError("expected the 40-byte parameter block and 352-byte address table")
    if boot_control and (not dictionaries or trace_catalog):
        raise ValueError("boot control requires dictionaries and a separate run from catalog tracing")
    if trace_later_stages and not boot_control:
        raise ValueError("later stages require the boot-control fixture")
    block = 0x100
    ext = block + word(fdn, block + 0x30)
    bank_src = block + word(fdn, block + 0x10)
    image = fdn[ext:bank_src]
    stack = word(image, 0x10)
    machine = DmaTraceMachine(image)
    machine.capture_dma = trace_catalog
    machine.dma_submit_pcs = (A_TO_B_DICTIONARY_SUBMIT,)
    machine.pc = ENTRY
    # This is a fixture mapping, not evidence for how the CP loads the live table.
    machine.regions = ((ENTRY, image), (0x49e00, address_table),
                       (FIXTURE_PARAMETER_VA, parameter_bytes))
    machine.stack_low, machine.stack_high = stack - 0x1000, stack
    machine.stack = bytearray(0x1000)
    machine.reg[4] = parameter_address
    first = machine.run(8192, stop_at=VALIDATOR)
    result = {"fixture_table_sha256": sha256(address_table).hexdigest(),
              "cp_parameter_address": parameter_address,
              "cp_parameter_hex": parameter_bytes.hex(" "),
              "validator_entry": first,
              "converted_parameter_address": machine.reg[4]}
    if first["reason"] != "slice boundary":
        return result
    second = machine.run(8192, stop_at=VALIDATOR_RETURN)
    if second["reason"] != "slice boundary":
        raise ValueError(f"unexpected FDN validator stop: {second}")
    error_mask = machine.reg[0]
    result["validator_mask"] = error_mask
    if error_mask != 0:
        result["stop"] = machine.run(128, stop_at=ENTRY + 8)
        result["mmio_writes"] = machine.mmio_writes
        return result

    result["stop"] = machine.run(8192, stop_at=NEXT_CALL)
    if result["stop"]["reason"] == "slice boundary":
        address = machine.reg[4]
        copied = bytes(machine.read(address + i, 1) for i in range(PARAMETER_SIZE))
        result["normalized_parameter_hex"] = copied.hex(" ")
        result["converted_addresses"] = list(unpack("<4I", copied[:0x10]))
        if trace_setup or dictionaries is not None:
            write_start = len(machine.mmio_writes)
            scratch_start = len(machine.scratch_trace)
            result["setup_stop"] = machine.run(8192)
            result["setup_writes"] = machine.mmio_writes[write_start:]
            result["setup_scratch_writes"] = machine.scratch_trace[scratch_start:]
        if dictionaries is not None and result["setup_stop"] == {"pc": 0x50002524, "reason": "wfi"}:
            result["dictionary_fixture"] = trace_dictionary_request(
                machine, result["converted_addresses"], dictionaries)
            boot_table = (seed_boot_control(machine, fdn, block, bank_src, result["dictionary_fixture"], dictionaries)
                          if boot_control else None)
            result["first_workload_fixture"] = trace_first_workload(
                machine, fdn, block, bank_src, trace_catalog=trace_catalog, boot_table=boot_table,
                later_dictionary=result["dictionary_fixture"]["dictionary"] if trace_later_stages else None,
                dictionaries=dictionaries)
    return result


def probe(cp, fdn, *, window=SQUARE_WINDOW, trace_setup=False, dictionaries=None, trace_catalog=False,
          boot_control=False, trace_later_stages=False):
    # Let the original CP encode two disjoint synthetic dictionary buffers for
    # the trace; offsets remain unchanged for the original packing fixture.
    offset = 0x100000 if dictionaries is not None else 0x19400
    request = cp_probe(cp, manager_id=3, search_type=0, usecase=1, window=window,
                       side_dictionary_offset=offset)
    return run_fdn_parameters(fdn, int(request["sammu_parameter_address"], 16),
                              bytes.fromhex(request["parameter_block_hex"]), bytes(0x160),
                              trace_setup=trace_setup, dictionaries=dictionaries, trace_catalog=trace_catalog,
                              boot_control=boot_control, trace_later_stages=trace_later_stages)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("fdn", type=Path)
    window_args = cli.add_mutually_exclusive_group()
    window_args.add_argument("--original-window", action="store_true",
                             help="Use the old rectangular CP packing fixture as a negative control")
    window_args.add_argument("--window", type=int, nargs=6, metavar=("WIDTH", "HEIGHT", "PITCH", "X", "Y", "EDGE"),
                             help="Use an explicit square-window CP fixture")
    cli.add_argument("--trace-setup", action="store_true",
                     help="Continue until the first unmodeled operation after validation")
    cli.add_argument("--dict-dir", type=Path, help="Check descriptor spans against both original dictionaries")
    cli.add_argument("--trace-catalog", action="store_true",
                     help="Trace later original program-plane requests with explicit synthetic wakes")
    cli.add_argument("--boot-control", action="store_true", help="Replay the boot table up to the next A compute")
    cli.add_argument("--trace-later-stages", action="store_true",
                     help="Follow later original A control slices without executing the datapath")
    cli.add_argument("--node", type=int, default=2, help="Parameter block node (default: 2)")
    args = cli.parse_args()
    if not 0 <= args.node <= 0xffff:
        cli.error("--node must fit a 16-bit field")
    if (args.trace_catalog or args.boot_control) and args.dict_dir is None:
        cli.error("--trace-catalog and --boot-control require --dict-dir")
    if args.trace_catalog and args.boot_control:
        cli.error("--trace-catalog and --boot-control must be run separately")
    if args.trace_later_stages and not args.boot_control:
        cli.error("--trace-later-stages requires --boot-control")
    if args.window is not None:
        if not all(0 <= word <= 0xffff for word in args.window):
            cli.error("--window fields must fit in 16 bits")
        width, height, pitch, x, y, edge = args.window
        window = (width, height, pitch, x, y, edge, edge, 0, args.node)
    else:
        window = (*(WINDOW if args.original_window else SQUARE_WINDOW)[:-1], args.node)
    dictionaries = None if args.dict_dir is None else tuple(
        (args.dict_dir / f"fdn_{name}_dict.bin").read_bytes() for name in ("front", "side"))
    print(json.dumps(probe(args.cp.read_bytes(), args.fdn.read_bytes(),
                           window=window, trace_setup=args.trace_setup,
                           dictionaries=dictionaries, trace_catalog=args.trace_catalog,
                           boot_control=args.boot_control, trace_later_stages=args.trace_later_stages), indent=2))
