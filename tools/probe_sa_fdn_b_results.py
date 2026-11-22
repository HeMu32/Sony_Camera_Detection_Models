"""Run the pinned FDN B control and result writer, without executing the datapath."""

import argparse
from collections import deque
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from struct import pack

from probe_a7iv_fdn_cp_request import probe as cp_probe
from probe_sa_fdn_boot import BootMachine, DMA_POINTER, DmaTraceMachine, FIXTURE_BASE, fixture_wakes
from probe_sa_fdn_dispatch import MEMBER_SHA, ROW, executable_banks, word
from probe_sa_fdn_parameters import DICTIONARY_HASHES, SQUARE_WINDOW, probe as probe_a
from sa_local_dma import copy_local_rows


ENTRY = 0x50800000
WRITER = 0x50801de4
FIXTURE_RETURN = 0x50803c00
OUTPUT_PAGE = 0xe0004000
RESULT_BYTES = 0x4c
BLOCK = 0x1a800
B_CATALOG_REG = 0x48168
B_PADDING_SUBMIT = 0x50802b12
B_PADDING_SOURCE = (0x17f80, 0x17fb0)
B_PADDING_DESTINATION = (0xb800, 0xfc10)
WORKER_MAILBOX = 0x17800
WORKER_MAILBOX_BYTES = 0xb8
COMPACT_WEIGHTS_SUBMIT = 0x50802756
PREHEAD_SUBMITS = (0x50802898, COMPACT_WEIGHTS_SUBMIT)
B_LOCAL_DATA = ((0x3800, 0x7800), (0xa000, 0x17800))
B_COMPUTE = {"pc": 0x50800760, "reason": "unsupported opcode", "opcode": 0xa}
PACKED_RECORD = 0x5080013c
PACKED_DESTINATION = (0x12800, 0x138e0)
PACKED_COEFFICIENT_BYTES = 168
PREHEAD_RECORD = 0x50800640
COMPACT_RECORD = 0x50800720
PREHEAD_COEFFICIENT_BYTES = 360


class FdnBMachine(DmaTraceMachine):
    def __init__(self, image):
        super().__init__(image, scratch_base=0x17800, scratch_size=0x500)
        self.dictionary_ranges = ()
        self.dictionary_reads = []
        self.boot_bias_pointer = None
        self.boot_bias_reads = []

    def read(self, address, size, *, fetch=False, sign=False):
        value = super().read(address, size, fetch=fetch, sign=sign)
        if not fetch:
            if address == self.boot_bias_pointer and size == 4:
                self.boot_bias_reads.append({"pc": self.pc, "address": address, "value": value})
            for name, start, end in self.dictionary_ranges:
                if start <= address <= end - size:
                    self.dictionary_reads.append({"pc": self.pc, "dictionary": name,
                                                  "offset": address - start, "size": size, "value": value})
                    break
        return value


class PreheadWorker(BootMachine):
    bias_pointer = None
    first_bias_read = None

    def read(self, address, size, *, fetch=False, sign=False):
        value = super().read(address, size, fetch=fetch, sign=sign)
        if not fetch and address == self.bias_pointer and size == 4 and self.first_bias_read is None:
            self.first_bias_read = {"pc": self.pc, "address": address, "value": value}
        return value


def build_machine(fdn):
    if sha256(fdn).hexdigest() != MEMBER_SHA:
        raise ValueError("expected pinned A7 IV FDN member")
    start, end = (BLOCK + word(fdn, BLOCK + p) for p in (0x30, 0x10))
    table = BLOCK + word(fdn, BLOCK + 0x20)
    src, dst, a, b = ROW.unpack_from(fdn, table + 10 * ROW.size)
    if (dst, a, b) != (0x17d00, 0x011f4000, 0x011f0000):
        raise ValueError("pinned B table arrangement changed")
    image = fdn[start:end]
    m = FdnBMachine(image)
    m.regions = ((ENTRY, image), (dst, fdn[end + src:end + src + 0x120]))
    m.stack_high = word(image, 0x10)
    m.stack_low = m.stack_high - 0x1000
    m.stack = bytearray(0x1000)
    m.extra_scratch = ((0, 0x800), (OUTPUT_PAGE, OUTPUT_PAGE + 0x1000))
    return m


def probe_writer(fdn, normalized, raw_result, *, direction, matrix_group, route_mode):
    """Run only the result writer with explicit synthetic incoming values."""
    if len(normalized) != 0x28 or len(raw_result) != 8:
        raise ValueError("expected original layout parameter and 8 temporary result words")
    machine = build_machine(fdn)
    param, raw, out = 0x6000, 0x6100, OUTPUT_PAGE + 0x40
    machine.regions += ((param, normalized), (raw, pack("<8I", *raw_result)))
    sp = machine.stack_high - 0x80
    machine.reg[4:8] = [param, raw, direction, matrix_group]
    machine.reg[15] = sp
    machine.write(sp, 4, route_mode)
    machine.write(sp + 4, 4, out)
    machine.pc, machine.link = WRITER, FIXTURE_RETURN
    stop = machine.run(1024, stop_at=FIXTURE_RETURN)
    if stop != {"pc": FIXTURE_RETURN, "reason": "slice boundary"}:
        raise ValueError(f"isolated B result writer stopped unexpectedly: {stop}")
    writes = [e for e in machine.scratch_trace if out <= e["address"] < out + 0x20]
    return {"stop": stop, "writes": writes, "words": [machine.read(out + i, 4) for i in range(0, 0x20, 4)]}


def describe_worker_call(machine, stop, image_base, file_base):
    wrapper = deepcopy(machine)
    wrapper.pc += 2  # Isolated known instructions between the two unknown opcodes.
    body_stop = wrapper.run(32)
    start = machine.reg[4]
    size = stop["pc"] - start
    if (body_stop.get("opcode") != 9 or wrapper.reg[0] != start or
            size <= 0 or start < image_base or size % 4):
        raise ValueError("expected a bounded worker record")
    raw = bytes(machine.read(start + i, 1, fetch=True) for i in range(size))
    return {"record_member_offset": file_base + start - image_base, "record_hex": raw.hex(" "),
            "stop_at_000a": stop, "isolated_stop_at_0009": body_stop,
            "registers_r5_to_r7": wrapper.reg[5:8]}


def worker_machine(fdn, scratch, bank, region, machine_class=BootMachine):
    mailbox = WORKER_MAILBOX + bank * WORKER_MAILBOX_BYTES
    base, src, size = (region[k] for k in ("destination", "file_offset", "length"))
    data = fdn[src:src + size]
    m = machine_class(data, scratch_base=mailbox, scratch_size=WORKER_MAILBOX_BYTES)
    m.regions = ((base, data),)
    m.pc = base
    m.stack_low, m.stack_high = base + size, region["stack"]
    m.stack = bytearray(m.stack_high - m.stack_low)
    m.scratch = {a: b for a, b in scratch.items() if mailbox <= a < mailbox + WORKER_MAILBOX_BYTES}
    return m, mailbox, base, src


def run_head_workers(fdn, scratch):
    workers = []
    for bank, region in enumerate(executable_banks(fdn, BLOCK)):
        if bank == 3:
            break
        m, _, base, src = worker_machine(fdn, scratch, bank, region)
        wait, = fixture_wakes(m, 1)
        stop = m.run(256)
        if stop.get("opcode") != 10:
            raise ValueError(f"B head worker did not reach compute boundary: {stop}")
        setup = {w["address"]: w["value"] for w in m.mmio_writes}
        call = describe_worker_call(m, stop, base, src)
        workers.append({"bank": bank, "wait": wait, "mailbox_reads": m.scratch_reads,
                        "count_word": setup[0x40000], "bias_word": setup[0x40004], **call})
    return workers


def read_dictionary_transfers(rows, dictionaries):
    bases = (0x20040000, 0x20080000)
    transfers = []
    for src, dst, a, b in rows:
        if a != b or a & 0xffff:
            continue
        size = (a >> 16) + 1
        for name, base, data in zip(("front", "side"), bases, dictionaries):
            if 0 <= src - base <= len(data) - size:
                transfers.append({"dictionary": name, "offset": src - base,
                                  "destination": dst, "size": size,
                                  "sha256": sha256(data[src - base:src - base + size]).hexdigest()})
                break
    return transfers


def map_dictionary_requests(machine, transfers, dictionaries):
    for transfer in transfers:
        start, size, destination = (transfer[k] for k in ("offset", "size", "destination"))
        if not any(lo <= destination <= hi - size for lo, hi in B_LOCAL_DATA):
            raise ValueError(f"dictionary transfer outside B local data: {destination:#x}+{size:#x}")
        data = dictionaries[transfer["dictionary"] == "side"]
        machine.scratch.update(enumerate(data[start:start + size], destination))


def seed_boot_control(machine, cp, fdn, dictionaries, window):
    table = BLOCK + word(fdn, BLOCK + 0x20)
    offset, destination, w2, w3 = ROW.unpack_from(fdn, table + 5 * ROW.size)
    src = BLOCK + word(fdn, BLOCK + 0x10) + offset
    size = (w3 >> 16) + 1
    if (destination, src, size, w2, w3) != (0x7800, 0x21310, 0x540, 0x053f4000, 0x053f0000):
        raise ValueError("unexpected B boot control-table descriptor")
    machine.extra_scratch += (*B_LOCAL_DATA, (destination, destination + size))
    machine.scratch.update(enumerate(fdn[src:src + size], destination))
    zero_offset, zero_dst, za, zb = ROW.unpack_from(fdn, table + 11 * ROW.size)
    zero_src = BLOCK + word(fdn, BLOCK + 0x10) + zero_offset
    zero_size = (zb >> 16) + 1
    if (zero_src, zero_dst, zero_dst + zero_size, za, zb) != (
            0x23630, *B_PADDING_SOURCE, 0x002f4000, 0x002f0000):
        raise ValueError("unexpected B boot padding-source descriptor")
    machine.scratch.update(enumerate(fdn[zero_src:zero_src + zero_size], zero_dst))
    a = probe_a(cp, fdn, dictionaries=dictionaries, window=window,
                boot_control=True, trace_later_stages=True)
    early = a["first_workload_fixture"]["boot_control_fixture"]["isolated_after_first_compute"]
    submit = early["later_control"]["dictionary_submit"]
    transfers = read_dictionary_transfers(submit["rows"], dictionaries)
    if len(transfers) != len(submit["rows"]) or len(transfers) != 48:
        raise ValueError("expected 48 known A dictionary requests before the B slice")
    map_dictionary_requests(machine, transfers, dictionaries)
    machine.boot_bias_pointer = 0x3b3c
    return {"boot_table": {"member_offset": src, "destination": destination, "size": size},
            "padding_source": {"member_offset": zero_src, "destination": zero_dst, "size": zero_size},
            "a_dictionary_submission": submit, "a_dictionary_transfers": transfers}


def materialize_boot_padding(machine, request, stop):
    rows = request["rows"]
    if stop != {"pc": 0x50802b1c, "reason": "wfi"} or len(rows) != 36:
        raise ValueError(f"unexpected initial B padding boundary: {stop}")
    updates = copy_local_rows(rows, machine.scratch, source_span=B_PADDING_SOURCE,
                              destination_span=B_PADDING_DESTINATION)
    machine.scratch.update(updates)
    return len(updates)


def boot_padding_coverage(machine, dma_bytes):
    lo, hi = B_PADDING_DESTINATION
    scalar = {i for w in machine.scratch_trace if lo <= w["address"] <= hi - w["size"]
              for i in range(w["address"], w["address"] + w["size"])}
    known = {i for i in range(lo, hi) if i in machine.scratch}
    border = {lo + 2 * ((plane * 22 + y) * 22 + x) + byte
              for plane in range(18) for y in range(22) for x in range(22)
              if y in (0, 21) or x in (0, 21) for byte in (0, 1)}
    return {"destination": lo, "size": hi - lo, "dma_bytes": dma_bytes,
            "scalar_written_bytes": len(scalar), "known_bytes": len(known),
            "unknown_bytes": hi - lo - len(known),
            "first_unknown": next((i for i in range(lo, hi) if i not in known), None),
            "matches_zero_border_18x22x22": known == border and not any(machine.scratch[i] for i in known)}


def run_prehead_workers(fdn, scratch, known_dictionary_regions):
    workers = []
    for bank, region in enumerate(executable_banks(fdn, BLOCK)):
        m, mailbox, base, src = worker_machine(fdn, scratch, bank, region)
        offsets = (0, 0x3c, 0x9c, 0xa0, 0xa4)
        selector, gate, weight, bias, subselector = (m.read(mailbox + off, 4) for off in offsets)
        if (selector, gate, subselector) != (3, 0, 0):
            raise ValueError("expected an unaltered selector-3 pre-head mailbox with zero gate")
        wait, = fixture_wakes(m, 1)
        stop = m.run(1024)
        if stop != wait:
            raise ValueError(f"unmodified pre-head worker did not return to wait: {stop}")

        # Differential branch test only, using known requested dictionary bytes.
        # No activations, unknown DMA routes, or completed computation are modeled.
        control, _, _, _ = worker_machine(fdn, scratch, bank, region, PreheadWorker)
        control.regions += known_dictionary_regions
        control.bias_pointer = bias
        control.write(mailbox + 0x3c, 4, 1)
        fixture_wakes(control, 1)
        alternative = control.run(1024)
        if alternative.get("opcode") != 10 or control.first_bias_read is None:
            raise ValueError(f"perturbed pre-head worker missed bias read or compute boundary: {alternative}")
        call = describe_worker_call(control, alternative, base, src)
        call["first_bias_read"] = control.first_bias_read
        workers.append({"bank": bank, "mailbox": mailbox, "weight_pointer": weight, "bias_pointer": bias,
                        "gate_word": gate, "stop": stop, "perturbed_gate_one": call})
    return workers


def scheduled_call(machine, dictionary_bindings, coefficient_bytes):
    cfg = {w["address"]: w["value"] for w in machine.mmio_writes}
    activation, weight, result = machine.reg[5:8]
    for binding in reversed(dictionary_bindings):
        start, length = binding["destination"], binding["size"]
        delta = weight - start
        if 0 <= delta <= length - coefficient_bytes:
            return {"input_pointer": activation, "weight_pointer": weight, "result_argument": result,
                    "dictionary": binding["dictionary"], "dictionary_offset": binding["offset"] + delta,
                    "aperture_40330": cfg[0x40330], "aperture_40218": cfg[0x40218],
                    "bias_word": cfg[0x40320]}
    raise ValueError(f"no submitted coefficient span for {weight:#x}")


def run_scheduled_workers(fdn, machine, bindings, *, expected_selector, expected_gate, coefficient_bytes):
    workers = []
    local_data = {a: v for a, v in machine.scratch.items() if any(lo <= a < hi for lo, hi in B_LOCAL_DATA)}
    for bank, region in enumerate(executable_banks(fdn, BLOCK)):
        worker, mailbox, base, src = worker_machine(fdn, machine.scratch, bank, region)
        worker.extra_scratch = B_LOCAL_DATA
        worker.scratch.update(local_data)
        gate = worker.read(mailbox + 0x3c, 4)
        selector, subselector = (worker.read(mailbox + off, 4) for off in (0, 0xa4))
        if (selector, gate) != (expected_selector, expected_gate):
            raise ValueError(f"unexpected B compute mailbox in bank {bank}: {selector}, {subselector}, {gate}")
        wait, = fixture_wakes(worker, 1)
        scheduled = []
        for _ in range(gate):
            stop = worker.run(2048)
            if stop.get("opcode") != 0xa:
                raise ValueError(f"B worker missed scheduled compute: {stop}")
            scheduled.append({"record_member_offset": src + worker.reg[4] - base,
                              "stop": stop, **scheduled_call(worker, bindings, coefficient_bytes)})
            # Follow the original return control only, with no computed buffer.
            worker.pc, worker.reg[0] = worker.link, 0
        stop = worker.run(2048)
        if stop != wait:
            raise ValueError(f"B worker did not return to wait: {stop}")
        workers.append({"bank": bank, "selector": selector, "subselector": subselector,
                        "gate_word": gate, "scheduled_calls": scheduled, "after": stop})
    return workers


def probe(cp, fdn, *, dictionaries=None, window=SQUARE_WINDOW, branch=0, output_delta=0, exec_id=None,
          trace_heads=False, boot_control=False, trace_later_stages=False):
    """Run the B controller with original CP parameters, synthetic wake and branch inputs."""
    if branch not in (0, 1) or output_delta not in (0, 0x20):
        raise ValueError("unsupported B control fixture")
    if trace_heads and (not dictionaries or branch != 0):
        raise ValueError("tracing head jobs requires the dictionaries and the zero branch")
    if boot_control and (trace_heads or not dictionaries or branch != 0):
        raise ValueError("boot control requires dictionaries, zero branch, and no head bypass")
    if trace_later_stages and not boot_control:
        raise ValueError("later B stages require the boot-control fixture")
    request = cp_probe(cp, manager_id=3, search_type=0, usecase=1, window=window,
                       side_dictionary_offset=0x100000)
    parameter_address = int(request["sammu_parameter_address"], 16)
    parameter_mapping = 0xa0000400
    params = bytearray.fromhex(request["parameter_block_hex"])
    if output_delta:
        params[0xc:0x10] = pack("<I", word(params, 0xc) + output_delta)
    if exec_id is not None:
        params[0x24:0x28] = pack("<I", exec_id)
    out = OUTPUT_PAGE + 0x40 + output_delta
    machine = build_machine(fdn)
    machine.capture_dma = trace_heads or boot_control
    table = BLOCK + word(fdn, BLOCK + 0xc)
    # Express the B catalog location with the same synthetic A load base used
    # by trace_linked_b, not as a live address from a camera.
    catalog_va = FIXTURE_BASE + table - 0x100
    machine.regions += ((0x49e00, bytes(0x160)), (parameter_mapping, params),
                        (catalog_va, fdn[table:]))
    if dictionaries is not None:
        if tuple(sha256(d).hexdigest() for d in dictionaries) != DICTIONARY_HASHES:
            raise ValueError("expected both original FDN dictionaries")
        machine.regions += tuple(zip((0xa0040000, 0xa0080000), dictionaries))
        machine.dictionary_ranges = tuple((name, base, base + len(data)) for name, base, data in
                                          zip(("front", "side"), (0xa0040000, 0xa0080000), dictionaries))
    machine.fixture_reads = {0x481cc: deque((branch,)), B_CATALOG_REG: deque((catalog_va,) * 64)}
    machine.pc, machine.reg[4] = ENTRY, parameter_address
    boot_fixture = seed_boot_control(machine, cp, fdn, dictionaries, window) if boot_control else None
    dictionary_bindings = list(boot_fixture["a_dictionary_transfers"]) if trace_later_stages else []
    packed_outer_calls = []
    packed_plane_stage = None
    prehead_outer_calls = []
    waits, calls, programs = [], [], []
    score_head, writer_input = None, None
    landmark_batches = []
    worker_calls, dictionary_transfers = [], []
    prehead_batches, known_dictionary_regions = [], []
    outer_head_call = None
    for _ in range(192 if trace_later_stages else 80):
        i = len(machine.mmio_writes)
        submitted = len(machine.dma_submissions)
        stop = machine.run(8192, stop_at=WRITER)
        for event in machine.mmio_writes[i:]:
            if event["address"] != DMA_POINTER:
                continue
            address = event["value"]
            src, dst = (machine.read(address + n, 4) for n in (0, 4))
            if 0x04000000 <= dst < 0x04006000 and dst % 0x1000 == 0:
                w2, w3 = (machine.read(address + n, 4) for n in (8, 12))
                programs.append({"member_offset": src - catalog_va + table,
                                 "words": [src, dst, w2, w3], "wait_pc": stop["pc"]})
        for request in machine.dma_submissions[submitted:]:
            transfers = read_dictionary_transfers(request["rows"], dictionaries)
            if (trace_heads or trace_later_stages) and request["submit_pc"] == COMPACT_WEIGHTS_SUBMIT:
                dictionary_transfers.extend(transfers)
            if boot_control:
                if request["submit_pc"] == B_PADDING_SUBMIT and not calls:
                    boot_fixture["padding_dma_bytes"] = materialize_boot_padding(machine, request, stop)
                if transfers and (trace_later_stages or stop["reason"] == "wfi"):
                    map_dictionary_requests(machine, transfers, dictionaries)
                if trace_later_stages:
                    dictionary_bindings.extend(transfers)
                continue
            for transfer in transfers:
                data = dictionaries[transfer["dictionary"] == "side"]
                start = transfer["offset"]
                known_dictionary_regions.insert(0, (transfer["destination"], data[start:start + transfer["size"]]))
            if request["submit_pc"] in PREHEAD_SUBMITS:
                if stop != {"pc": 0x50800cfc, "reason": "wfi"}:
                    raise ValueError(f"unexpected wait after pre-head submission: {stop}")
                prehead_batches.append({"submit_pc": request["submit_pc"],
                                        "workers": run_prehead_workers(fdn, machine.scratch,
                                                                       tuple(known_dictionary_regions))})
        if stop["reason"] == "wfi":
            waits.append(stop["pc"])
            if trace_later_stages and stop["pc"] == 0x50800cfc and calls:
                if calls[-1]["r4"] == PACKED_RECORD and len(packed_outer_calls) == 12:
                    packed_plane_stage = {
                        "outer_calls": packed_outer_calls,
                        "workers": run_scheduled_workers(fdn, machine, dictionary_bindings,
                                                         expected_selector=1, expected_gate=12,
                                                         coefficient_bytes=PACKED_COEFFICIENT_BYTES)}
                elif calls[-1]["r4"] == PREHEAD_RECORD:
                    prehead_batches.append({
                        "outer_calls": prehead_outer_calls[-6:],
                        "workers": run_scheduled_workers(fdn, machine, dictionary_bindings,
                                                         expected_selector=3, expected_gate=6,
                                                         coefficient_bytes=PREHEAD_COEFFICIENT_BYTES)})
            if stop["pc"] == 0x508009b8:
                fields = ("r5_source", "r6_source", "bias_word", "temporary_result")
                landmark_batches.append([dict(zip(fields, (machine.read(addr + 4 * j, 4)
                                                            for j in range(4))))
                                         for addr in (0x178a8, 0x17960, 0x17a18)])
                if trace_heads or trace_later_stages:
                    worker_calls.extend(run_head_workers(fdn, machine.scratch))
            machine.pc += 2
        elif stop == B_COMPUTE:
            calls.append({"r4": machine.reg[4], "return_pc": machine.link,
                          "requests_before_call": len(programs)})
            if boot_control and "first_compute" not in boot_fixture:
                boot_fixture["first_compute"] = {
                    "registers_r4_to_r7": machine.reg[4:8],
                    "record_member_offset": BLOCK + word(fdn, BLOCK + 0x30) + machine.reg[4] - ENTRY,
                    "return_pc": machine.link,
                    "runtime_bias_reads": machine.boot_bias_reads,
                }
                boot_fixture["first_compute_buffer_coverage"] = boot_padding_coverage(
                    machine, boot_fixture.pop("padding_dma_bytes"))
                if not trace_later_stages:
                    break
            if trace_later_stages and machine.reg[4] == PREHEAD_RECORD:
                prehead_outer_calls.append(scheduled_call(machine, dictionary_bindings,
                                                          PREHEAD_COEFFICIENT_BYTES))
            if (trace_later_stages and machine.reg[4] == PACKED_RECORD and
                    PACKED_DESTINATION[0] <= machine.reg[7] < PACKED_DESTINATION[1]):
                packed_outer_calls.append(scheduled_call(machine, dictionary_bindings,
                                                         PACKED_COEFFICIENT_BYTES))
            if machine.reg[4] == COMPACT_RECORD:
                score_head = dict(zip(("r5_source", "r6_source", "temporary_result"), machine.reg[5:8]))
                if trace_heads or trace_later_stages:
                    outer_head_call = describe_worker_call(machine, stop, ENTRY,
                                                           BLOCK + word(fdn, BLOCK + 0x30))
                    config = {w["address"]: w["value"] for w in machine.mmio_writes}
                    outer_head_call.update(count_word=config[0x40000], bias_word=config[0x40004])
            # Independent post-call control slice, without supplying any data
            # that would have been produced by the unseen program.
            machine.pc, machine.reg[0] = machine.link, 0
        elif stop == {"pc": WRITER, "reason": "slice boundary"}:
            writer_input = machine.reg[5]
            if trace_later_stages:
                # Without seven real values there is no valid result record.
                break
            end = machine.link
            if machine.run(1024, stop_at=end) != {"pc": end, "reason": "slice boundary"}:
                raise ValueError("B result writer did not return to the original controller")
        else:
            break
    else:
        raise ValueError("did not reach a new control boundary")
    output = [e for e in machine.scratch_trace if out <= e["address"] < out + RESULT_BYTES]
    result = {"fixture_branch": branch, "fixture_output": out, "cp_parameters_hex": params.hex(" "),
              "branch_reads": [event for event in machine.fixture_trace if event["address"] == 0x481cc],
              "waits": waits, "program_requests": programs, "isolated_calls": calls,
              "score_head_arguments": score_head, "landmark_head_mailboxes": landmark_batches,
              "writer_temporary_pointer": writer_input, "dictionary_reads": machine.dictionary_reads,
              "writes": output, "stop": stop}
    if trace_heads or trace_later_stages:
        result["head_fixture"] = {"outer_call": outer_head_call, "worker_calls": worker_calls,
                                  "dictionary_transfers": dictionary_transfers,
                                  "dma_submissions": machine.dma_submissions,
                                  "prehead_batches": prehead_batches}
    if trace_later_stages:
        result["head_fixture"]["packed_plane_stage"] = packed_plane_stage
        result["head_fixture"]["outer_40218_writes"] = [
            w for w in machine.mmio_writes if w["address"] == 0x40218]
    if boot_control:
        boot_fixture["b_submissions"] = machine.dma_submissions
        result["boot_control_fixture"] = boot_fixture
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("fdn", type=Path)
    cli.add_argument("--dict-dir", type=Path)
    cli.add_argument("--branch", type=int, choices=(0, 1), default=0,
                     help="Fixture for 0x481cc; 1 selects the zero-record path, not a negative inference")
    cli.add_argument("--trace-heads", action="store_true")
    cli.add_argument("--boot-control", action="store_true",
                     help="Map original boot/hand-off data and stop before the first B compute")
    cli.add_argument("--trace-later-stages", action="store_true",
                     help="Trace original later control slices and stop before the result writer")
    args = cli.parse_args()
    dictionaries = None if not args.dict_dir else [
        (args.dict_dir / f"fdn_{name}_dict.bin").read_bytes() for name in ("front", "side")]
    print(json.dumps(probe(args.cp.read_bytes(), args.fdn.read_bytes(),
                           dictionaries=dictionaries, branch=args.branch, trace_heads=args.trace_heads,
                           boot_control=args.boot_control, trace_later_stages=args.trace_later_stages), indent=2))
