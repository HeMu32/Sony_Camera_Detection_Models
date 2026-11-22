"""Run original SA boot slices without emulating hardware or detector execution."""

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
from struct import pack, unpack

from probe_sa_fdn_dispatch import MEMBER_SHA, check, executable_banks, word
from run_sa_fdn_dispatch import BLOCKS, DispatchMachine


# A test load address, not a claim about the CP loader's actual relocation.
FIXTURE_BASE = 0x50010000
FIXTURE_READS = {
    0x48004: (0x11223300,), 0x48024: (0x22334400,),
    0x481c4: (0, 0), 0x48020: (0x33440000, 0x33440000),
    0x48000: (0x44550000,), 0x481c0: (0,),
    0x49d24: (0x1000,), 0x49d2c: (0x2000,),
}
STARTUP_OFFSET, STARTUP_SIZE = 0xa09300, 0x7a0
STARTUP_SHA = "35f5618a615aae41b322681a372b28290d3a6d7a32f16841a7366ec4cf95d428"
STARTUP_ARGUMENTS = {0x481d0: 0x40ff, 0x481d4: 0x45001000, 0x481d8: 0xffffffff}
DMA_POINTER = 0x48234
DMA_SUBMIT = 0x48230


class BootMachine(DispatchMachine):
    def __init__(self, image, *, scratch_base=0, scratch_size=0x400):
        end = FIXTURE_BASE + len(image)
        super().__init__({"stage_dsp_address": hex(FIXTURE_BASE),
                          "first_transfer_destination": hex(FIXTURE_BASE),
                          "stack": hex(end), "mailbox_selector": "0x17800"}, image)
        self.scratch = {}
        self.scratch_base = scratch_base
        self.scratch_end = scratch_base + scratch_size
        self.extra_scratch = ()
        self.fixture_reads = {}
        self.fixture_trace = []
        self.scratch_trace = []
        self.scratch_reads = []
        self.extra_registers = ()

    def in_scratch(self, address, size):
        return (self.scratch_base <= address <= self.scratch_end - size or
                any(lo <= address <= hi - size for lo, hi in self.extra_scratch))

    def read(self, address, size, *, fetch=False, sign=False):
        address &= 0xffffffff
        if not fetch and self.in_scratch(address, size):
            if all(i in self.scratch for i in range(address, address + size)):
                value = int.from_bytes(bytes(self.scratch[i] for i in range(address, address + size)),
                                       "little", signed=sign)
                self.scratch_reads.append(dict(pc=self.pc, address=address, size=size, value=value))
                return value
        if not fetch and not sign and size == 4 and self.fixture_reads.get(address):
            value = self.fixture_reads[address].popleft()
            self.fixture_trace.append(dict(pc=self.pc, address=address, value=value))
            return value
        return super().read(address, size, fetch=fetch, sign=sign)

    def write(self, address, size, value):
        if self.in_scratch(address, size):
            raw = (value & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
            self.scratch.update(enumerate(raw, address))
            self.scratch_trace.append(dict(pc=self.pc, address=address, size=size, value=value))
        elif size == 4 and address in self.extra_registers:
            self.mmio_writes.append(dict(pc=self.pc, address=address, size=size, value=value))
        else:
            super().write(address, size, value)


class DmaTraceMachine(BootMachine):
    """Snapshot original descriptor bytes at submission, not at pointer setup."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_dma = False
        self.dma_pointer = None
        self.dma_submissions = []
        self.dma_submit_pcs = ()

    def write(self, address, size, value):
        super().write(address, size, value)
        if not self.capture_dma or size != 4:
            return
        if address == DMA_POINTER:
            self.dma_pointer = (self.pc, value)
        elif (address == DMA_SUBMIT and value == 1 and
              (not self.dma_submit_pcs or self.pc in self.dma_submit_pcs)):
            if self.dma_pointer is None:
                raise ValueError("DMA submission without a descriptor pointer")
            pointer_pc, pointer = self.dma_pointer
            rows = self.descriptor_rows(pointer)
            self.dma_submissions.append({"pointer_pc": pointer_pc, "pointer": pointer,
                                         "submit_pc": self.pc, "rows": rows})

    def descriptor_rows(self, address):
        rows = []
        for offset in range(0, 0x800, 16):
            src, dst = (self.read(address + offset + i, 4) for i in (0, 4))
            if (src, dst) == (1, 0x40000000):
                return rows
            a, b = (self.read(address + offset + i, 4) for i in (8, 12))
            rows.append([src, dst, a, b])
        raise ValueError("unterminated descriptor list")


def fixture_wakes(machine, count):
    waits = []
    for _ in range(count):
        stop = machine.run(1024)
        if stop["reason"] != "wfi":
            raise ValueError(f"expected startup to wait: {stop}")
        waits.append(stop)
        machine.pc += 2  # Explicit synthetic wake; does not transfer memory.
    return waits


def startup_fdn_handoff(machine, fdn, scratch):
    if sha256(fdn).hexdigest() != MEMBER_SHA:
        raise ValueError("expected pinned A7 IV FDN member for startup handoff")
    trace_start = len(machine.fixture_trace)
    machine.fixture_reads.update({a: deque((v,)) for a, v in STARTUP_ARGUMENTS.items()})
    machine.fixture_reads[0x48004] = deque((0x11223303,) * 3)
    machine.fixture_reads[0x48024] = deque((0x22334403,) * 6)
    machine.extra_registers = (0x50860, 0x50864, 0x50868)
    first = machine.run(1024)
    writes = {e["address"]: e["value"] for e in machine.mmio_writes}
    dst, end, source = (writes[a] for a in (0x49d24, 0x49d28, 0x49d2c))
    if (first["reason"], first.get("address"), end - dst + 1, source, writes[0x49d20]) != (
            "unmodeled read", dst + 0x18, 0x100, STARTUP_ARGUMENTS[0x481d4], 1):
        raise ValueError(f"unexpected stop before FDN header fixture: {first}")
    header_request = {"source_register_value": source, "destination": dst, "size": end - dst + 1}
    argument_reads = machine.fixture_trace[trace_start:]

    # Bind the synthetic source address to the pinned FDN block, not to live MMIO.
    block = BLOCKS["A"][0]
    machine.regions += ((dst, fdn[block:block + header_request["size"]]),)
    fdn_entry_stop = machine.run(1024)
    end = next(e["value"] for e in reversed(machine.mmio_writes) if e["address"] == 0x49d28)
    amount = end - dst + 1
    if not header_request["size"] <= amount <= len(fdn) - block:
        raise ValueError(f"FDN mapping outside pinned block: {amount:#x}")
    machine.regions = machine.regions[:-1] + ((dst, fdn[block:block + amount]),)
    machine.scratch_base, machine.scratch_end = 0, 0x400
    machine.fixture_reads = {k: deque(v) for k, v in FIXTURE_READS.items()}
    boot_writes = len(machine.mmio_writes)
    fdn_waits = fixture_wakes(machine, 2)

    mapping = check(fdn, *BLOCKS["A"])
    boot_words = {e["address"]: e["value"] for e in machine.mmio_writes[boot_writes:]}
    ext, last = (boot_words[a] for a in (0x49d04, 0x49d08))
    ext_size = last - ext + 1
    ext_src = block + word(fdn, block + 0x30)
    src, first_bank, stack = (int(mapping[k], 16) for k in (
        "stage_file_offset", "first_transfer_destination", "stack"))
    if (ext != machine.reg[1] or ext != int(mapping["boot_external_entry"], 16)
            or ext_size <= 0 or ext_src + ext_size != src):
        raise ValueError("FDN external image and local bank must be mapped separately")
    machine.regions += ((ext, fdn[ext_src:src]),)
    stack_top = word(fdn, ext_src + 0x10)
    machine.stack_low, machine.stack_high = stack_top - 0x1000, stack_top
    machine.stack = bytearray(0x1000)
    machine.fixture_reads = {}
    external_stop = machine.run(1024)

    # Re-enter the first local bank independently; do not skip the external read
    # and then claim that it dispatched this bank.
    bank_size = mapping["first_transfer_bytes"]
    machine.pc = first_bank
    machine.regions += ((first_bank, fdn[src:src + bank_size]),)
    machine.stack_low, machine.stack_high = first_bank + bank_size, stack
    machine.stack = bytearray(machine.stack_high - machine.stack_low)
    machine.scratch_base, machine.scratch_end = scratch, scratch + 0x18
    machine.fixture_reads = {}
    dispatch_wait = machine.run(1024)
    if dispatch_wait["reason"] != "wfi":
        raise ValueError(f"expected FDN to wait for an actual request: {dispatch_wait}")

    # A wake with no mailbox refill must not turn stale boot data into a job.
    machine.pc += 2
    read_start, write_start = len(machine.scratch_reads), len(machine.mmio_writes)
    stale_stop = machine.run(1024)
    return {
        "fixture_startup_register_reads": argument_reads,
        "header_request": header_request,
        "before_header_copy": first,
        "fdn_entry_stop": fdn_entry_stop,
        "fdn_mapped_bytes": amount,
        "fdn_boot_waits": fdn_waits,
        "fdn_external_mapping": {"file_offset": ext_src, "destination": ext, "size": ext_size},
        "fdn_external_stop": external_stop,
        "isolated_bank_wait": dispatch_wait,
        "stale_mailbox_reads": machine.scratch_reads[read_start:],
        "after_stale_wake": stale_stop,
        "writes_after_stale_wake": machine.mmio_writes[write_start:],
    }


def fdn_mailbox_producer(fdn, external):
    ext, start, size = (external[k] for k in ("destination", "file_offset", "size"))
    machine = BootMachine(fdn[start:start + size], scratch_base=0x17800, scratch_size=0x400)
    machine.pc = 0x50000ca4
    machine.regions = ((ext, fdn[start:start + size]),
                       (0x7000, pack("<3I", 0x40110000, 0x40220000, 0x40330000)))
    machine.stack_low, machine.stack_high = 0x7100, 0x8000
    machine.stack = bytearray(0xf00)
    machine.reg[4:8] = [0x7000, 480, 32, 0x40440000]
    machine.reg[15] = 0x7ff0
    machine.write(0x7ff0, 4, 0x40550000)
    machine.write(0x7ff4, 4, 0x40660000)
    producer_stop = machine.run(1024, stop_at=0x50000d44)
    if producer_stop["reason"] != "slice boundary":
        raise ValueError(f"FDN writer did not reach the next call: {producer_stop}")
    return {"producer_stop": producer_stop, "writes": machine.scratch_trace,
            "banks": consume_a_mailboxes(fdn, machine.scratch, 2)}


def consume_a_mailboxes(fdn, scratch, count, *, data_regions=(), trace_compute=False, call_count=1):
    records = []
    for bank, region in enumerate(executable_banks(fdn, BLOCKS["A"][0])):
        if bank >= count:
            break
        mailbox = 0x17800 + bank * 0xe4
        # Use the producer's bytes. Missing fields stay unmapped, not zero-filled.
        local, src, length = (region[k] for k in ("destination", "file_offset", "length"))
        worker = BootMachine(fdn[src:src + length], scratch_base=mailbox, scratch_size=0xe4)
        worker.pc = local
        worker.regions = ((local, fdn[src:src + length]), *data_regions)
        worker.stack_low, worker.stack_high = local + length, region["stack"]
        worker.stack = bytearray(worker.stack_high - worker.stack_low)
        worker.scratch = {a: v for a, v in scratch.items() if mailbox <= a < mailbox + 0xe4}
        wait = fixture_wakes(worker, 1)[0]
        stop = worker.run(1024)
        record = {"bank": bank, "mailbox": mailbox, "wait": wait, "stop": stop,
                  "reads": worker.scratch_reads, "wrapper_r4": worker.reg[4]}
        if trace_compute:
            record.update(registers_r4_to_r7=worker.reg[4:8],
                          aperture_writes=[w for w in worker.mmio_writes if 0x40000 <= w["address"] < 0x40400])
        if call_count > 1:
            calls = []
            for _ in range(call_count):
                if stop.get("opcode") != 0xa:
                    raise ValueError(f"worker did not reach a scheduled compute boundary: {stop}")
                cfg = {w["address"]: w["value"] for w in worker.mmio_writes}
                calls.append({"stop": stop, "return_pc": worker.link,
                              "registers_r4_to_r7": worker.reg[4:8], "bias_word": cfg[0x40320],
                              "aperture_40330": cfg[0x40330]})
                if len(calls) < call_count:
                    # Start a separate return slice; supply no computed buffer.
                    worker.pc, worker.reg[0] = worker.link, 0
                    stop = worker.run(1024)
            record["scheduled_calls"] = calls
            worker.pc, worker.reg[0] = worker.link, 0
            end = worker.run(1024)
            if end != wait:
                raise ValueError(f"worker did not return to its wait after {call_count} calls: {end}")
            record["after_scheduled_calls"] = end
        records.append(record)
    return records


def probe_startup_control(cp, fdn=None):
    image = cp[STARTUP_OFFSET:STARTUP_OFFSET + STARTUP_SIZE]
    if sha256(image).hexdigest() != STARTUP_SHA:
        raise ValueError("expected the pinned SA_Z_STA boot bytes in A7 IV 6.02 CP")
    scratch = word(image, 0x28)
    machine = BootMachine(image, scratch_base=scratch, scratch_size=0x18)
    first = machine.run(1024)
    machine.fixture_reads = {k: deque(v) for k, v in FIXTURE_READS.items()}
    waits = fixture_wakes(machine, 2)
    raw = bytes(machine.scratch[scratch + i] for i in range(0x18))
    registers = {e["address"]: e["value"] for e in machine.mmio_writes}
    # Copy the stage-2 code as a fixture, using the emitted destination and length.
    dst = registers[0x49d04]
    size = registers[0x49d08] - dst + 1
    start = word(image, 0x30)
    if len(image[start:start + size]) != size:
        raise ValueError("computed startup copy falls outside the pinned block")
    machine.regions += ((dst, image[start:start + size]),)
    machine.stack_low, machine.stack_high = 0xf000, 0x10000
    machine.stack = bytearray(machine.stack_high - machine.stack_low)
    stage2_stop = machine.run(1024)
    result = {
        "first_stop": first, "fixture_waits": waits, "scratch_address": scratch,
        "descriptor_words": list(unpack("<6I", raw)),
        "descriptor_pointer_writes": [e["value"] for e in machine.mmio_writes
                                      if e["address"] == 0x48234],
        "fixture_stage2": {"src_block_offset": start, "destination": dst, "size": size},
        "stage2_stop": stage2_stop,
    }
    if fdn is not None:
        handoff = startup_fdn_handoff(machine, fdn, scratch)
        result["fdn_handoff_fixture"] = handoff
        result["fdn_mailbox_producer_fixture"] = fdn_mailbox_producer(
            fdn, handoff["fdn_external_mapping"])
    return result


def probe(blob, block, *, fixture=None):
    if sha256(blob).hexdigest() != MEMBER_SHA:
        raise ValueError("expected pinned A7 IV FDN member")
    if fixture not in (None, "header-only", "with-external-table"):
        raise ValueError("unknown register fixture")
    offset = BLOCKS[block][0]
    image = blob[offset:offset + word(blob, offset + 0x10)]
    machine = BootMachine(image)
    first = machine.run(1024)
    result = {"block": block, "fixture_base": hex(FIXTURE_BASE), "first_stop": first,
              "fixture_reads": machine.fixture_trace, "scratch_writes": machine.scratch_trace,
              "mmio_writes": machine.mmio_writes}
    if fixture is None:
        return result
    if first != {"pc": FIXTURE_BASE + 0x94, "reason": "unmodeled read",
                 "address": 0x48004, "size": 4}:
        raise ValueError(f"unexpected no-fixture stop: {first}")
    machine.fixture_reads = {k: deque(v) for k, v in FIXTURE_READS.items()}
    if fixture == "with-external-table":
        machine.fixture_reads[0x481c4] = deque((1, 1))
    result["fixture"] = fixture
    stops = []
    for _ in range(3):
        stop = machine.run(1024)
        stops.append(stop)
        if stop["reason"] != "wfi":
            break
        machine.pc += 2  # Synthetic wake only; no simulated DMA completion or RAM copy.
    result["fixture_stops"] = stops
    raw = bytes(machine.scratch[i] for i in range(len(machine.scratch)))
    result["bootstrap_descriptor_words"] = [hex(w) for w in unpack(f"<{len(raw) // 4}I", raw)]
    result["instructions_seen"] = len(machine.pcs)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("member", type=Path)
    cli.add_argument("--block", choices=BLOCKS, default="A")
    cli.add_argument("--fixture", choices=("header-only", "with-external-table"),
                     help="Supply explicit MMIO inputs and at most two synthetic wakes")
    cli.add_argument("--startup-control", action="store_true",
                     help="Trace embedded SA_Z_STA using the CP image instead of an FDN member")
    cli.add_argument("--fdn-member", type=Path,
                     help="With --startup-control, stage this separate pinned FDN member")
    args = cli.parse_args()
    blob = args.member.read_bytes()
    if args.startup_control and args.fixture is not None:
        cli.error("--startup-control uses its own fixed fixture")
    if args.fdn_member is not None and not args.startup_control:
        cli.error("--fdn-member requires --startup-control")
    fdn = args.fdn_member.read_bytes() if args.fdn_member is not None else None
    result = probe_startup_control(blob, fdn) if args.startup_control else probe(
        blob, args.block, fixture=args.fixture)
    print(json.dumps(result, indent=2))
