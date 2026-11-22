"""Find candidate ARM/Thumb PC-relative references to unique firmware strings."""

import argparse
from pathlib import Path
from struct import unpack_from

from capstone import CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB, Cs
from capstone.arm import ARM_OP_MEM, ARM_OP_REG, ARM_REG_PC


def references(blob, targets, mode):
    dis = Cs(CS_ARCH_ARM, mode)
    dis.detail = True
    dis.skipdata = True
    recent = {}
    for ins in dis.disasm(blob, 0):
        if ins.id == 0:
            recent.clear()
            continue
        ops = ins.operands
        pc = ins.address + (4 if mode == CS_MODE_THUMB else 8)
        if ins.mnemonic.startswith("add") and len(ops) >= 2:
            regs = [op.reg for op in ops if op.type == ARM_OP_REG]
            if len(regs) == len(ops) and ARM_REG_PC in regs[1:]:
                source = next((r for r in regs[1:] if r != ARM_REG_PC), regs[0])
                if source in recent:
                    load, literal, displacement = recent[source]
                    if 0 <= ins.address - load <= 72 and pc + displacement in targets:
                        yield load, ins.address, literal, pc + displacement
        if (ins.mnemonic.startswith("ldr") and len(ops) == 2 and
                ops[0].type == ARM_OP_REG and ops[1].type == ARM_OP_MEM and
                ops[1].mem.base == ARM_REG_PC):
            literal = (pc & ~3) + ops[1].mem.disp
            if 0 <= literal <= len(blob) - 4:
                recent[ops[0].reg] = ins.address, literal, unpack_from("<i", blob, literal)[0]
        else:
            for register in ins.regs_access()[1]:
                recent.pop(register, None)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("binary", type=Path)
    cli.add_argument("strings", nargs="+")
    args = cli.parse_args()
    data = args.binary.read_bytes()
    targets = {}
    for label in args.strings:
        offset = data.find(label.encode("ascii"))
        if offset < 0:
            print(f"Missing: {label}")
            continue
        # A reference can point to the string or to an enclosing diagnostic.
        start = data.rfind(b"\0", 0, offset) + 1
        for address in (start, offset):
            targets[address] = label
        print(f"{label}: start={start:#x} substring={offset:#x}")
    for mode, label in ((CS_MODE_ARM, "ARM"), (CS_MODE_THUMB, "Thumb")):
        for load, add, literal, string in references(data, targets, mode):
            print(f"{label} LOAD {load:#x} ADD {add:#x} LIT {literal:#x}"
                  f" STR {string:#x} ({targets[string]})")


if __name__ == "__main__":
    main()
