"""Disassemble a selected raw program range with a local compiled SA-DSP SLEIGH spec."""

import argparse
from pathlib import Path
from xml.etree import ElementTree

from pypcode import ArchLanguage, Context


def number(value):
    return int(value, 0)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("languages", type=Path, help="Directory with dsp.ldefs and dsp.sla")
    cli.add_argument("member", type=Path, help="Exact SP member from sony_sa_post.py")
    cli.add_argument("--offset", type=number, required=True, help="Member-relative file offset")
    cli.add_argument("--address", type=number, required=True, help="DSP address for that offset")
    cli.add_argument("--count", type=number, default=32)
    args = cli.parse_args()
    data = args.member.read_bytes()
    if not 0 <= args.offset < len(data) or args.offset % 2 or args.address < 0 or args.address % 2:
        cli.error("expected valid halfword-aligned file offset and DSP address")
    if args.count < 1:
        cli.error("--count must be positive")
    root = ElementTree.parse(args.languages / "dsp.ldefs").getroot()
    context = Context(ArchLanguage(str(args.languages), root.find("language")))
    insns = context.disassemble(data[args.offset:], args.address,
                                max_instructions=args.count).instructions
    for insn in insns:
        offset = insn.addr.offset - args.address + args.offset
        raw = data[offset:offset + insn.length].hex(" ")
        print(f"{offset:08x} {insn.addr.offset:08x}  {raw:11} {insn.mnem:8} {insn.body}")
    if len(insns) != args.count:
        raise SystemExit(f"decoded {len(insns)} of {args.count} requested instructions")


if __name__ == "__main__":
    main()
