"""Compare SA SP bundle members by name and SHA-256 across models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parsers.sony_sa_post import members


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bundles", nargs="+", type=Path)
    args = cli.parse_args()

    maps = []
    for path in args.bundles:
        records = members(path.read_bytes())
        maps.append((path, {(r.name, r.size): r.sha256 for r in records.values()}))
    all_keys = sorted({key for _, mapping in maps for key in mapping})
    print("member\tsize\t" + "\t".join(path.name for path, _ in maps))
    for key in all_keys:
        name, size = key
        values = [mapping.get(key, "MISSING") for _, mapping in maps]
        print(f"{name}\t{size:#x}\t" + "\t".join(value[:16] for value in values))


if __name__ == "__main__":
    main()
