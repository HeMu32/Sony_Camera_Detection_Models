"""Bounded host replay of local-to-local requests corroborated by SA_FILL."""


def copy_local_rows(rows, memory, *, source_span, destination_span):
    """Return only destination bytes backed by an explicit, initialized source."""
    updates = {}
    src_lo, src_hi = source_span
    dst_lo, dst_hi = destination_span
    for dst, tagged_src, word, other in rows:
        if tagged_src >> 30 != 3 or word != other or word & 0xffff:
            raise ValueError("expected an unstrided tagged local copy")
        src = tagged_src & 0x3fffffff
        size = (word >> 16) + 1
        if not (src_lo <= src <= src_hi - size and dst_lo <= dst <= dst_hi - size):
            raise ValueError("local copy falls outside its proven source or destination span")
        if src < dst + size and dst < src + size:
            raise ValueError("overlapping local copy is not modeled")
        try:
            data = bytes(memory[addr] for addr in range(src, src + size))
        except KeyError as exc:
            raise ValueError(f"local copy has unknown source byte {exc.args[0]:#x}") from exc
        if any(addr in updates for addr in range(dst, dst + size)):
            raise ValueError("overlapping local copy destinations are not modeled")
        updates.update(enumerate(data, dst))
    return updates
