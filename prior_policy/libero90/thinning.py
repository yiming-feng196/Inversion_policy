"""Per-episode temporal thinning; full chunks and episode splits stay intact."""
from collections import defaultdict


def select_rows(rows, stride=8):
    if stride < 1:
        raise ValueError('Stride must be positive')
    groups = defaultdict(list)
    for row in rows:
        groups[row[0]].append(row)
    result = []
    for name in groups:
        ordered = sorted(groups[name], key=lambda row: row[1])
        if len({row[1] for row in ordered}) != len(ordered):
            raise ValueError(f'Duplicate window in {name}')
        chosen = ordered[::stride]
        if chosen and chosen[-1][1] != ordered[-1][1]:
            chosen.append(ordered[-1])
        result.extend(chosen)
    return result
