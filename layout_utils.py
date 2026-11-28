"""
layout_utils.py - helpers for the "Find a slot" demo layout.

IMPORTANT: the data only contains how many lots are free in a carpark, never which bay.
So bay positions and which bays are taken are SIMULATED. The simulation is made consistent with the
real/forecast COUNT: each bay gets a fixed random fill-order, and the first n bays in that order are taken.
"""
import numpy as np

W, H = 100.0, 44.0
YC = H / 2
# zone letter, share of bays. A top row, B/C centre island (two rows back to back), D bottom row, E left, F right
ZONES = [("A", 0.22), ("B", 0.20), ("C", 0.20), ("D", 0.22), ("E", 0.08), ("F", 0.08)]
_GEO = {
    "A": lambda k: (np.linspace(10, 90, k), np.full(k, H - 4.0)),
    "D": lambda k: (np.linspace(10, 90, k), np.full(k, 4.0)),
    "B": lambda k: (np.linspace(24, 76, k), np.full(k, YC + 3.0)),
    "C": lambda k: (np.linspace(24, 76, k), np.full(k, YC - 3.0)),
    "E": lambda k: (np.full(k, 4.0), np.linspace(10, H - 10, k)),
    "F": lambda k: (np.full(k, 96.0), np.linspace(10, H - 10, k)),
}
# grey blocks: (x0, y0, x1, y1, text)
BLOCKS = [(0, H - 9, 8, H, "Stairs"), (92, H - 9, 100, H, ""), (0, 0, 8, 9, ""), (92, 0, 100, 9, "Lift"),
          (20, YC - 8, 23, YC + 8, ""), (77, YC - 8, 80, YC + 8, "")]


def build_layout(n):
    """Return labels, xs, ys (numpy arrays of length n) for a schematic carpark with exactly n bays."""
    counts = [int(n * share) for _, share in ZONES]
    order, i = [0, 3, 1, 2, 4, 5], 0
    while sum(counts) < n:
        counts[order[i % 6]] += 1
        i += 1
    labels, xs, ys = [], [], []
    for (zone, _), k in zip(ZONES, counts):
        x, y = _GEO[zone](k)
        labels += [f"{zone}{j + 1:02d}" for j in range(k)]
        xs.append(x)
        ys.append(y)
    return np.array(labels), np.concatenate(xs), np.concatenate(ys)


def fill_rank(n, key):
    """Fixed random fill-order for n bays (same carpark -> same order every time)."""
    seed = sum((i + 1) * ord(c) for i, c in enumerate(str(key))) + 12345
    u = np.random.default_rng(seed).random(n)
    return np.argsort(np.argsort(u))


def taken_mask(rank, n_taken):
    """Bays whose fill-order position is below n_taken are taken. Smaller n_taken is always a subset of larger."""
    return rank < int(n_taken)


def slot_from_point(x, y, xs, ys, tol=1.5):
    """Index of the bay nearest to a clicked point, or None if nothing is within tol (layout units)."""
    d = np.hypot(np.asarray(xs) - x, np.asarray(ys) - y)
    j = int(np.argmin(d))
    return j if d[j] <= tol else None
