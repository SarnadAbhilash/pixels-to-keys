"""Dino jump timing: how far away the cactus is when the human vs the agent jumps.

    python -m pixels_to_keys.analyze --human data/dino --agent runs/dino/games --plot runs/jump_timing.png

Distances are measured from pixels only (160x40 frames), so human recordings and
agent rollouts are measured the same way. Larger distance = earlier jump.
"""
import argparse
from pathlib import Path

import numpy as np

DINO_RIGHT = 17                # dino's right edge (column)
OBSTACLE_ROWS = slice(27, 35)  # rows just above the ground line
BRIGHT = 80
SPEED_BINS = [(6, 8, "slow"), (8, 10, "medium"), (10, 13.1, "fast")]


def cactus_distance(frame):
    """Columns between the dino and the nearest ground obstacle, or None."""
    area = frame[OBSTACLE_ROWS, DINO_RIGHT + 1:] > BRIGHT
    cols = np.where(area.any(axis=0))[0]
    if len(cols) == 0:
        return None
    first = cols[0]
    if not area[-1, first:first + 6].any():  # not touching the ground: a bird
        return None
    return int(first)


def speed_from_time(t):
    """Game speed after t steps of 50 ms (starts at 6, +0.001 per 60 Hz frame, max 13)."""
    return min(13.0, 6.0 + 0.06 * t * 0.05)


def jump_distances(folder):
    """(speed, distance) for every jump over a ground obstacle, using the last frame before the press."""
    rows = []
    for f in sorted(Path(folder).glob("*.npz")):
        d = np.load(f)
        frames, actions = d["frames"], d["actions"]
        speeds = d["speed"] if "speed" in d.files else None
        for t in np.where(actions == 1)[0]:
            if t < 1:
                continue
            dist = cactus_distance(frames[t - 1])
            if dist is not None and dist < 100:
                rows.append((speeds[t] if speeds is not None else speed_from_time(t), dist))
    return np.array(rows).reshape(-1, 2)


def medians(rows):
    out = []
    for lo, hi, _ in SPEED_BINS:
        d = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi), 1]
        out.append(float(np.median(d)) if len(d) >= 3 else float("nan"))
    return out


def plot(results, path):
    import matplotlib.pyplot as plt
    x = np.arange(len(SPEED_BINS))
    colors = ["#2a78d6", "#eb6834"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    width = 0.8 / len(results)
    for i, (name, vals) in enumerate(results.items()):
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.95, color=colors[i % 2], label=name)
        for xi, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [b[2] for b in SPEED_BINS])
    ax.set_ylabel("Obstacle distance at jump (px)")
    ax.set_title("Obstacle distance at jump (higher = earlier)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--human", type=Path, default=Path("data/dino"))
    ap.add_argument("--agent", type=Path, default=Path("runs/dino/games"))
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    results = {}
    for name, folder in (("human", args.human), ("agent", args.agent)):
        if not folder.exists():
            continue
        rows = jump_distances(folder)
        results[name] = medians(rows)
        cells = "  ".join(f"{b[2]} {m:5.1f}" for b, m in zip(SPEED_BINS, results[name]))
        print(f"{name:6s} n={len(rows):5d}  median distance (px): {cells}")

    if args.plot and results:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(results, args.plot)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
