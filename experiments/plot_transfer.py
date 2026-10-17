"""Summarize and plot the Flappy transfer experiment (runs/transfer)."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = {"scratch": "From scratch", "dino_ft": "From Dino, fine-tune all",
           "dino_head": "From Dino, train head only"}
COLORS = {"scratch": "#a3a29c", "dino_ft": "#2a78d6", "dino_head": "#eb6834"}
FPS = 30


def train_minutes(data: Path, max_frames: int) -> float:
    """Minutes of training data actually used (matches train.py's split)."""
    import random
    files = sorted(data.glob("round_*.npz"))
    random.seed(0)
    random.shuffle(files)
    train = files[max(1, len(files) // 5):]
    total = 0
    for f in train:
        if max_frames and total >= max_frames:
            break
        total += len(np.load(f)["actions"])
    return total / FPS / 60


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs/transfer"))
    ap.add_argument("--data", type=Path, default=Path("data/flappy"))
    ap.add_argument("--plot", type=Path, default=Path("runs/transfer/transfer.png"))
    args = ap.parse_args()

    zero = json.load(open(args.runs / "zero_shot" / "results.json"))["scores"]
    print(f"zero-shot Dino policy: median {np.median(zero):.0f}, mean {np.mean(zero):.1f}")

    # results[method][size] = list of per-seed mean scores
    results = defaultdict(lambda: defaultdict(list))
    for d in args.runs.iterdir():
        m = re.fullmatch(r"(scratch|dino_ft|dino_head)_f(\d+)_s(\d+)", d.name)
        if m and (d / "results.json").exists():
            scores = json.load(open(d / "results.json"))["scores"]
            results[m[1]][int(m[2])].append(float(np.mean(scores)))

    sizes = sorted({s for r in results.values() for s in r}, key=lambda s: s or 1e9)
    minutes = {s: train_minutes(args.data, s) for s in sizes}
    print(f"\n{'train data':>12} " + " ".join(f"{METHODS[k]:>28}" for k in METHODS))
    for s in sizes:
        cells = []
        for k in METHODS:
            v = results[k].get(s, [])
            cells.append(f"{np.mean(v):6.1f} ({', '.join(f'{x:.1f}' for x in v)})" if v else "-")
        print(f"{minutes[s]:9.1f} min " + " ".join(f"{c:>28}" for c in cells))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = [minutes[s] for s in sizes]
    for k, label in METHODS.items():
        means = [np.mean(results[k][s]) if results[k].get(s) else np.nan for s in sizes]
        ax.plot(x, means, marker="o", lw=2, ms=7, color=COLORS[k], label=label)
    ax.axhline(np.mean(zero), color="#52514e", ls="--", lw=1)
    ax.text(x[-1], np.mean(zero), " Dino policy, no Flappy data", va="bottom", ha="right",
            fontsize=9, color="#52514e")
    ax.set_xlabel("Minutes of human Flappy Bird play used for training")
    ax.set_ylabel("Mean pipes passed (50 fixed seeds)")
    ax.set_title("Does a Dino policy help on Flappy Bird?")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot)
    print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()
