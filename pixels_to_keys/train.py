"""Behavior cloning: train the policy to predict the human's next key press.

    python -m pixels_to_keys.train --data data/dino --out checkpoints/dino.pt
    python -m pixels_to_keys.train --data data/flappy --init checkpoints/dino.pt --out checkpoints/flappy_ft.pt
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import STACK, PolicyNet


def make_pairs(path):
    """Input: frames t-3..t. Label: key pressed right after frame t (reaction to what was seen)."""
    d = np.load(path)
    frames, actions = d["frames"], d["actions"]
    xs = [frames[t - STACK + 1:t + 1] for t in range(STACK - 1, len(frames) - 1)]
    ys = [actions[t + 1] for t in range(STACK - 1, len(frames) - 1)]
    return xs, ys


def load_split(files):
    xs, ys = [], []
    for f in files:
        x, y = make_pairs(f)
        xs += x
        ys += y
    X = torch.tensor(np.array(xs), dtype=torch.float32) / 255.0
    Y = torch.tensor(np.array(ys), dtype=torch.long)
    return X, Y


def random_shift(x, max_shift):
    """Shift the whole batch horizontally by up to max_shift pixels, zero-filled."""
    k = int(torch.randint(-max_shift, max_shift + 1, (1,)))
    x = torch.roll(x, k, dims=3)
    if k > 0:
        x[..., :k] = 0
    elif k < 0:
        x[..., k:] = 0
    return x


def evaluate(model, X, Y, device):
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).argmax(1).cpu()
    acc = (pred == Y).float().mean().item()
    jumps = Y == 1
    recall = (pred[jumps] == 1).float().mean().item() if jumps.any() else 0.0
    return acc, recall


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/dino"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/dino.pt"))
    ap.add_argument("--init", type=Path, default=None, help="start from these weights")
    ap.add_argument("--freeze-conv", action="store_true", help="train only the head")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="use only the first training rounds up to this many frames")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--jump-weight", type=float, default=5.0,
                    help="loss weight for the jump class (jumps are ~4%% of frames)")
    ap.add_argument("--shift", type=int, default=0, help="max random horizontal shift (px)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Split by round, not by frame: neighbouring frames are near-duplicates.
    files = sorted(args.data.glob("round_*.npz"))
    random.shuffle(files)
    n_val = max(1, len(files) // 5)
    val_files, train_files = files[:n_val], files[n_val:]
    if args.max_frames:
        # The validation rounds stay fixed; only the training subset shrinks.
        chosen, total = [], 0
        for f in train_files:
            if total >= args.max_frames:
                break
            chosen.append(f)
            total += len(np.load(f)["actions"])
        train_files = chosen
    X_tr, Y_tr = load_split(train_files)
    X_val, Y_val = load_split(val_files)
    print(f"train {len(Y_tr)} ({Y_tr.float().mean():.1%} jump) | "
          f"val {len(Y_val)} ({Y_val.float().mean():.1%} jump)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PolicyNet()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    if args.freeze_conv:
        for p in model.conv.parameters():
            p.requires_grad = False
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.jump_weight], device=device))

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(Y_tr))
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            x, y = X_tr[idx], Y_tr[idx]
            if args.shift:
                x = random_shift(x, args.shift)
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        acc, recall = evaluate(model, X_val, Y_val, device)
        print(f"epoch {epoch:2d}  loss {total / len(Y_tr):.3f}  "
              f"val acc {acc:.1%}  jump recall {recall:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
