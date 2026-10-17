"""Evaluate a policy on fixed seeds in the deterministic environment.

    python -m pixels_to_keys.evaluate --model checkpoints/model.pt --games 50

Writes to --out: results.json (scores), games/ (frames + actions per game, same
format as recordings) and crashes/ (GIF of the last 2 s before each crash).
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from playwright.sync_api import sync_playwright

from .env import DinoEnv
from .model import STACK, DinoNet

CLIP_FRAMES = 40  # 2 s at 20 Hz


def save_clip(frames, actions, path):
    """GIF of what the agent saw; the top bar lights up on frames after a jump press."""
    images = []
    for f, a in zip(frames, actions):
        img = np.zeros((44, 160), dtype=np.uint8)
        img[4:] = f
        if a:
            img[:3] = 255
        images.append(Image.fromarray(img).resize((480, 132), Image.NEAREST))
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)


def play_game(env, model, seed, max_steps):
    frame = env.reset(seed)
    recent = deque([frame], maxlen=STACK)
    frames, actions, speeds = [frame], [0], [env.speed()]
    latency = []
    score = 0
    for _ in range(max_steps):
        jump = False
        if len(recent) == STACK:
            t0 = time.perf_counter()
            x = torch.tensor(np.array(recent), dtype=torch.float32)[None] / 255.0
            with torch.no_grad():
                jump = model(x).argmax(1).item() == 1
            latency.append((time.perf_counter() - t0) * 1000)
        frame, crashed, score = env.step(jump)
        recent.append(frame)
        frames.append(frame)
        actions.append(int(jump))  # actions[t] = pressed after frames[t-1], as in recordings
        speeds.append(env.speed())
        if crashed:
            break
    return score, frames, actions, speeds, latency


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=Path("checkpoints/model.pt"))
    ap.add_argument("--games", type=int, default=50, help="seeds 1..N")
    ap.add_argument("--max-steps", type=int, default=6000, help="6000 steps = 5 min of game time")
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    ap.add_argument("--watch", action="store_true", help="show the browser window")
    args = ap.parse_args()

    model = DinoNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    games_dir, crash_dir = args.out / "games", args.out / "crashes"
    for d in (games_dir, crash_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()

    scores, latency = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.watch)
        env = DinoEnv(browser)
        for seed in range(1, args.games + 1):
            score, frames, actions, speeds, lat = play_game(env, model, seed, args.max_steps)
            np.savez_compressed(games_dir / f"game_{seed:02d}.npz", frames=np.array(frames),
                                actions=np.array(actions), speed=np.array(speeds), score=score)
            save_clip(frames[-CLIP_FRAMES:], actions[-CLIP_FRAMES:],
                      crash_dir / f"game_{seed:02d}_score_{score}.gif")
            scores.append(score)
            latency += lat
            print(f"seed {seed:2d}: score {score}", flush=True)
        browser.close()

    s = np.array(scores)
    print(f"\ngames {len(s)} | median {np.median(s):.0f} | mean {s.mean():.0f} | "
          f"min {s.min()} | max {s.max()} | {np.mean(latency):.2f} ms/decision")
    json.dump({"model": str(args.model), "max_steps": args.max_steps, "scores": scores,
               "ms_per_decision": float(np.mean(latency))},
              open(args.out / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
