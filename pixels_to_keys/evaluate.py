"""Evaluate a policy on fixed seeds in a deterministic environment.

    python -m pixels_to_keys.evaluate --model checkpoints/dino.pt

Writes to --out (default runs/dino): results.json, games/ (frames + actions per
game, same format as recordings) and crashes/ (GIF of the last frames before each crash).
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
from .model import STACK, PolicyNet

MAX_STEPS = 6000  # 5 minutes of game time at 50 ms per step
CLIP_STEPS = 40   # 2 s


def save_clip(frames, actions, path):
    """GIF of what the agent saw; the top bar lights up on frames after a press."""
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
    frames, actions, speeds = [frame], [0], []
    latency = []
    score = 0
    for _ in range(max_steps):
        speeds.append(env.speed())
        press = False
        if len(recent) == STACK:
            t0 = time.perf_counter()
            x = torch.tensor(np.array(recent), dtype=torch.float32)[None] / 255.0
            with torch.no_grad():
                press = model(x).argmax(1).item() == 1
            latency.append((time.perf_counter() - t0) * 1000)
        frame, crashed, score = env.step(press)
        recent.append(frame)
        frames.append(frame)
        actions.append(int(press))  # actions[t] = pressed after frames[t-1], as in recordings
        if crashed:
            break
    speeds.append(env.speed())
    return score, frames, actions, speeds, latency


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=Path("checkpoints/dino.pt"))
    ap.add_argument("--games", type=int, default=50, help="seeds 1..N")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS, help="default: 5 min of game time")
    ap.add_argument("--out", type=Path, default=Path("runs/dino"))
    ap.add_argument("--watch", action="store_true", help="show the browser window")
    args = ap.parse_args()
    model_path, max_steps, out = args.model, args.max_steps, args.out

    model = PolicyNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    games_dir, crash_dir = out / "games", out / "crashes"
    for d in (games_dir, crash_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()

    scores, latency = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.watch)
        env = DinoEnv(browser)
        for seed in range(1, args.games + 1):
            score, frames, actions, speeds, lat = play_game(env, model, seed, max_steps)
            np.savez_compressed(games_dir / f"game_{seed:02d}.npz", frames=np.array(frames),
                                actions=np.array(actions), score=score, speed=np.array(speeds))
            save_clip(frames[-CLIP_STEPS:], actions[-CLIP_STEPS:],
                      crash_dir / f"game_{seed:02d}_score_{score}.gif")
            scores.append(score)
            latency += lat
            print(f"seed {seed:2d}: score {score}", flush=True)
        browser.close()

    s = np.array(scores)
    print(f"\ngames {len(s)} | median {np.median(s):.0f} | mean {s.mean():.0f} | "
          f"min {s.min()} | max {s.max()} | {np.mean(latency):.2f} ms/decision")
    json.dump({"model": str(model_path), "max_steps": max_steps,
               "scores": scores, "ms_per_decision": float(np.mean(latency))},
              open(out / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
