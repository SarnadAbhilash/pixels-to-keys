"""Evaluate a policy on fixed seeds in a deterministic environment.

    python -m pixels_to_keys.evaluate --game dino --model checkpoints/dino.pt
    python -m pixels_to_keys.evaluate --game flappy --model checkpoints/flappy_ft.pt

Writes to --out (default runs/<game>): results.json, games/ (frames + actions per
game, same format as recordings) and crashes/ (GIF of the last frames before each crash).
"""
import argparse
import contextlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import STACK, PolicyNet

# 5 minutes of game time: Dino steps are 50 ms, Flappy steps are one 30 fps tick.
DEFAULT_MAX_STEPS = {"dino": 6000, "flappy": 9000}
CLIP_STEPS = {"dino": 40, "flappy": 60}  # 2 s


@contextlib.contextmanager
def make_env(game, watch=False):
    if game == "dino":
        from playwright.sync_api import sync_playwright

        from .envs.dino import DinoEnv
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not watch)
            yield DinoEnv(browser)
            browser.close()
    else:
        from .envs.flappy import FlappyEnv
        yield FlappyEnv(render=watch)


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


def play_game(env, model, seed, max_steps, watch=False):
    frame = env.reset(seed)
    recent = deque([frame], maxlen=STACK)
    frames, actions, speeds = [frame], [0], []
    latency = []
    score = 0
    has_speed = hasattr(env, "speed")
    for _ in range(max_steps):
        if has_speed:
            speeds.append(env.speed())
        press = False
        if len(recent) == STACK:
            t0 = time.perf_counter()
            x = torch.tensor(np.array(recent), dtype=torch.float32)[None] / 255.0
            with torch.no_grad():
                press = model(x).argmax(1).item() == 1
            latency.append((time.perf_counter() - t0) * 1000)
        frame, crashed, score = env.step(press)
        if watch and hasattr(env, "show"):
            env.show()
            env.config.clock.tick(30)
        recent.append(frame)
        frames.append(frame)
        actions.append(int(press))  # actions[t] = pressed after frames[t-1], as in recordings
        if crashed:
            break
    if has_speed:
        speeds.append(env.speed())
    return score, frames, actions, speeds, latency


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", choices=("dino", "flappy"), default="dino")
    ap.add_argument("--model", type=Path, default=None, help="default: checkpoints/<game>.pt")
    ap.add_argument("--games", type=int, default=50, help="seeds 1..N")
    ap.add_argument("--max-steps", type=int, default=None, help="default: 5 min of game time")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/<game>")
    ap.add_argument("--watch", action="store_true", help="show the game window")
    args = ap.parse_args()
    model_path = args.model or Path("checkpoints") / f"{args.game}.pt"
    max_steps = args.max_steps or DEFAULT_MAX_STEPS[args.game]
    out = args.out or Path("runs") / args.game

    model = PolicyNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    games_dir, crash_dir = out / "games", out / "crashes"
    for d in (games_dir, crash_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()

    scores, latency = [], []
    clip = CLIP_STEPS[args.game]
    with make_env(args.game, args.watch) as env:
        for seed in range(1, args.games + 1):
            score, frames, actions, speeds, lat = play_game(env, model, seed, max_steps,
                                                            args.watch)
            extra = {"speed": np.array(speeds)} if speeds else {}
            np.savez_compressed(games_dir / f"game_{seed:02d}.npz", frames=np.array(frames),
                                actions=np.array(actions), score=score, **extra)
            save_clip(frames[-clip:], actions[-clip:],
                      crash_dir / f"game_{seed:02d}_score_{score}.gif")
            scores.append(score)
            latency += lat
            print(f"seed {seed:2d}: score {score}", flush=True)

    s = np.array(scores)
    print(f"\ngames {len(s)} | median {np.median(s):.0f} | mean {s.mean():.0f} | "
          f"min {s.min()} | max {s.max()} | {np.mean(latency):.2f} ms/decision")
    json.dump({"game": args.game, "model": str(model_path), "max_steps": max_steps,
               "scores": scores, "ms_per_decision": float(np.mean(latency))},
              open(out / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
