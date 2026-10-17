"""Record human demonstrations: frames + jump presses at 20 Hz, one file per round.

    python -m pixels_to_keys.record --out data
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from .browser import KEY_LOGGER, game_state, pop_jump_pressed, read_frame, serve_game

TICK = 0.05  # seconds between frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    url = serve_game()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=[
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ])
        page = browser.new_page()
        page.add_init_script(KEY_LOGGER)
        page.goto(url)
        print("Click the game and press SPACE to start. Ctrl+C to stop.")

        try:
            while True:
                page.wait_for_function(
                    "Runner.instance_.playing && !Runner.instance_.crashed", timeout=0)
                pop_jump_pressed(page)
                frames, actions = [], []
                next_tick = time.time()
                while True:
                    frames.append(read_frame(page))
                    actions.append(int(pop_jump_pressed(page)))
                    crashed, score = game_state(page)
                    if crashed:
                        break
                    next_tick += TICK
                    time.sleep(max(0.0, next_tick - time.time()))

                n = len(list(args.out.glob("round_*.npz"))) + 1
                np.savez_compressed(args.out / f"round_{n:03d}.npz", frames=np.array(frames),
                                    actions=np.array(actions), score=score)
                print(f"round {n}: score {score}, {len(frames)} frames, {sum(actions)} jumps")
        except (KeyboardInterrupt, PlaywrightError):
            print(f"Stopped. Rounds saved in {args.out}/")
            os._exit(0)


if __name__ == "__main__":
    main()
