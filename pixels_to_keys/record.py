"""Record human demonstrations: frames + jump/flap presses, one file per round.

    python -m pixels_to_keys.record --game dino --out data/dino
    python -m pixels_to_keys.record --game flappy --out data/flappy

Dino is recorded in the real browser game at 20 Hz. Flappy is recorded at 30 Hz,
one frame per game tick. Recorded rounds use seeds >= 1000, so they never overlap
with the evaluation seeds (1-50).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

TICK = 0.05  # Dino recording interval (s)


def next_round(out: Path) -> int:
    return len(list(out.glob("round_*.npz"))) + 1


def save_round(out, frames, actions, score):
    n = next_round(out)
    np.savez_compressed(out / f"round_{n:03d}.npz", frames=np.array(frames),
                        actions=np.array(actions), score=score)
    print(f"round {n}: score {score}, {len(frames)} frames, {sum(actions)} presses")


def record_dino(out: Path):
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from .browser import KEY_LOGGER, game_state, pop_jump_pressed, read_frame, serve_game

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
                save_round(out, frames, actions, score)
        except (KeyboardInterrupt, PlaywrightError):
            print(f"Stopped. Rounds saved in {out}/")
            os._exit(0)


def record_flappy(out: Path):
    from .envs.flappy import FPS, FlappyEnv

    env = FlappyEnv(render=True)
    pg = env.pygame
    press_keys = (pg.K_SPACE, pg.K_UP)
    print("SPACE / UP / click to flap. Close the window or press ESC to stop.")

    def wait_for_start():
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    return False
                if (e.type == pg.KEYDOWN and e.key in press_keys) or e.type == pg.MOUSEBUTTONDOWN:
                    return True
            env.config.clock.tick(FPS)

    while True:
        frame = env.reset(seed=1000 + next_round(out))
        env.show()
        if not wait_for_start():
            break
        # action[t] = flap applied right before frame t, same convention as Dino
        frames, actions = [frame], [0]
        flap = True  # the start key press is the first flap
        while True:
            frame, crashed, score = env.step(flap)
            env.show()
            frames.append(frame)
            actions.append(int(flap))
            if crashed:
                break
            env.config.clock.tick(FPS)
            flap = False
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    save_round(out, frames, actions, score)
                    return
                if (e.type == pg.KEYDOWN and e.key in press_keys) or e.type == pg.MOUSEBUTTONDOWN:
                    flap = True
        save_round(out, frames, actions, score)
    print(f"Stopped. Rounds saved in {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", choices=("dino", "flappy"), default="dino")
    ap.add_argument("--out", type=Path, default=None, help="default: data/<game>")
    args = ap.parse_args()
    out = args.out or Path("data") / args.game
    out.mkdir(parents=True, exist_ok=True)
    (record_dino if args.game == "dino" else record_flappy)(out)


if __name__ == "__main__":
    main()
