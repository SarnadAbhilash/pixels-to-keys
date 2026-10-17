"""Flappy Bird environment built on FlapPyBird (games/flappy, MIT).

- Deterministic: the game loop is stepped from Python; pipe gaps come from a seeded RNG.
- One step = one game tick at 30 fps. Action: flap or no-op.
- Fixed sprites: day background, yellow bird, green pipes.

Observations use the same format as Dino: 160x40 uint8, objects bright on a dark
background. Each pixel is its color distance from the empty background image
(the Dino equivalent is 255 - gray, since its background is white). The floor
strip below the play area is cropped out.
"""
import contextlib
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ..browser import FRAME_SIZE

GAME_DIR = Path(__file__).resolve().parents[2] / "games" / "flappy"
PLAY_HEIGHT = 404  # rows above the floor
FPS = 30


class _Silent:
    def play(self):
        pass


class _NoSounds:
    die = hit = point = swoosh = wing = _Silent()


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class FlappyEnv:
    def __init__(self, render: bool = False):
        """render=True opens a window (for human play); otherwise runs headless."""
        if not render:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        if str(GAME_DIR) not in sys.path:
            sys.path.insert(0, str(GAME_DIR))
        from flappybird.utils import GameConfig, Images, Window

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        screen = pygame.display.set_mode((288, 512))
        with _cwd(GAME_DIR):
            images = Images()
            load = pygame.image.load
            images.background = load("assets/sprites/background-day.png").convert()
            images.player = tuple(load(f"assets/sprites/yellowbird-{k}flap.png").convert_alpha()
                                  for k in ("up", "mid", "down"))
            pipe = load("assets/sprites/pipe-green.png").convert_alpha()
            images.pipe = (pygame.transform.flip(pipe, False, True), pipe)
        self.config = GameConfig(screen=screen, clock=pygame.time.Clock(), fps=FPS,
                                 window=Window(288, 512), images=images, sounds=_NoSounds())
        self._background = self._pixels(images.background)
        self.score = 0

    def reset(self, seed: int):
        """Start a new game with this seed and return the first frame."""
        from flappybird.entities import Background, Floor, Pipes, Player, PlayerMode, Score
        random.seed(seed)
        c = self.config
        self.background, self.floor = Background(c), Floor(c)
        self.player, self.pipes, self.scoreboard = Player(c), Pipes(c), Score(c)
        self.player.set_mode(PlayerMode.NORMAL)
        self.score = 0
        self._tick()
        return self.frame()

    def step(self, flap: bool):
        """Apply the action, advance one tick. Returns (frame, crashed, score)."""
        if flap:
            self.player.flap()
        for pipe in self.pipes.upper:
            if self.player.crossed(pipe):
                self.scoreboard.add()
        self._tick()
        crashed = self.player.collided(self.pipes, self.floor)
        self.score = self.scoreboard.score
        return self.frame(), crashed, self.score

    def frame(self) -> np.ndarray:
        rgb = self._pixels(self.config.screen).astype(np.int16)
        diff = np.abs(rgb - self._background).max(axis=2)[:PLAY_HEIGHT].astype(np.uint8)
        return np.asarray(Image.fromarray(diff).resize(FRAME_SIZE, Image.BOX), dtype=np.uint8)

    def show(self):
        """Push the current frame to the window (human play only)."""
        self.pygame.display.update()

    def _tick(self):
        for entity in (self.background, self.floor, self.pipes, self.scoreboard, self.player):
            entity.tick()

    def _pixels(self, surface):
        return self.pygame.surfarray.array3d(surface).swapaxes(0, 1)
