"""Deterministic Dino environment for evaluation.

The game normally runs on the browser's real clock, so results depend on machine
load and window focus. This environment takes control of the page:

- Clock: performance.now and requestAnimationFrame are replaced. The game only
  advances when step() is called; each step is exactly 3 game frames (50 ms).
- Randomness: Math.random is seeded per game, so seed N is always the same course.
- Start state: the dino is placed where it stands in normal play.

Same model + same seeds -> same scores, on any machine.
"""
from playwright.sync_api import Browser

from .browser import DEFAULT_PORT, serve_game, to_frame

FRAMES_PER_STEP = 3  # 3 frames at 60 fps = 50 ms, matching the 20 Hz recording rate
DINO_X = 24          # dino x position in normal play (measured from recordings)

_CONTROL = """
(() => {
  let now = 0;
  performance.now = () => now;
  let nextId = 1;
  const queue = new Map();
  window.requestAnimationFrame = cb => { const id = nextId++; queue.set(id, cb); return id; };
  window.cancelAnimationFrame = id => queue.delete(id);
  window.__advance = frames => {
    for (let i = 0; i < frames; i++) {
      now += 1000 / 60;
      const cbs = [...queue.values()];
      queue.clear();
      cbs.forEach(cb => cb(now));
    }
  };
  const seed = Number(new URLSearchParams(location.search).get('seed') || 1);
  let s = (seed * 2654435761) >>> 0 || 1;           // xorshift32
  Math.random = () => {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
})();
"""

_START = """() => {
    const r = Runner.instance_;
    r.onVisibilityChange = () => {};              // never pause on blur
    const start = r.startGame;
    r.startGame = function () {                   // run once; normally fired by a CSS animation
        if (this.__started) return;
        this.__started = true;
        start.call(this);
    };
}"""

_FINISH_INTRO = """dinoX => {
    const r = Runner.instance_;
    for (let i = 0; i < 120 && (r.tRex.jumping || r.tRex.jumpCount < 1); i++) __advance(1);
    r.startGame();
    r.tRex.xPos = dinoX;
}"""


_FAST_FORWARD = """speed => {
    const r = Runner.instance_;
    const n = (speed - r.config.SPEED) / r.config.ACCELERATION;  // frames to reach this speed
    r.currentSpeed = speed;
    r.distanceRan = r.config.SPEED * n + r.config.ACCELERATION * n * n / 2;
}"""


class DinoEnv:
    def __init__(self, browser: Browser, port: int = DEFAULT_PORT):
        self.url = serve_game(port)
        self.page = browser.new_page()
        self.page.add_init_script(_CONTROL)

    def reset(self, seed: int, start_speed: float = None):
        """Load a fresh game for this seed and return the first frame.

        start_speed (training only) starts the game as if it had already been
        running until it reached that speed.
        """
        self.page.goto(f"{self.url}?seed={seed}")
        self.page.wait_for_function("typeof Runner !== 'undefined' && Runner.instance_",
                                    polling=50)
        self.page.evaluate(_START)
        self.page.keyboard.press("Space")          # the game starts with one jump
        self.page.evaluate(_FINISH_INTRO, DINO_X)
        if start_speed:
            self.page.evaluate(_FAST_FORWARD, start_speed)
        frame, _, _ = self._observe()
        return frame

    def step(self, jump: bool):
        """Apply the action, advance 50 ms. Returns (frame, crashed, score)."""
        if jump:
            self.page.keyboard.press("Space")
        self.page.evaluate(f"__advance({FRAMES_PER_STEP})")
        return self._observe()

    def speed(self) -> float:
        return self.page.evaluate("Runner.instance_.currentSpeed")

    def _observe(self):
        data_url, crashed, score = self.page.evaluate("""() => {
            const r = Runner.instance_;
            return [document.querySelector('canvas').toDataURL(), r.crashed,
                    r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan))];
        }""")
        return to_frame(data_url), crashed, score
