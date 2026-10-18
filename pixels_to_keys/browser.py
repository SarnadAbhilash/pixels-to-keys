"""Browser plumbing: serve the game, read frames, keys and game state."""
import base64
import functools
import io
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
from PIL import Image

GAME_DIR = Path(__file__).resolve().parent.parent / "game"
FRAME_SIZE = (160, 40)  # (width, height) of every observation
DEFAULT_PORT = 8765

# Injected into the page while recording. Remembers whether jump was pressed
# since Python last asked, so no key press between two frames is lost.
KEY_LOGGER = """
window.__jumpPressed = false;
window.addEventListener('keydown', e => {
  if (e.code === 'Space' || e.code === 'ArrowUp') window.__jumpPressed = true;
});
"""


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve_game(port: int = DEFAULT_PORT) -> str:
    """Serve the game folder on localhost and return the game URL.

    The canvas can only be read back from a page served over http, not file://.
    If the port is already serving (another process), that server is reused.
    """
    handler = functools.partial(_QuietHandler, directory=str(GAME_DIR))
    try:
        server = HTTPServer(("localhost", port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    except OSError:
        pass
    return f"http://localhost:{port}/index.html"


def to_frame(data_url: str) -> np.ndarray:
    """Canvas data URL -> uint8 array (40, 160). Objects are bright, background is 0."""
    img = Image.open(io.BytesIO(base64.b64decode(data_url.split(",")[1]))).convert("RGBA")
    # The canvas background is transparent: composite onto white before converting.
    gray = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img).convert("L")
    small = gray.resize(FRAME_SIZE, Image.BOX)
    return 255 - np.asarray(small, dtype=np.uint8)


def read_frame(page) -> np.ndarray:
    return to_frame(page.evaluate("document.querySelector('canvas').toDataURL()"))


def pop_jump_pressed(page) -> bool:
    """True if jump was pressed since the last call. Resets the flag."""
    return page.evaluate(
        "() => { const j = window.__jumpPressed; window.__jumpPressed = false; return j; }"
    )


def game_state(page):
    """Return (crashed, score)."""
    return page.evaluate("""() => {
        const r = Runner.instance_;
        return [r.crashed, r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan))];
    }""")
