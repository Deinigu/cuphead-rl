"""Record the Cuphead window to MP4 with the ffmpeg binary bundled by imageio-ffmpeg.

    rec = WindowRecorder("videos/game_smoke.mp4")
    rec.start(); ...; rec.stop()

``method="ddagrab"`` (default) uses the Desktop Duplication API on monitor ``output_idx`` and
crops to the game window. It captures whatever is on screen, so the game must be visible and
on top, but it works in fullscreen. ``method="gdigrab"`` reads the window through GDI. That
works for a composed (windowed) game but returns a stale frame forever when Cuphead runs
fullscreen and in the foreground: measured, a whole episode showed only the title screen.
ffmpeg is stopped cleanly (``q`` on stdin) so the MP4 gets its index; if it has to be killed the
file is likely truncated and a warning is raised. A finalizer kills a forgotten recorder at exit.

Sync: frames are stamped with the wall clock (gdigrab natively; ddagrab via
``-use_wallclock_as_timestamps``). ``start_epoch`` is the Unix-epoch time of
the first captured frame (from ffmpeg's ``Input #0 ... start:`` line), so video time ``t``
corresponds to ``time.time() == start_epoch + t``; it is set once capture has started.

Encoding: ``encoder="libx264"`` (default, CRF ``crf``, preset veryfast) or a hardware encoder
(``"h264_amf"``, ``"h264_nvenc"``, ``"h264_qsv"``, constant QP ``crf``). ffmpeg runs at below-normal
priority. Measured with the game in free-run: libx264 on the full-resolution capture takes enough
CPU that the agent's decisions came every ~9 ticks instead of 4 (frame_skip); with h264_amf the
cadence stays at 4 ticks.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import threading
import time
import warnings
import weakref
from pathlib import Path

import imageio_ffmpeg


def _window_rect(title: str) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of the top-level window with this exact title, or None."""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return None
    r = (ctypes.c_long * 4)()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r[0], r[1], r[2] - r[0], r[3] - r[1]


_HW_ENCODER_ARGS = {
    "h264_amf": lambda q: ["-quality", "quality", "-rc", "cqp", "-qp_i", q, "-qp_p", q, "-qp_b", q],
    "h264_nvenc": lambda q: ["-preset", "p5", "-rc", "constqp", "-qp", q],
    "h264_qsv": lambda q: ["-global_quality", q],
}
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


class WindowRecorder:
    def __init__(self, out: str | Path, window_title: str = "Cuphead", fps: int = 30, crf: int = 23,
                 method: str = "ddagrab", output_idx: int = 0, encoder: str = "libx264"):
        if method not in ("ddagrab", "gdigrab"):
            raise ValueError(f"method must be 'ddagrab' or 'gdigrab', got {method!r}")
        if encoder != "libx264" and encoder not in _HW_ENCODER_ARGS:
            raise ValueError(f"encoder must be 'libx264' or one of {sorted(_HW_ENCODER_ARGS)}, got {encoder!r}")
        self.encoder = encoder
        self.method = method
        self.output_idx = output_idx
        self.out = Path(out)
        self.window_title = window_title
        self.fps = fps
        self.crf = crf
        self.proc: subprocess.Popen | None = None
        self._finalizer: weakref.finalize | None = None
        self.start_epoch: float | None = None
        self._stderr_tail: list[str] = []
        self._reader: threading.Thread | None = None

    def start(self) -> "WindowRecorder":
        self.out.parent.mkdir(parents=True, exist_ok=True)
        even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        if self.method == "gdigrab":
            source = ["-f", "gdigrab", "-framerate", str(self.fps), "-draw_mouse", "0", "-i", f"title={self.window_title}"]
            vf = even
        else:
            rect = _window_rect(self.window_title)
            if rect is None:
                raise RuntimeError(f"no visible window titled '{self.window_title}'")
            x, y, w, h = rect
            source = ["-use_wallclock_as_timestamps", "1", "-f", "lavfi",
                      "-i", f"ddagrab=output_idx={self.output_idx}:framerate={self.fps}:draw_mouse=0"]
            # The window rect is in desktop coordinates; monitor 0 starts at (0, 0).
            vf = f"hwdownload,format=bgra,crop={w}:{h}:{max(0, x)}:{max(0, y)},{even}"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel", "info",
            "-nostats",
            *source,
            "-vf", vf,
            "-r", str(self.fps),
            "-c:v", self.encoder,
            *(["-preset", "veryfast", "-crf", str(self.crf)] if self.encoder == "libx264"
              else _HW_ENCODER_ARGS[self.encoder](str(self.crf))),
            "-pix_fmt", "yuv420p",
            str(self.out),
        ]
        self.start_epoch = None
        self._stderr_tail = []
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_BELOW_NORMAL_PRIORITY_CLASS)
        self._finalizer = weakref.finalize(self, _kill, self.proc)  # also runs at interpreter exit
        # Read stderr continuously (a full pipe would stall ffmpeg) and pick up the start time.
        self._reader = threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True)
        self._reader.start()
        deadline = time.monotonic() + 5.0
        while self.start_epoch is None and self.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.proc.poll() is not None:
            self._reader.join(timeout=2)
            raise RuntimeError(f"ffmpeg failed to start capturing '{self.window_title}': {''.join(self._stderr_tail)}")
        if self.start_epoch is None:
            warnings.warn("capture: no start time from ffmpeg; video/log sync unavailable")
        return self

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode(errors="replace")
            if self.start_epoch is None and "gdigrab" not in line:
                m = re.search(r"start:\s*([0-9.]+)", line)
                if m and float(m.group(1)) > 1e9:  # epoch seconds, not a stream-relative start
                    self.start_epoch = float(m.group(1))
            self._stderr_tail = (self._stderr_tail + [line])[-40:]

    def stop(self) -> Path:
        proc, self.proc = self.proc, None
        if proc is None:
            return self.out
        killed = False
        try:
            try:
                proc.stdin.write(b"q")
                proc.stdin.flush()
                proc.stdin.close()
            except OSError:  # ffmpeg already exited
                pass
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            killed = True
        if self._reader is not None:
            self._reader.join(timeout=5)
        if self._finalizer is not None:
            self._finalizer.detach()
        if killed or proc.returncode not in (0, None) or not self.out.exists() or self.out.stat().st_size < 1024:
            warnings.warn(f"capture {self.out} may be truncated (ffmpeg exit code {proc.returncode}, killed={killed})")
        return self.out

    def __enter__(self) -> "WindowRecorder":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
