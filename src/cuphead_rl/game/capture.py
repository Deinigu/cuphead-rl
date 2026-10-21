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
(``"h264_amf"``, ``"h264_nvenc"``, ``"h264_qsv"``, constant QP ``crf``); ``max_height`` downscales.
Measured with the game in free-run: libx264 on the full-resolution capture takes enough
CPU that the agent's decisions came every ~9 ticks instead of 4 (frame_skip); with h264_amf the
cadence stays at 4 ticks. ffmpeg must not run at below-normal priority: measured in paced lockstep,
it then took only ~5 frames/s from ddagrab and duplicated the rest, so the video froze for seconds.

Audio: with ``audio=True`` the default output device is recorded through WASAPI loopback (the
``soundcard`` package), so the file gets whatever the PC plays: mute other apps while recording.
The audio is stamped with the wall clock like the video, trimmed or padded to start at
``start_epoch`` and muxed into the MP4 (AAC) when the recorder stops.
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


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


class LoopbackRecorder:
    """Records the default output device (WASAPI loopback) in a separate process
    (``cuphead_rl.game.loopback``), stamped with the wall clock."""

    def __init__(self, samplerate: int = 48000):
        self.samplerate = samplerate
        self.start_epoch: float | None = None
        self._proc: subprocess.Popen | None = None
        self._wav: Path | None = None

    def start(self, wav: Path) -> "LoopbackRecorder":
        import os
        import sys

        self._wav = Path(wav)
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        self._proc = subprocess.Popen([sys.executable, "-m", "cuphead_rl.game.loopback", str(self._wav)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        line: list[bytes] = []
        reader = threading.Thread(target=lambda: line.append(self._proc.stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout=10)
        if not line or not line[0].strip():
            self._proc.kill()
            err = self._proc.stderr.read().decode(errors="replace")[-500:]
            raise RuntimeError(f"audio capture failed to start: {err}")
        self.start_epoch = float(line[0])
        return self

    def stop(self):
        """Stop and return the samples as an (n, 2) float32 array in [-1, 1]."""
        import wave

        import numpy as np

        proc, self._proc = self._proc, None
        if proc is None:
            return np.zeros((0, 2), np.float32)
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        with wave.open(str(self._wav), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels())
        self._wav.unlink()
        return pcm.astype(np.float32) / 32767


def align_audio(samples, audio_start: float, video_start: float, samplerate: int):
    """Shift ``samples`` so sample 0 plays at ``video_start``: drop or prepend silence."""
    import numpy as np

    lag = int(round((audio_start - video_start) * samplerate))
    if lag > 0:
        return np.concatenate([np.zeros((lag, samples.shape[1]), samples.dtype), samples])
    return samples[-lag:]


def mux_audio(video: Path, samples, samplerate: int) -> Path:
    """Write ``samples`` into ``video`` as AAC (video stream copied), padded/cut to the video length."""
    import wave

    import numpy as np

    # Pad/cut here: ffmpeg's apad + -shortest never ends when the video stream is copied.
    _, secs = imageio_ffmpeg.count_frames_and_secs(str(video))
    n = int(round(secs * samplerate))
    samples = samples[:n] if len(samples) >= n else np.concatenate(
        [samples, np.zeros((n - len(samples), samples.shape[1]), samples.dtype)])
    wav = video.with_suffix(".audio.wav")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.tobytes())
    tmp = video.with_suffix(".mux.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(tmp)],
                   check=True, stdin=subprocess.DEVNULL)
    tmp.replace(video)
    wav.unlink()
    return video


class WindowRecorder:
    def __init__(self, out: str | Path, window_title: str = "Cuphead", fps: int = 30, crf: int = 23,
                 method: str = "ddagrab", output_idx: int = 0, encoder: str = "libx264", audio: bool = False,
                 max_height: int | None = None):
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
        self.max_height = max_height
        self.proc: subprocess.Popen | None = None
        self._finalizer: weakref.finalize | None = None
        self.start_epoch: float | None = None
        self._stderr_tail: list[str] = []
        self._reader: threading.Thread | None = None
        self.audio = audio
        self._audio: LoopbackRecorder | None = None

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
            if self.max_height and h > self.max_height:
                vf += f",scale=-2:{self.max_height}:flags=bicubic"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel", "verbose",
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
        if self.audio:  # started before ffmpeg, so it covers the first video frame
            self._audio = LoopbackRecorder().start(self.out.with_suffix(".loopback.wav"))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=0)
        self._finalizer = weakref.finalize(self, _kill, self.proc)  # also runs at interpreter exit
        # Read stderr continuously (a full pipe would stall ffmpeg) and pick up the start time.
        self._reader = threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True)
        self._reader.start()
        deadline = time.monotonic() + 5.0
        while self.start_epoch is None and self.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.proc.poll() is not None:
            self._reader.join(timeout=2)
            if self._audio is not None:
                self._audio.stop()
                self._audio = None
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
            self._stderr_tail = (self._stderr_tail + [line])[-400:]

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
        self.out.with_suffix(".ffmpeg.log").write_text("".join(self._stderr_tail), encoding="utf-8")
        if self._finalizer is not None:
            self._finalizer.detach()
        if killed or proc.returncode not in (0, None) or not self.out.exists() or self.out.stat().st_size < 1024:
            warnings.warn(f"capture {self.out} may be truncated (ffmpeg exit code {proc.returncode}, killed={killed})")
        if self._audio is not None:
            rec, self._audio = self._audio, None
            samples = rec.stop()
            if self.start_epoch is None or rec.start_epoch is None or not len(samples):
                warnings.warn(f"capture {self.out}: no audio (no sync time or no samples)")
            elif self.out.exists():
                aligned = align_audio(samples, rec.start_epoch, self.start_epoch, rec.samplerate)
                mux_audio(self.out, aligned, rec.samplerate)
        return self.out

    def __enter__(self) -> "WindowRecorder":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
