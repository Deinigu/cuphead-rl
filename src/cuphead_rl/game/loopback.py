"""Record the default output device (WASAPI loopback) to a 16-bit WAV until stdin closes.

    python -m cuphead_rl.game.loopback out.wav

Prints the Unix-epoch time of the first sample on stdout, then records. It runs as its own
process (see ``capture.LoopbackRecorder``) so that the recording loop never competes for the GIL
with the client stepping the game: as a thread it slowed paced play to ~0.78x real time.
"""

from __future__ import annotations

import sys
import threading
import time
import wave

import numpy as np

SAMPLERATE = 48000
BLOCK = 480  # 10 ms


def main() -> None:
    import soundcard as sc  # imported in this (main) thread: its import initializes COM here

    out = sys.argv[1]
    stop = threading.Event()

    def wait_stdin() -> None:
        sys.stdin.read()
        stop.set()

    threading.Thread(target=wait_stdin, daemon=True).start()
    mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    with wave.open(out, "wb") as w, mic.recorder(samplerate=SAMPLERATE, channels=2) as rec:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLERATE)
        first = True
        while not stop.is_set():
            data = rec.record(numframes=BLOCK)
            if first:  # the first block ends now
                print(f"{time.time() - len(data) / SAMPLERATE:.6f}", flush=True)
                first = False
            w.writeframes((np.clip(data, -1.0, 1.0) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    main()
