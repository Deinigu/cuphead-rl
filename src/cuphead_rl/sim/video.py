"""Record an episode to MP4, rendering every 60 Hz simulation frame (not just decision steps)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np

from cuphead_rl import spec


def record_episode(
    env,
    policy_fn: Callable[[np.ndarray], Any],
    path: str | Path,
    fps: int = 60,
    overlay_fn: Callable[[], dict[str, Any] | None] | None = None,
    max_seconds: float | None = None,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    crf: int = 20,
) -> dict[str, Any]:
    """Run one episode with ``policy_fn(obs) -> action`` and write it to ``path`` (H.264).

    ``env`` may be wrapped (e.g. frame stacking); rendering goes through ``env.unwrapped``,
    which must be a ``GoopyEnv``. ``overlay_fn()`` is called for every rendered frame and its
    dict is drawn in the side panel. ``fps`` below 60 drops frames so playback stays real-time.
    Encoding is libx264 CRF ``crf`` (preset medium).
    """
    base = env.unwrapped
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, round(spec.GAME_FPS / fps))
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=None,
                                pixelformat="yuv420p", macro_block_size=8,
                                ffmpeg_params=["-crf", str(crf), "-preset", "medium"])
    counter = {"n": 0}

    def on_frame() -> None:
        if counter["n"] % stride == 0:
            writer.append_data(base.render(overlay_fn() if overlay_fn is not None else None))
        counter["n"] += 1

    total_reward = 0.0
    info: dict[str, Any] = {}
    base.frame_callback = on_frame
    try:
        obs, info = env.reset(seed=seed, options=options)
        while True:
            obs, r, terminated, truncated, info = env.step(policy_fn(obs))
            total_reward += float(r)
            if terminated or truncated or (max_seconds is not None and info["time"] >= max_seconds):
                break
        # Hold the final frame (KO / death banner) for a second.
        for _ in range(spec.GAME_FPS // stride):
            writer.append_data(base.render(overlay_fn() if overlay_fn is not None else None))
    finally:
        base.frame_callback = None
        writer.close()
    return {
        "won": bool(info.get("won", False)),
        "phase": int(info.get("phase", 0)),
        "boss_hp_frac": float(info.get("boss_hp_frac", 1.0)),
        "player_hp": int(info.get("player_hp", 0)),
        "time": float(info.get("time", 0.0)),
        "total_reward": total_reward,
        "parries": int(info.get("parries", 0)),
        "frames": counter["n"],
        "path": str(path),
    }
