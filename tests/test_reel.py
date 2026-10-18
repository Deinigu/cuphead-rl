"""scripts/make_reel.py on two tiny synthetic runs."""

import json
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_reel  # noqa: E402

COLORS = [(220, 40, 40), (40, 200, 40), (40, 60, 220)]  # one solid colour per second


def write_clip(path: Path, seconds: int, size=(160, 90), fps: int = 60) -> None:
    gen = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", quality=None,
                                      pix_fmt_out="yuv420p", macro_block_size=2,
                                      output_params=["-crf", "18", "-preset", "ultrafast"])
    gen.send(None)
    for i in range(seconds * fps):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:] = COLORS[(i // fps) % len(COLORS)]
        gen.send(frame.tobytes())
    gen.close()


def make_run(root: Path, name: str, entries: list[tuple[int, list[str], int | None, bool]]) -> Path:
    """entries: (steps, tags, clip seconds or None for a missing file, won)."""
    vids = root / name / "videos"
    vids.mkdir(parents=True)
    with open(vids / "index.jsonl", "w") as f:
        for steps, tags, secs, won in entries:
            file = f"{steps:010d}_{'+'.join(tags)}.mp4"
            if secs is not None:
                write_clip(vids / file, secs)
            f.write(json.dumps({"file": file, "steps": steps, "tags": tags,
                                "eval": {"win_rate": 0.5 if won else 0.0, "boss_damage": 0.4, "mean_phase": 1.0,
                                         "episodes": 4},
                                "episode": {"won": won, "phase": 2 if won else 0, "time": 3.0,
                                            "boss_hp_frac": 0.0 if won else 0.9}}) + "\n")
    return root / name


def read_all(path: Path) -> tuple[dict, list[np.ndarray]]:
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)
    w, h = meta["size"]
    frames = [np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in gen]
    return meta, frames


def test_reel_duration_resolution_and_win_keeps_the_end(tmp_path):
    run = make_run(tmp_path, "ppo_test", [
        (2_000_000, ["periodic"], 3, True),  # a win longer than the cap: keep its last 2 s
        (8, ["untrained"], 1, False),  # listed later, shown first (step order)
        (3_000_000, ["first_win"], None, True),  # file missing: skipped with a warning
    ])
    out = tmp_path / "reel.mp4"
    with pytest.warns(UserWarning, match="missing clip"):
        res = make_reel.make_reel([run], out, max_clip=2.0, card_seconds=0.5, preset="ultrafast")
    assert res["clips"] == 2 and res["skipped"] == 1
    meta, frames = read_all(out)
    assert meta["size"] == (1280, 720) and meta["fps"] == 60
    assert len(frames) == 2 * 30 + 60 + 120  # two 0.5 s cards + 1 s + 2 s (trimmed) clips
    centre = lambda f: f[360, 640].astype(int)  # noqa: E731
    first_of_win = centre(frames[30 + 60 + 30 + 5])
    assert np.abs(first_of_win - COLORS[1]).max() < 40  # starts at second 1 (green), not red
    assert np.abs(centre(frames[-5]) - COLORS[2]).max() < 40  # and ends on the knockout


def test_side_by_side_pairs_comparable_steps(tmp_path):
    a = make_run(tmp_path, "a", [(1_000_000, ["periodic"], 1, False), (5_000_000, ["periodic"], 1, False)])
    b = make_run(tmp_path, "b", [(1_050_000, ["first_phase2"], 2, False), (9_000_000, ["first_win"], 1, True)])
    out = tmp_path / "vs.mp4"
    res = make_reel.make_versus(a, b, out, max_clip=None, card_seconds=0.5, preset="ultrafast")
    assert res["pairs"] == 1  # 5M has no partner within 10 %
    meta, frames = read_all(out)
    assert meta["size"] == (1280, 720)
    assert len(frames) == 30 + 120  # card + the longer clip
    top = (720 - 360) // 2
    assert np.abs(frames[-1][top + 180, 320].astype(int) - COLORS[0]).max() < 40  # A holds its last frame
    assert np.abs(frames[-1][top + 180, 960].astype(int) - COLORS[1]).max() < 40


def test_tag_words():
    assert make_reel.headline_tags(["first_phase2", "damage_25", "periodic"]) == [
        "First time reaching phase 2", "Boss damage 25 %"]
    assert make_reel.headline_tags(["periodic"]) == ["Progress check"]
    assert make_reel.tag_text("winrate_50") == "Win rate 50 %"
    assert make_reel.steps_text(1_500_010) == "1.50M steps"
