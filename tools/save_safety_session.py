"""Exercise every save-writing path the bridge can hit, as Mugman, for the save-safety check.

    python tools/check_saves.py --out data/save_safety/before.json   # game closed
    (launch the game)
    python tools/save_safety_session.py [--video videos/game_mugman.mp4]
    (close the game)
    python tools/check_saves.py --assert-unchanged data/save_safety/before.json

Session: 3 seeded resets, one natural death (Mugman idles until Goopy kills him), one win (no-hit,
boss damaged to near death through the debug command, finished by peashots), which loads the win
screen and makes the game save the level result. Records a short capture of the fight.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuphead_rl.game.client import BridgeClient  # noqa: E402
from cuphead_rl.spec import Action  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "videos" / "game_mugman.mp4")
    args = ap.parse_args()
    c = BridgeClient(timeout=60)
    hello = c.connect(wait=120)
    print("hello:", {k: hello.get(k) for k in ("character", "save_slot", "save_guard", "saves_blocked", "no_hit", "loadout")})
    assert hello.get("save_slot") == 2, hello
    try:
        c.configure(lockstep=True, frames=4, time_scale=1.0, stream_every=0, debug=False)
        for i in range(3):
            m = c.reset(seed=100 + i, character="mugman")
            assert m["character"] == "mugman", m.get("character")
            print(f"reset {i}: tick {m['tick']} character {m['character']}")

        # a natural death, recorded (free-run keeps the capture at real-time speed)
        from cuphead_rl.game.capture import WindowRecorder

        c.configure(lockstep=False, stream_every=1)
        with WindowRecorder(args.video, fps=30):
            m = c.reset(seed=7, character="mugman")
            for _ in range(60 * 60):
                m = c.step_free(Action(shoot=True), 4)
                if not m.get("in_fight") or m["lost"]:
                    break
        print(f"death: lost={m.get('lost')} time={m.get('time', 0):.1f}s -> {args.video}")
        assert m.get("lost"), "Mugman did not die within 60 s"

        # a win: no-hit, boss to 1 HP through the bridge, then peashots finish it
        c.configure(lockstep=True, frames=4, stream_every=0)
        m = c.reset(seed=8, character="mugman")
        c.invincible(True)
        for _ in range(4000):
            b = m["boss"]
            if m.get("won") or not m.get("in_fight"):
                break
            if b["hp"] > 5 and b["vulnerable"]:
                m = c.damage(b["hp"] - 4)
            dx = b["x"] - m["player"]["x"]
            aim = 1 if b["entity"] == "tomb" and abs(dx) < 60 else 0
            m = c.step_lockstep(Action(move=0 if abs(dx) < 40 else (1 if dx > 0 else -1), shoot=True, aim=aim), 4)
        print(f"win: won={m.get('won')} in_fight={m.get('in_fight')}")
        assert m.get("won") or not m.get("in_fight"), "did not win"
        # let the game reach the win screen (where the result is saved), then ask the plugin
        # how many writes to other slots it dropped
        c.configure(lockstep=False, stream_every=0)
        time.sleep(15)
        c.close()
        c = BridgeClient(timeout=60)
        hello = c.connect(wait=30)
        print("hello after win:", {k: hello.get(k) for k in ("scene", "save_slot", "saves_blocked")})
    finally:
        try:
            c.invincible(False)
            c.configure(lockstep=False, time_scale=1.0, stream_every=0)
        except Exception:  # noqa: BLE001
            pass
        c.close()


if __name__ == "__main__":
    main()
