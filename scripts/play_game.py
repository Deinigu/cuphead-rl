"""Play a trained agent in the real Cuphead (CupheadRLBridge plugin) and report its stats.

    python -m uv run python scripts/play_game.py runs/<name>/best.zip --episodes 10
    python -m uv run python scripts/play_game.py runs/<name>/best.zip --episodes 5 --record videos/game_eval

The game must be running with the plugin (the client refuses to run unless the plugin reports its
save guard complete). The run's ``config.yaml`` (next to the checkpoint or one level up) gives
frame_skip, frame_stack, the reward and ``env_kwargs.max_seconds``.

Stats are printed and written as JSON: the keys of ``cuphead_rl.train.evaluate.summarize`` plus
``per_episode`` (EpisodeResult fields + wall-clock seconds) and the run settings.

``--record DIR`` writes, per episode, ``DIR/episode_XX.mp4`` (window capture, 60 fps, from
the level (re)load to ~2 s after the end) and ``DIR/episode_XX.jsonl``: line 1 is
``{"meta": ...}`` (incl. ``video_start_epoch``, the wall-clock time of the first video frame),
then one line per decision ``{step, t, wall, action, value, probs, reward, return, phase,
boss_hp_frac, player_hp, state}`` (``state``: post-step player ``[x, y, grounded, dashing,
invuln]``, boss ``[move, x, y, half_w, half_h, state_time]``, hazards ``[[kind, x, y, half_w,
half_h], ...]``, for hit-cause analysis). ``t`` is game time since the episode started; ``wall`` is
``time.time()`` when the action was sent. ``scripts/overlay_game_video.py`` turns the pair into
the side-panel video.

Modes: lockstep (default) is the evaluation setting, about 2.9x real time. A wall-clock capture
of lockstep play looks fast-forwarded, so ``--record`` defaults to ``--mode paced``: the same
lockstep play, paced to real time one tick at a time (see ``PacedGameEnv``). ``--mode free`` plays
the free-running game (uncapped render rate, variable Update deltaTime, 0-1 tick action latency).
That is not the setting the sim was calibrated on: r2_curriculum_hit3 won 16/20 in lockstep and
0/10 in free-run.

``--log DIR`` writes the same ``episode_XX.jsonl`` logs without a video (e.g. for lockstep
evaluations, for ``scripts/game_hits.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuphead_rl import spec  # noqa: E402
from cuphead_rl.game.env import CupheadGameEnv  # noqa: E402
from cuphead_rl.train.agent import AgentPolicy, load_model  # noqa: E402
from cuphead_rl.train.config import ExperimentConfig  # noqa: E402
from cuphead_rl.train.evaluate import EpisodeResult, summarize  # noqa: E402

END_HOLD_SECONDS = 2.0  # keep capturing after the end, for the knockout / death screen


def compact_state(s: spec.GameState) -> dict:
    """The post-step game state, small enough to log every decision (for hit-cause analysis)."""
    p, b = s.player, s.boss
    r = lambda v: round(float(v), 1)  # noqa: E731
    return {
        "p": [r(p.x), r(p.y), int(p.grounded), int(p.dashing), round(float(p.invuln), 2)],
        "b": [b.state.name, r(b.x), r(b.y), r(b.half_w), r(b.half_h), round(float(b.state_time), 2)],
        "hz": [[h.kind.name, r(h.x), r(h.y), r(h.half_w), r(h.half_h)] for h in s.hazards],
    }


class PacedGameEnv(CupheadGameEnv):
    """Lockstep, paced to real time, for recording.

    Lockstep renders one frame per tick (1/60 s of game time) but runs ~2.9x real time. Here each
    decision's ``frame_skip`` ticks are sent one at a time, at most one per 1/60 s of wall time, so
    a capture plays at normal speed with every game frame on screen. Dash and EX are one-tick
    presses, so they go on the first tick only, as in a single ``frames=frame_skip`` act; held
    buttons are held for all ticks. The agent sees exactly what it sees in lockstep evaluation.
    """

    TICK = 1.0 / 60.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, lockstep=True, **kwargs)
        self._next_tick = 0.0

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self._next_tick = time.perf_counter()
        return out

    def _step_raw(self, a: spec.Action, frames: int) -> dict:
        msg: dict = {}
        for k in range(frames):
            delay = self._next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self._next_tick = max(self._next_tick + self.TICK, time.perf_counter() - self.TICK)
            msg = self.client.step_lockstep(a if k == 0 else dataclasses.replace(a, dash=False, ex=False), 1)
            if not msg.get("in_fight") or msg.get("won") or msg.get("lost"):
                break
        return msg


def find_config(ckpt: Path) -> Path:
    for d in (ckpt.parent, ckpt.parent.parent):
        if (d / "config.yaml").exists():
            return d / "config.yaml"
    raise FileNotFoundError(f"no config.yaml next to or above {ckpt}; pass --config")


def make_env(cfg: ExperimentConfig, mode: str, character: str) -> CupheadGameEnv:
    # Only the keys the game env understands: env_kwargs may hold sim-only settings.
    kwargs = dict(
        frame_skip=cfg.frame_skip,
        reward_config=spec.RewardConfig(**cfg.reward),
        max_seconds=float(cfg.env_kwargs.get("max_seconds", spec.MAX_EPISODE_SECONDS)),
        time_scale=1.0,
        no_hit=False,
        character=character,
    )
    if mode == "paced":
        return PacedGameEnv(**kwargs)
    return CupheadGameEnv(lockstep=mode == "lockstep", **kwargs)


def play_episode(env: CupheadGameEnv, policy: AgentPolicy, seed: int, start_phase: int | None,
                 log_dir: Path | None, index: int, meta: dict, record: bool = False,
                 encoder: str = "h264_amf") -> tuple[EpisodeResult, float]:
    """One episode. With ``log_dir`` it writes the per-decision log there, and with ``record`` also the video."""
    rec = None
    log: list[dict] = []
    if record:
        from cuphead_rl.game.capture import WindowRecorder

        rec = WindowRecorder(log_dir / f"episode_{index:02d}.mp4", fps=60, crf=18, encoder=encoder).start()
    wall0 = time.time()
    try:
        options = None if start_phase is None else {"start_phase": start_phase}
        obs, info = env.reset(seed=seed, options=options)
        policy.reset()
        start_hp = info.get("player_hp", spec.PLAYER_MAX_HP)
        total, done, step = 0.0, False, 0
        while not done:
            action = policy(obs)
            entry = {"step": step, "t": round(float(info["time"]), 4), "wall": time.time(),
                     "action": [int(v) for v in np.asarray(action).ravel()]}
            obs, r, term, trunc, info = env.step(action)
            total += float(r)
            done = term or trunc
            if log_dir is not None:
                ov = policy.overlay()
                entry.update(value=ov.get("value"), probs=ov.get("probs"), reward=round(float(r), 4),
                             **{"return": round(total, 4)}, phase=int(info["phase"]),
                             boss_hp_frac=round(float(info["boss_hp_frac"]), 4), player_hp=int(info["player_hp"]),
                             state=compact_state(env.state))
                log.append(entry)
            step += 1
        if rec is not None and (isinstance(env, PacedGameEnv) or not env.lockstep):
            time.sleep(END_HOLD_SECONDS)
    finally:
        if rec is not None:
            rec.stop()
    wall = time.time() - wall0
    res = EpisodeResult(
        seed=seed,
        won=bool(info["won"]),
        phase=int(info["phase"]),
        boss_damage=1.0 - float(info["boss_hp_frac"]),
        hits_taken=int(start_hp - info["player_hp"]),
        time=float(info["time"]),
        reward=total,
        parries=int(info.get("parries", 0)),
    )
    if log_dir is not None:
        head = {"meta": {**meta, "episode": index, "seed": seed, "video": f"episode_{index:02d}.mp4" if rec else None,
                         "video_start_epoch": rec.start_epoch if rec else None, "fps": 60,
                         "result": dataclasses.asdict(res)}}
        with open(log_dir / f"episode_{index:02d}.jsonl", "w") as f:
            for line in [head, *log]:
                f.write(json.dumps(line, separators=(",", ":")) + "\n")
    return res, wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--config", type=Path, help="defaults to config.yaml next to / above the checkpoint")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--start-phase", type=int, choices=(0, 1, 2))
    ap.add_argument("--record", type=Path, metavar="DIR")
    ap.add_argument("--log", type=Path, metavar="DIR", help="per-decision logs without a video")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--character", default="mugman", choices=("mugman", "cuphead"))
    ap.add_argument("--mode", choices=("lockstep", "paced", "free"), help="default: lockstep, or paced with --record")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--encoder", default="h264_amf",
                    help="capture encoder: h264_amf (default), h264_nvenc, h264_qsv or libx264. libx264 costs enough "
                         "CPU to slow free-run decisions to ~9 ticks instead of frame_skip")
    ap.add_argument("--out", type=Path, help="stats JSON (default DIR/stats.json with --record)")
    args = ap.parse_args()

    cfg_path = args.config or find_config(args.checkpoint)
    cfg = ExperimentConfig.load(cfg_path)
    mode = args.mode or ("paced" if args.record else "lockstep")
    policy = AgentPolicy(load_model(args.checkpoint), frame_stack=cfg.frame_stack,
                         deterministic=not args.stochastic, introspect=bool(args.record or args.log))
    log_dir = args.record or args.log
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoint": str(args.checkpoint), "config": str(cfg_path), "mode": mode, "character": args.character,
            "deterministic": not args.stochastic, "frame_skip": cfg.frame_skip, "frame_stack": cfg.frame_stack,
            "start_phase": args.start_phase, "encoder": args.encoder if args.record else None}

    env = make_env(cfg, mode, character=args.character)
    results, walls = [], []
    try:
        for i in range(args.episodes):
            res, wall = play_episode(env, policy, args.seed + i, args.start_phase, log_dir, i, meta,
                                     record=bool(args.record), encoder=args.encoder)
            results.append(res)
            walls.append(wall)
            print(f"episode {i}: won={res.won} phase={res.phase} boss_damage={res.boss_damage:.3f} "
                  f"hits={res.hits_taken} time={res.time:.1f}s reward={res.reward:.1f} wall={wall:.1f}s", flush=True)
    finally:
        env.close()

    stats = summarize(results)
    stats["per_episode"] = [{**dataclasses.asdict(r), "wall_seconds": round(w, 2)} for r, w in zip(results, walls)]
    stats["mean_survival_time"] = float(np.mean([r.time for r in results]))
    stats["settings"] = meta
    text = json.dumps(stats, indent=2)
    print(text)
    out = args.out or (log_dir / "stats.json" if log_dir else None)
    if out:
        out.write_text(text)
        print(f"stats -> {out}")


if __name__ == "__main__":
    main()
