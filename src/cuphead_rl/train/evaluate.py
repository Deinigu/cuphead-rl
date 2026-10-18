"""Evaluation and video recording of trained (or scripted) policies.

CLI:
    python -m cuphead_rl.train.evaluate runs/<name>/best.zip --episodes 50 --video out.mp4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from cuphead_rl.train.agent import AgentPolicy, load_model
from cuphead_rl.train.config import ExperimentConfig
from cuphead_rl.train.envs import make_base_env


@dataclass
class EpisodeResult:
    won: bool
    phase: int
    boss_damage: float  # fraction of total boss HP removed, 0..1
    hits_taken: int
    time: float
    reward: float
    parries: int
    seed: int = -1


def run_episode(env, policy: Callable[[np.ndarray], np.ndarray], seed: int, start_phase: int | None = None) -> EpisodeResult:
    options = None if start_phase is None else {"start_phase": start_phase}
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "reset"):
        policy.reset()
    start_hp = info.get("player_hp", 3)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += float(r)
        done = term or trunc
    return EpisodeResult(
        seed=seed,
        won=bool(info["won"]),
        phase=int(info["phase"]),
        boss_damage=1.0 - float(info["boss_hp_frac"]),
        hits_taken=int(start_hp - info["player_hp"]),
        time=float(info["time"]),
        reward=total,
        parries=int(info.get("parries", 0)),
    )


def summarize(results: list[EpisodeResult]) -> dict:
    """Aggregate stats, plus the seeds of the most impressive episodes so a video can replay
    exactly the episode that earned a milestone (the policy is deterministic and the sim seeded)."""
    wins = [r for r in results if r.won]
    best = max(results, key=lambda r: (r.won, r.phase, r.boss_damage, -r.time))
    return {
        "best_seed": best.seed,
        "win_seed": wins[0].seed if wins else None,
        "episodes": len(results),
        "win_rate": len(wins) / max(len(results), 1),
        "boss_damage": float(np.mean([r.boss_damage for r in results])),
        "max_boss_damage": float(np.max([r.boss_damage for r in results])),
        "max_phase": int(max(r.phase for r in results)),
        "mean_phase": float(np.mean([r.phase for r in results])),
        "hits_taken": float(np.mean([r.hits_taken for r in results])),
        "reward": float(np.mean([r.reward for r in results])),
        "win_time": float(np.mean([r.time for r in wins])) if wins else None,
        "parries": float(np.mean([r.parries for r in results])),
    }


def evaluate(policy, cfg: ExperimentConfig, episodes: int, seed: int = 10_000, start_phase: int | None = None,
             nominal: bool = True) -> dict:
    """``nominal=False`` keeps the config's domain randomization (a robustness test)."""
    env = make_base_env(cfg, nominal=nominal)
    try:
        results = [run_episode(env, policy, seed + i, start_phase) for i in range(episodes)]
    finally:
        env.close()
    return summarize(results)


def record_video(policy: AgentPolicy | Callable, cfg: ExperimentConfig, path: str | Path, seed: int = 10_000,
                 title: str = "", start_phase: int | None = None) -> dict:
    """Render one evaluation episode to MP4 at 60 fps with the agent's 'thoughts' overlaid."""
    from cuphead_rl.sim import video

    env = make_base_env(cfg, render_mode="rgb_array", nominal=True)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        overlay_fn = (lambda: policy.overlay(title)) if hasattr(policy, "overlay") else (lambda: {"title": title})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return video.record_episode(env, policy, str(path), fps=60, overlay_fn=overlay_fn, seed=seed,
                                    options=None if start_phase is None else {"start_phase": start_phase})
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", help="defaults to config.yaml next to the checkpoint")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--start-phase", type=int)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--video", help="also record one episode to this MP4 path")
    ap.add_argument("--title", default="")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--randomized", action="store_true",
                    help="keep domain randomization on (with -o randomize=true -o randomize_scale=...)")
    ap.add_argument("-o", "--override", action="append", default=[], help="config override key=value")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    default_cfg = next((d / "config.yaml" for d in (ckpt.parent, ckpt.parent.parent) if (d / "config.yaml").exists()), None)
    cfg = ExperimentConfig.load(args.config or default_cfg, args.override)
    policy = AgentPolicy(load_model(ckpt), frame_stack=cfg.frame_stack, deterministic=not args.stochastic,
                         introspect=bool(args.video))
    if args.randomized and not cfg.randomize:
        raise SystemExit("--randomized needs randomize=true in the config or -o randomize=true")
    stats = evaluate(policy, cfg, args.episodes, seed=args.seed, start_phase=args.start_phase,
                     nominal=not args.randomized)
    print(json.dumps(stats, indent=2))
    if args.video:
        print(json.dumps(record_video(policy, cfg, args.video, seed=args.seed, title=args.title,
                                      start_phase=args.start_phase), indent=2, default=str))


if __name__ == "__main__":
    main()
