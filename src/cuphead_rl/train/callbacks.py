"""Training callbacks: periodic evaluation, checkpointing, "interesting moment" videos and
the adaptive phase curriculum."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from cuphead_rl.train.agent import AgentPolicy
from cuphead_rl.train.config import ExperimentConfig
from cuphead_rl.train.evaluate import EpisodeResult, evaluate, record_video, summarize

DAMAGE_MILESTONES = (0.25, 0.5, 0.75)
WIN_RATE_MILESTONES = (0.25, 0.5, 0.9)


class EvalVideoCallback(BaseCallback):
    """Every ``cfg.eval_every`` env steps: evaluate, checkpoint, maybe record a video.

    A video is recorded on the first eval, every ``video_every_evals`` evals, and whenever
    a milestone is crossed for the first time (reaching phase 2/3, boss damage thresholds,
    first win, win-rate thresholds). Each video gets a line in ``videos/index.jsonl``
    stating why it was recorded, which is what the retrospective/YouTube edit is built from.
    """

    def __init__(self, cfg: ExperimentConfig, curriculum: "AdaptiveCurriculum | None" = None):
        super().__init__()
        self.cfg = cfg
        self.curriculum = curriculum
        self.run_dir = cfg.run_dir
        self.video_dir = self.run_dir / "videos"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.next_eval = 0
        self.n_evals = 0
        self.best_score = -float("inf")
        self.reached: set[str] = set()
        self.t0 = time.time()
        self._restored = False

    def restore(self, rows: list[dict]) -> None:
        """Continue a run from its ``metrics.jsonl`` rows (eval count, best score, milestones
        already reached, wall clock) so a resumed run doesn't re-record "first_*" videos."""
        if not rows:
            return
        for r in rows:
            self._milestones(r)
            self.best_score = max(self.best_score, r["win_rate"] * 10 + r["boss_damage"])
        self.n_evals = len(rows)
        self.next_eval = rows[-1]["steps"] + self.cfg.eval_every
        self.t0 = time.time() - rows[-1]["wall_time"]
        self._restored = True

    def _on_training_start(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if not self._restored:
            self.next_eval = self.num_timesteps  # evaluate the untrained policy too

    def _on_rollout_end(self) -> None:
        # Training-time outcome stats from Monitor's info_keywords (stochastic policy, all phases).
        buf = [ep for ep in self.model.ep_info_buffer if "won" in ep]
        if buf:
            self.logger.record("rollout/win_rate", float(np.mean([ep["won"] for ep in buf])))
            self.logger.record("rollout/boss_damage", float(np.mean([1.0 - ep["boss_hp_frac"] for ep in buf])))
            self.logger.record("rollout/phase", float(np.mean([ep["phase"] for ep in buf])))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval = self.num_timesteps + self.cfg.eval_every
            self._evaluate()
        return True

    def _milestones(self, stats: dict) -> list[str]:
        new = []
        cands = []
        if stats["max_phase"] >= 1:
            cands.append("first_phase2")
        if stats["max_phase"] >= 2:
            cands.append("first_phase3")
        cands += [f"damage_{int(m * 100)}" for m in DAMAGE_MILESTONES if stats["boss_damage"] >= m]
        if stats["win_rate"] > 0:
            cands.append("first_win")
        cands += [f"winrate_{int(m * 100)}" for m in WIN_RATE_MILESTONES if stats["win_rate"] >= m]
        for c in cands:
            if c not in self.reached:
                self.reached.add(c)
                new.append(c)
        return new

    def _rollout_stats(self) -> dict:
        """Real-game backend: the game serves one client, so no separate eval env can connect.
        Report the most recent *training* episodes instead (stochastic policy); proper
        evaluation runs between fine-tuning chunks with scripts/play_game.py."""
        # Monitor sits inside the reward-scaling wrapper, so ep["r"] is already the unscaled return.
        eps = [ep for ep in self.model.ep_info_buffer if "won" in ep][-self.cfg.eval_episodes:]
        if not eps:
            return {}
        results = [EpisodeResult(won=bool(ep["won"]), phase=int(ep["phase"]), boss_damage=1.0 - float(ep["boss_hp_frac"]),
                                 hits_taken=int(3 - ep["player_hp"]), time=float(ep["time"]),
                                 reward=float(ep["r"]), parries=int(ep["parries"]))
                   for ep in eps]
        return {**summarize(results), "source": "training_episodes"}

    def _evaluate(self) -> None:
        steps = self.num_timesteps
        if self.cfg.backend == "game":
            stats = self._rollout_stats()
            if not stats:
                return  # no finished episode yet; try again at the next eval point
            stats.update(steps=steps, wall_time=round(time.time() - self.t0, 1), eval=self.n_evals)
            self.model.save(self.ckpt_dir / f"step_{steps:010d}.zip")
            score = stats["win_rate"] * 10 + stats["boss_damage"]
            if score > self.best_score:
                self.best_score = score
                self.model.save(self.run_dir / "best.zip")
            with open(self.run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")
            if self.verbose:
                print(f"[game {self.n_evals}] steps={steps:,} win={stats['win_rate']:.2f} "
                      f"dmg={stats['boss_damage']:.2f} phase={stats['mean_phase']:.2f} (last {stats['episodes']} training eps)")
            self.n_evals += 1
            return
        policy = AgentPolicy(self.model, frame_stack=self.cfg.frame_stack, deterministic=True, introspect=False)
        # Headline stats are always the full fight on the same episode count, so curves from
        # curriculum and non-curriculum experiments are comparable.
        stats = evaluate(policy, self.cfg, self.cfg.eval_episodes, start_phase=0)
        start = self.curriculum.eval_start_phase() if self.curriculum else None
        if start not in (None, 0):
            cur = evaluate(policy, self.cfg, max(self.cfg.eval_episodes // 2, 4), start_phase=start)
            stats["curriculum"] = {"start_phase": start, **cur}
        stats.update(steps=steps, wall_time=round(time.time() - self.t0, 1), eval=self.n_evals)

        for k in ("win_rate", "boss_damage", "mean_phase", "hits_taken", "reward"):
            self.logger.record(f"eval/{k}", stats[k])
        if self.curriculum:
            self.curriculum.update(stats, self.training_env)
            self.logger.record("curriculum/stage", self.curriculum.stage)
            stats["curriculum_probs"] = self.curriculum.probs
            stats["curriculum_stage"] = self.curriculum.stage  # after this eval's promotion

        ckpt = self.ckpt_dir / f"step_{steps:010d}.zip"
        self.model.save(ckpt)
        score = stats["win_rate"] * 10 + stats["boss_damage"]
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.run_dir / "best.zip")

        tags = self._milestones(stats)
        if self.n_evals == 0 and steps > self.model.n_envs:
            # Fine-tuning from a trained model (init_from): milestones it already had aren't news.
            tags = ["start"]
        elif self.n_evals == 0:
            tags.insert(0, "untrained")
        elif self.n_evals % self.cfg.video_every_evals == 0:
            tags.append("periodic")
        if tags:
            name = f"{steps:010d}_{'+'.join(tags)}.mp4"
            title = f"{self.cfg.name} | {steps / 1e6:.2f}M steps | {', '.join(tags)}"
            # Replay the eval episode that shows the milestone (a win if there was one, else the
            # episode that got furthest), not an arbitrary seed.
            seed = stats["win_seed"] if stats["win_seed"] is not None else stats["best_seed"]
            vpolicy = AgentPolicy(self.model, frame_stack=self.cfg.frame_stack, deterministic=True)
            vstats = record_video(vpolicy, self.cfg, self.video_dir / name, seed=seed, title=title, start_phase=0)
            with open(self.video_dir / "index.jsonl", "a") as f:
                f.write(json.dumps({"file": name, "steps": steps, "tags": tags, "eval": stats,
                                    "episode": vstats}, default=str) + "\n")

        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(stats, default=str) + "\n")
        if self.verbose:
            print(f"[eval {self.n_evals}] steps={steps:,} win={stats['win_rate']:.2f} "
                  f"dmg={stats['boss_damage']:.2f} phase={stats['mean_phase']:.2f} tags={tags}")
        self.n_evals += 1


class AdaptiveCurriculum:
    """Reverse ("backward") phase curriculum.

    Stage 2: practise mostly the tombstone (phase 3). Once the agent clears it reliably,
    stage 1: mostly phase 2 onward. Then stage 0: mostly the full fight. Learning the end
    of the fight first means the big terminal reward is reachable early in training.
    """

    def __init__(self, promote_at: float, focus: float = 0.7, start_stage: int = 2):
        assert 0 <= start_stage <= 2
        self.promote_at = promote_at
        self.focus = focus
        self.stage = start_stage
        self.probs = self._probs()

    def _probs(self) -> list[float]:
        later = [k for k in range(3) if k > self.stage]
        p = [0.0, 0.0, 0.0]
        p[self.stage] = self.focus if later else 1.0
        for k in later:
            p[k] = (1.0 - self.focus) / len(later)
        return p

    def eval_start_phase(self) -> int:
        return self.stage

    def update(self, stats: dict, venv) -> None:
        rate = stats.get("curriculum", stats)["win_rate"]
        if self.stage > 0 and rate >= self.promote_at:
            self.stage -= 1
            self.probs = self._probs()
            venv.env_method("set_probs", self.probs)
