"""Experiment configuration, loaded from YAML files in ``configs/``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 1024  # per env, per rollout
    batch_size: int = 2048
    n_epochs: int = 6
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    lstm_hidden_size: int = 128  # recurrent_ppo only
    lr_schedule: str = "constant"  # "constant" | "linear"


@dataclass
class CurriculumConfig:
    """Phase curriculum: each episode starts in phase k with probability ``probs[k]``.

    With ``adaptive`` set, probability mass moves from later phases back to phase 0
    as the agent's eval win-rate from that phase passes ``promote_at``.
    """

    enabled: bool = False
    probs: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    adaptive: bool = False
    promote_at: float = 0.6
    init_stage: int = 2  # adaptive: start phase to begin from (set when resuming a curriculum run)


@dataclass
class ExperimentConfig:
    name: str = "ppo_baseline"
    algo: str = "ppo"  # "ppo" | "recurrent_ppo"
    backend: str = "sim"  # "sim" | "game"
    seed: int = 0
    n_envs: int = 16
    torch_threads: int = 4  # measured best on the i5-13400F (16 envs x 4 threads: ~6.2k steps/s)
    total_timesteps: int = 20_000_000
    frame_skip: int = 4
    frame_stack: int = 1
    randomize: bool = False
    randomize_scale: float = 0.15
    reward: dict[str, float] = field(default_factory=dict)  # overrides for spec.RewardConfig
    # Training-only reward multiplier. Returns reach +150; unscaled, the value loss dominates the
    # shared grad-norm clip and shrinks policy updates ~4x. Eval metrics use raw rewards.
    reward_scale: float = 0.1
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval_every: int = 250_000  # env steps between evaluations
    eval_episodes: int = 20
    video_every_evals: int = 4  # also record when something "interesting" happens
    init_from: str | None = None  # checkpoint to warm-start from (fine-tuning)
    out_dir: str = "runs"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        ppo = PPOConfig(**d.pop("ppo", {}))
        cur = CurriculumConfig(**d.pop("curriculum", {}))
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(ppo=ppo, curriculum=cur, **d)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "ExperimentConfig":
        d = yaml.safe_load(Path(path).read_text()) or {}
        for ov in overrides or []:
            key, _, raw = ov.partition("=")
            _set_dotted(d, key, yaml.safe_load(raw))
        return cls.from_dict(d)


def _set_dotted(d: dict, key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value
