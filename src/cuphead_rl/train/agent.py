"""A stateful wrapper that runs a trained model one decision at a time.

Used for evaluation and video recording, where the env is a plain (non-vectorized)
Gymnasium env. It handles frame stacking and LSTM state itself, and keeps the last value
estimate and action probabilities around so the renderer can show what the agent "thinks".
"""

from __future__ import annotations

import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from cuphead_rl import spec

HEAD_NAMES = ("move", "jump", "dash", "shoot", "ex", "aim")
assert len(HEAD_NAMES) == len(spec.ACTION_NVEC)


def load_model(path: str | Path):
    """Load a PPO or RecurrentPPO checkpoint, picking the class from the saved policy."""
    with zipfile.ZipFile(path) as zf:
        recurrent = "Recurrent" in zf.read("data").decode("utf-8", "ignore")
    if recurrent:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(path, device="cpu")
    return PPO.load(path, device="cpu")


class AgentPolicy:
    def __init__(self, model, frame_stack: int = 1, deterministic: bool = True, introspect: bool = True):
        self.model = model
        self.introspect = introspect  # value + probs for the video overlay; costs a 2nd forward pass
        self.frame_stack = frame_stack
        self.deterministic = deterministic
        self.recurrent = model.__class__.__name__ == "RecurrentPPO"
        self.reset()

    def reset(self) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._lstm_state = None
        self._episode_start = np.ones((1,), dtype=bool)
        self.last_value: float = 0.0
        self.last_probs: list[np.ndarray] = []
        self.last_action: np.ndarray | None = None

    def _stack(self, obs: np.ndarray) -> np.ndarray:
        if self.frame_stack == 1:
            return obs
        if not self._frames:
            for _ in range(self.frame_stack - 1):
                self._frames.append(np.zeros_like(obs))
        self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=-1)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = self._stack(np.asarray(obs, dtype=np.float32))
        if self.recurrent:
            action, self._lstm_state = self.model.predict(
                x, state=self._lstm_state, episode_start=self._episode_start, deterministic=self.deterministic
            )
            self._episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = self.model.predict(x, deterministic=self.deterministic)
            if self.introspect:
                self._introspect(x)
        self.last_action = np.asarray(action)
        return self.last_action

    @torch.no_grad()
    def _introspect(self, x: np.ndarray) -> None:
        policy = self.model.policy
        t, _ = policy.obs_to_tensor(x)
        self.last_value = float(policy.predict_values(t).item())
        dist = policy.get_distribution(t).distribution
        self.last_probs = [d.probs.squeeze(0).cpu().numpy() for d in dist]

    def overlay(self, title: str = "") -> dict:
        """Text/values for the renderer's side panel."""
        out: dict = {"title": title, "value": round(self.last_value, 2)}
        if self.last_action is not None:
            a = spec.decode_action(self.last_action)
            out["action"] = " ".join(
                s for s in (
                    {-1: "LEFT", 0: "", 1: "RIGHT"}[a.move],
                    "JUMP" if a.jump else "",
                    "DASH" if a.dash else "",
                    "SHOOT" if a.shoot else "",
                    "EX" if a.ex else "",
                    {-1: "DOWN", 0: "", 1: "UP"}[a.aim],
                ) if s
            ) or "idle"
        if self.last_probs:
            out["probs"] = {
                name: [round(float(p), 2) for p in probs]
                for name, probs in zip(HEAD_NAMES, self.last_probs)
            }
        return out
