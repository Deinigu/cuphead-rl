"""Gymnasium environment wrapping the Goopy simulator.

Observation and reward come exclusively from ``cuphead_rl.spec`` so the same policy can
later be pointed at the real-game bridge.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np

from cuphead_rl import spec
from cuphead_rl.sim.config import SimConfig
from cuphead_rl.sim.player import FrameInput
from cuphead_rl.sim.world import World


def _decode(action) -> spec.Action:
    """``spec.decode_action`` with a fast path for the common int-array / list case."""
    if isinstance(action, np.ndarray) and action.shape == (6,) and action.dtype.kind in "iu":
        m, j, d, sh, e, v = action.tolist()
        return spec.Action(move=m - 1, jump=bool(j), dash=bool(d), shoot=bool(sh), ex=bool(e), aim=v - 1)
    return spec.decode_action(action)


class GoopyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": spec.GAME_FPS}

    def __init__(
        self,
        config: SimConfig | None = None,
        frame_skip: int = 4,
        reward_config: spec.RewardConfig = spec.RewardConfig(),
        start_phase: int | None = None,
        randomize: bool = False,
        randomize_scale: float = 0.15,
        max_seconds: float = spec.MAX_EPISODE_SECONDS,
        render_mode: str | None = None,
        character: str = "mugman",
    ) -> None:
        super().__init__()
        assert render_mode in (None, "rgb_array")
        # The observation normalises time by spec.MAX_EPISODE_SECONDS; longer episodes
        # would push that feature past 1 (it is clipped), so don't allow them.
        assert max_seconds <= spec.MAX_EPISODE_SECONDS, "max_seconds exceeds spec.MAX_EPISODE_SECONDS"
        self.base_config = config or SimConfig()
        self.character = character  # cosmetic only: "mugman" or "cuphead"
        self.config = self.base_config
        self.frame_skip = int(frame_skip)
        self.reward_config = reward_config
        self.start_phase = start_phase
        self.randomize = randomize
        self.randomize_scale = randomize_scale
        self.max_seconds = max_seconds
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(spec.OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(spec.ACTION_NVEC)

        self.world: World | None = None
        self.state: spec.GameState | None = None
        self.episode_reward = 0.0
        self.last_action = spec.Action()
        self._jump_was_held = False
        # Called after every simulation frame (used by the video recorder for smooth 60 fps).
        self.frame_callback: Callable[[], None] | None = None
        # Extra values drawn in the renderer's side panel (value estimate, experiment name...).
        self.overlay: dict[str, Any] | None = None
        self._renderer = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if self.randomize:
            self.config = self.base_config.randomized(self.np_random, self.randomize_scale)
        start_phase = options.get("start_phase", self.start_phase)
        world_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.world = World(self.config, seed=world_seed, start_phase=start_phase)
        if self._renderer is not None:
            self._renderer.cfg = self.config  # randomized configs change per episode
        self.state = self.world.state()
        self.episode_reward = 0.0
        self.last_action = spec.Action()
        self._jump_was_held = False
        if self.frame_callback is not None:
            self.frame_callback()
        return spec.encode_observation(self.state, self.max_seconds), self._info()

    def step(self, action):
        assert self.world is not None, "call reset() first"
        a = _decode(action)
        self.last_action = a
        w = self.world
        first = FrameInput(a.move, a.jump, a.jump and not self._jump_was_held, a.dash, a.shoot, a.ex, a.aim)
        rest = FrameInput(a.move, a.jump, False, False, a.shoot, False, a.aim)
        self._jump_was_held = a.jump
        for i in range(self.frame_skip):
            w.step(first if i == 0 else rest)
            if self.frame_callback is not None:
                self.frame_callback()
            if w.won or w.lost:
                break
        prev, self.state = self.state, w.state()
        reward = spec.compute_reward(prev, self.state, self.reward_config)
        self.episode_reward += reward
        terminated = w.done
        truncated = (not terminated) and w.time >= self.max_seconds
        return spec.encode_observation(self.state, self.max_seconds), reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        s, w = self.state, self.world
        return {
            "phase": int(s.boss.phase),
            "boss_hp_frac": s.boss.hp / s.boss.max_hp,
            "player_hp": s.player.hp,
            "won": s.won,
            "damage_dealt": s.damage_dealt,
            "time": s.time,
            "parries": s.parries,
            "hits_taken": w.hits_taken,
        }

    # ------------------------------------------------------------------ rendering
    def render(self, overlay: dict[str, Any] | None = None) -> np.ndarray:
        """RGB frame (720, 1280, 3) of the current simulation frame."""
        from cuphead_rl.sim.render import Renderer  # lazy: pygame only when rendering

        if self._renderer is None:
            self._renderer = Renderer(self.config, character=self.character)
        assert self.world is not None
        return self._renderer.render(
            self.world, self.episode_reward, self.last_action,
            overlay if overlay is not None else self.overlay,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
