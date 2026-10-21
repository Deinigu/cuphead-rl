"""Gymnasium environment backed by the real Cuphead via the CupheadRLBridge plugin.

Mirrors the simulator env: same observation (spec.encode_observation), action
(spec.ACTION_NVEC / spec.decode_action) and reward (spec.compute_reward). The game must be
running with the plugin installed; ``reset`` loads (or reloads) the Goopy Le Grande fight.

``reset(options={"start_phase": k})`` starts in phase k (0 small, 1 big, 2 tombstone) by dealing
scripted damage through the bridge right after the reload, then waiting (player shielded by the
debug no-hit flag) until the boss has actually transformed. Scripted damage is excluded from
``info["damage_dealt"]``, and ``GameState.time`` counts from the end of ``reset`` (spec: seconds
since the episode started), so phase starts look like fresh episodes.

There is one game, hence at most one CupheadGameEnv per process (and the plugin refuses a second
connection): use n_envs == 1 for the game backend.

Player one is Mugman by default (``character="cuphead"`` to switch). Mugman's motor, hurtbox,
weapons and stats are the same code and values as Cuphead's (only sprites and animations
differ), so the sim and the spec contract are character-independent.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np

from cuphead_rl import spec
from cuphead_rl.game.client import DEFAULT_HOST, DEFAULT_PORT, BridgeClient, BridgeError, to_game_state

# Damage that pushes the boss just past each phase trigger (1200 HP, triggers at 76 % / 31 %).
_PHASE_DAMAGE = {1: 290.0, 2: 545.0}
_PHASE_ENTITY = {1: "big", 2: "tomb"}
# The contract loadout (a fresh save's), forced by the plugin on every reset.
CHARACTERS = ("mugman", "cuphead")
EXPECTED_LOADOUT = {"primary": "level_weapon_peashot", "secondary": "None", "charm": "None", "super": "None"}
# Steps without in-game time advancing before we assume the game is paused / showing a prompt.
PAUSE_STEPS = 50


class CupheadGameEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": spec.GAME_FPS}
    _live = 0  # open instances in this process

    def __init__(
        self,
        frame_skip: int = 4,
        reward_config: spec.RewardConfig = spec.RewardConfig(),
        max_seconds: float = spec.MAX_EPISODE_SECONDS,
        lockstep: bool = True,
        time_scale: float = 1.0,
        no_hit: bool = False,
        character: str = "mugman",
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_wait: float = 60.0,
    ):
        super().__init__()
        if max_seconds > spec.MAX_EPISODE_SECONDS:
            raise ValueError(f"max_seconds={max_seconds} exceeds spec.MAX_EPISODE_SECONDS={spec.MAX_EPISODE_SECONDS}")
        if CupheadGameEnv._live:
            raise RuntimeError(
                "only one CupheadGameEnv can exist per process: there is a single game. "
                "Use n_envs=1 for the 'game' backend."
            )
        self.frame_skip = int(frame_skip)
        self.reward_config = reward_config
        self.max_seconds = float(max_seconds)
        self.lockstep = lockstep
        self.time_scale = time_scale
        self.no_hit = no_hit
        if character not in CHARACTERS:
            raise ValueError(f"character must be one of {CHARACTERS}, got {character!r}")
        self.character = character
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(spec.OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(spec.ACTION_NVEC)
        self.client = BridgeClient(host, port)
        self.client.connect(wait=connect_wait)  # raises unless the plugin's save guard is complete
        CupheadGameEnv._live += 1
        self._open = True
        self._configure()
        self._state: spec.GameState | None = None
        self._dmg0 = 0.0
        self._t0 = 0.0
        self._stalled = 0
        self.last_msg: dict[str, Any] | None = None

    def _configure(self) -> None:
        self.client.configure(
            lockstep=self.lockstep,
            frames=self.frame_skip,
            time_scale=self.time_scale,
            stream_every=0 if self.lockstep else 1,
            control=True,
            debug=False,
        )
        self.client.invincible(self.no_hit)  # debug no-hit only when explicitly requested

    def _step_raw(self, a: spec.Action, frames: int) -> dict[str, Any]:
        if self.lockstep:
            return self.client.step_lockstep(a, frames)
        return self.client.step_free(a, frames)

    def _game_state(self, msg: dict[str, Any]) -> spec.GameState:
        s = to_game_state(msg)
        return dataclasses.replace(s, time=s.time - self._t0)

    def _obs(self, s: spec.GameState) -> np.ndarray:
        return spec.encode_observation(s, self.max_seconds)

    def _info(self, s: spec.GameState) -> dict[str, Any]:
        return {
            "won": s.won,
            "phase": int(s.boss.phase),
            "boss_hp_frac": s.boss.hp / max(s.boss.max_hp, 1e-6),
            "player_hp": s.player.hp,
            "damage_dealt": s.damage_dealt - self._dmg0,
            "time": s.time,
            "parries": s.parries,
        }  # the full bridge message (frame counters etc.) is env.last_msg

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        bridge_seed = int(self.np_random.integers(0, 2**31 - 1))
        msg = self.client.reset(seed=bridge_seed, character=self.character,
                                realtime_intro=bool((options or {}).get("realtime_intro", False)))
        if msg.get("character") != self.character:
            raise BridgeError(f"bridge plays {msg.get('character')!r}, expected {self.character!r} (outdated plugin?)")
        loadout = (self.client.last_reset or {}).get("loadout")
        if loadout != EXPECTED_LOADOUT:
            raise BridgeError(f"unexpected loadout {loadout}, expected {EXPECTED_LOADOUT} (outdated plugin?)")
        start_phase = int((options or {}).get("start_phase", 0))
        if start_phase > 0:
            msg = self._skip_to_phase(min(start_phase, 2))
        self.last_msg = msg
        self._t0 = 0.0
        raw = to_game_state(msg)
        self._t0 = raw.time  # spec: time counts from the start of the episode
        self._dmg0 = raw.damage_dealt
        self._state = self._game_state(msg)
        self._stalled = 0
        return self._obs(self._state), self._info(self._state)

    def _skip_to_phase(self, k: int) -> dict[str, Any]:
        self.client.invincible(True)
        try:
            msg = self.client.last_state
            for p in range(1, k + 1):
                self.client.damage(_PHASE_DAMAGE[p])
                for _ in range(2000):
                    msg = self._step_raw(spec.Action(), self.frame_skip)
                    b = msg["boss"]
                    if b["entity"] == _PHASE_ENTITY[p] and b["state"] not in (spec.BossMove.TRANSFORM, spec.BossMove.INTRO):
                        break
                else:
                    raise RuntimeError(f"boss did not reach phase {p}")
        finally:
            self.client.invincible(self.no_hit)
        return self.client.last_state if self.client.last_state else msg

    def step(self, action):
        a = spec.decode_action(action)
        msg = self._step_raw(a, self.frame_skip)
        prev = self._state
        assert prev is not None, "call reset() first"
        if not msg.get("in_fight"):  # the game left the fight (win screen loaded)
            return self._obs(prev), 0.0, True, False, self._info(prev) | {"left_fight": True}
        self.last_msg = msg
        curr = self._game_state(msg)
        self._state = curr
        if curr.time <= prev.time and not (curr.won or curr.lost):
            self._stalled += 1
            if self._stalled >= PAUSE_STEPS:
                raise RuntimeError(
                    f"in-game time has not advanced for {PAUSE_STEPS} steps: is the game paused or showing a prompt?"
                )
        else:
            self._stalled = 0
        reward = spec.compute_reward(prev, curr, self.reward_config)
        terminated = curr.won or curr.lost
        truncated = (not terminated) and curr.time >= self.max_seconds
        return self._obs(curr), reward, terminated, truncated, self._info(curr)

    @property
    def state(self) -> spec.GameState | None:
        return self._state

    def close(self):
        if not getattr(self, "_open", False):
            return
        self._open = False
        CupheadGameEnv._live -= 1
        try:
            self.client.invincible(False)
            self.client.configure(lockstep=False, time_scale=1.0, stream_every=0)
        except Exception:  # noqa: BLE001 - best effort while shutting down
            pass
        self.client.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
