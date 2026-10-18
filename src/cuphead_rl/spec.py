"""The shared contract between the simulator, the real-game bridge and the agent.

Both backends (``cuphead_rl.sim`` and ``cuphead_rl.game``) produce a ``GameState`` every
decision step. Everything the agent sees (``encode_observation``), everything it can do
(``decode_action``) and everything it is rewarded for (``compute_reward``) is defined
here, once. That is what makes sim-to-real transfer possible: a policy trained in the
simulator receives exactly the same vector layout when it is plugged into the game.

Coordinate system (shared by both backends):
    * x in game units, arena centre = 0, positive to the right. Arena spans
      [-ARENA_HALF_WIDTH, ARENA_HALF_WIDTH].
    * y in game units, ground = 0, positive up.
    * velocities in units / second.
The real-game bridge is responsible for mapping Unity world coordinates into this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

SPEC_VERSION = 2

ARENA_HALF_WIDTH = 640.0
ARENA_HEIGHT = 720.0
VEL_SCALE = 1500.0  # velocities are divided by this before entering the observation
PLAYER_MAX_HP = 3
SUPER_METER_MAX = 5.0  # five cards
MAX_HAZARDS = 8
GAME_FPS = 60


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------
class Horizontal(IntEnum):
    LEFT = 0
    NONE = 1
    RIGHT = 2


class Vertical(IntEnum):
    DOWN = 0  # duck when grounded (smaller hurtbox, low shots)
    NONE = 1
    UP = 2  # aim up: straight up when standing still, diagonal when moving


# MultiDiscrete layout:
#   [horizontal(3), jump(held), dash(pressed), shoot(held), ex(pressed), vertical(3)]
# The two 3-way heads are the controller stick; "vertical" was added in spec v2 because the
# phase-3 tombstone sits above standing shot height (measured live: standing peashots deal
# 0 damage to it), so aiming up is required to win, exactly as for a human player.
ACTION_NVEC = (3, 2, 2, 2, 2, 3)


@dataclass(frozen=True)
class Action:
    move: int = 0  # -1, 0, +1
    jump: bool = False  # button held this step (press edge = held now, not held before)
    dash: bool = False  # button pressed this step
    shoot: bool = False  # button held this step
    ex: bool = False  # button pressed this step
    aim: int = 0  # -1 down/duck, 0 neutral, +1 up


def decode_action(a) -> Action:
    a = np.asarray(a).astype(int).ravel()
    return Action(
        move=int(a[0]) - 1,
        jump=bool(a[1]),
        dash=bool(a[2]),
        shoot=bool(a[3]),
        ex=bool(a[4]),
        aim=int(a[5]) - 1,
    )


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------
class BossPhase(IntEnum):
    """The boss *entity* currently on screen, not the HP bracket.

    The phase changes when the transformation starts (small → big on landing), resp. when
    the tombstone appears, NOT the instant HP crosses the 76 % / 31 % threshold. (The game's
    internal ``CurrentState`` flips at the threshold, up to a few seconds earlier; the bridge
    must report the visible entity.)
    """

    SMALL = 0  # phase 1: little Goopy, hops and punches
    BIG = 1  # phase 2: big Goopy with the moustache
    TOMB = 2  # phase 3: tombstone that slides over you and slams


class BossMove(IntEnum):
    INTRO = 0
    IDLE = 1
    JUMP = 2
    PUNCH_WINDUP = 3
    PUNCH = 4
    TRANSFORM = 5  # phase transition animation (boss usually invulnerable)
    TOMB_MOVE = 6
    TOMB_SLAM_WINDUP = 7
    TOMB_SLAM = 8
    DEAD = 9


class HazardKind(IntEnum):
    """Hazard semantics (both backends MUST agree, or sim-to-real transfer breaks):

    A hazard is a collider that would damage the player *this tick* if overlapping, plus
    parryable objects. Something that is present but currently harmless is not a hazard
    (its position is still visible through ``BossInfo``).
    """

    BODY = 0  # contact damage from the boss body (incl. while dying, as in the game)
    FIST = 1  # every punch collider: small Goopy's lunge circle, big Goopy's thrown circle and arm box
    TOMB = 2  # tombstone ONLY while its slam deals damage (not while it patrols)
    PROJECTILE = 3  # damaging projectiles (none in the Goopy fight on Regular)
    OTHER = 4  # harmless parryable objects, e.g. the pink question marks (parryable=True)


@dataclass
class Hazard:
    kind: HazardKind
    x: float
    y: float
    half_w: float
    half_h: float
    vx: float = 0.0
    vy: float = 0.0
    parryable: bool = False


@dataclass
class PlayerState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    grounded: bool = True
    dashing: bool = False
    can_dash: bool = True  # air-dash available
    facing: int = 1  # -1 left, +1 right
    hp: int = PLAYER_MAX_HP
    invuln: float = 0.0  # seconds of remaining i-frames
    super_meter: float = 0.0  # 0..SUPER_METER_MAX (cards)


@dataclass
class BossInfo:
    phase: BossPhase
    state: BossMove
    x: float
    y: float  # centre of the hitbox
    half_w: float
    half_h: float
    vx: float = 0.0
    vy: float = 0.0
    state_time: float = 0.0  # seconds spent in current state
    facing: int = -1
    hp: float = 1.0  # remaining HP in absolute units
    max_hp: float = 1.0  # total HP over all phases
    phase_hp_frac: float = 1.0  # remaining fraction of the current phase's HP
    vulnerable: bool = True  # takes damage now (Goopy IS hittable during his transformation)


@dataclass
class GameState:
    player: PlayerState
    boss: BossInfo
    hazards: list[Hazard] = field(default_factory=list)
    time: float = 0.0  # seconds since the EPISODE started (also with start_phase)
    damage_dealt: float = 0.0  # cumulative damage dealt to the boss
    won: bool = False
    lost: bool = False
    parries: int = 0  # cumulative successful parries


# --------------------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------------------
PLAYER_FEATURES = 12
BOSS_FEATURES = 3 + len(BossMove) + 12
HAZARD_FEATURES = 8 + len(HazardKind)
GLOBAL_FEATURES = 1
OBS_SIZE = PLAYER_FEATURES + BOSS_FEATURES + MAX_HAZARDS * HAZARD_FEATURES + GLOBAL_FEATURES

MAX_EPISODE_SECONDS = 240.0


_BOSS_BASE = PLAYER_FEATURES
_HAZ_BASE = PLAYER_FEATURES + BOSS_FEATURES
_N_MOVES = len(BossMove)


def encode_observation(s: GameState, max_seconds: float = MAX_EPISODE_SECONDS) -> np.ndarray:
    """Flatten a GameState into a fixed-size float32 vector, roughly in [-1, 1].

    Boss and hazard positions are given relative to the player, which makes the policy
    translation-invariant and is the main thing a dodging policy needs. ``max_seconds``
    must be the env's truncation time so the time feature reaches 1 exactly at the limit.

    This runs once per env step, millions of times, so it builds one flat Python list,
    converts once, and writes the one-hot entries in place.
    """
    p, b = s.player, s.boss
    W, H, V = ARENA_HALF_WIDTH, ARENA_HEIGHT, VEL_SCALE
    px, py = p.x, p.y
    vals = [
        px / W, py / H, p.vx / V, p.vy / V,
        float(p.grounded), float(p.dashing), float(p.can_dash), float(p.facing),
        p.hp / PLAYER_MAX_HP, min(p.invuln, 2.0) / 2.0, p.super_meter / SUPER_METER_MAX,
        float(p.super_meter >= 1.0),
    ]
    vals += [0.0] * (3 + _N_MOVES)  # phase + move one-hots, set below
    vals += [
        (b.x - px) / W, (b.y - py) / H, b.x / W, b.half_w / W, b.half_h / H, b.vx / V, b.vy / V,
        min(b.state_time, 4.0) / 4.0, float(b.facing), b.hp / max(b.max_hp, 1e-6), b.phase_hp_frac,
        float(b.vulnerable),
    ]
    hz = s.hazards
    if len(hz) > 1:
        hz = sorted(hz, key=lambda h: (h.x - px) ** 2 + (h.y - py) ** 2)
    hz = hz[:MAX_HAZARDS]
    for h in hz:
        vals += [1.0, (h.x - px) / W, (h.y - py) / H, h.half_w / W, h.half_h / H, h.vx / V, h.vy / V,
                 float(h.parryable), 0.0, 0.0, 0.0, 0.0, 0.0]
    vals += [0.0] * ((MAX_HAZARDS - len(hz)) * HAZARD_FEATURES)
    vals.append(min(s.time, max_seconds) / max_seconds)

    out = np.array(vals, dtype=np.float32)
    out[_BOSS_BASE + int(b.phase)] = 1.0
    out[_BOSS_BASE + 3 + int(b.state)] = 1.0
    for i, h in enumerate(hz):
        out[_HAZ_BASE + i * HAZARD_FEATURES + 8 + int(h.kind)] = 1.0
    np.clip(out, -5.0, 5.0, out=out)
    return out


# --------------------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardConfig:
    damage: float = 1.0  # per 1% of total boss HP removed
    hit_taken: float = -3.0  # per HP lost
    win: float = 50.0
    loss: float = -10.0
    time: float = -0.001  # per step; mild pressure to finish
    parry: float = 0.5
    phase_clear: float = 5.0


def compute_reward(prev: GameState, curr: GameState, cfg: RewardConfig = RewardConfig()) -> float:
    dmg = max(0.0, curr.damage_dealt - prev.damage_dealt)
    r = cfg.damage * 100.0 * dmg / max(curr.boss.max_hp, 1e-6)
    r += cfg.hit_taken * max(0, prev.player.hp - curr.player.hp)
    r += cfg.parry * max(0, curr.parries - prev.parries)
    if curr.boss.phase > prev.boss.phase:
        r += cfg.phase_clear
    if curr.won and not prev.won:
        r += cfg.win
    if curr.lost and not prev.lost:
        r += cfg.loss
    r += cfg.time
    return float(r)
