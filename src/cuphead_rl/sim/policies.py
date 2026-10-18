"""Baseline policies: ``policy(obs) -> MultiDiscrete action``.

The heuristic reads only the spec observation vector (not simulator internals), so it can
be run unchanged against the real-game bridge as a sanity check. It is a tiny planner:
every decision it predicts Goopy's next second (all plausible continuations of the current
move) and picks the first candidate player manoeuvre that never touches a hazard.
"""

from __future__ import annotations

import math

import numpy as np

from cuphead_rl import spec
from cuphead_rl.spec import BossMove, BossPhase, HazardKind
from cuphead_rl.sim import live_tracks
from cuphead_rl.sim.config import SimConfig

_rng = np.random.default_rng(0)  # seeded so scripts are reproducible; pass rng= to override
_SIM = SimConfig()  # nominal numbers the heuristic plans with

W, H, V = spec.ARENA_HALF_WIDTH, spec.ARENA_HEIGHT, spec.VEL_SCALE
_P = 0
_PHASE = spec.PLAYER_FEATURES
_STATE = _PHASE + 3
_BOSS = _STATE + len(BossMove)
_HAZ = _BOSS + 12

_DT = 1.0 / 30.0  # planner resolution
_STEPS = 45  # 1.5 s horizon (covers a whole punch windup)
_TAP = 4.0 / spec.GAME_FPS  # one decision step at frame_skip=4
_MARGIN = 10.0
_REACT = 9  # planner steps: wait while the threat is further away than this

Shape = tuple[float, float, float, float, bool]  # (cx, cy, half_w, half_h, round)


def random_policy(obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniform random buttons (the 'mashing' baseline)."""
    r = rng or _rng
    return np.array([r.integers(n) for n in spec.ACTION_NVEC], dtype=np.int64)


def _act(move: int, jump: bool, dash: bool, shoot: bool, ex: bool, aim: int = 0) -> np.ndarray:
    return np.array([move + 1, int(jump), int(dash), int(shoot), int(ex), aim + 1], dtype=np.int64)


# ------------------------------------------------------------------------------ boss model
class _BossSim:
    """Coarse forward model of one possible continuation of Goopy's current move.

    Punches and the tomb smash replay the same live hitbox tracks as the simulator; ``x``
    is the boss transform x (not the body centre, which moves during a punch).
    """

    def __init__(self, phase: BossPhase, state: BossMove, x: float, lift: float, vx: float, vy: float,
                 facing: int, tick: int, variant: str = "") -> None:
        self.phase, self.state = phase, state
        self.x, self.lift, self.vx, self.vy = x, lift, vx, vy
        self.facing, self.tick, self.variant = facing, tick, variant
        self.t = 0.0  # time on the ground (IDLE variants)
        self.slime = _SIM.big if phase == BossPhase.BIG else _SIM.small
        self.track = None
        if state in (BossMove.PUNCH_WINDUP, BossMove.PUNCH):
            self.track = live_tracks.BIG_PUNCH if phase == BossPhase.BIG else live_tracks.SMALL_PUNCH
        elif state in (BossMove.TOMB_SLAM_WINDUP, BossMove.TOMB_SLAM):
            self.track = live_tracks.TOMB_SMASH

    def shapes(self) -> list[Shape]:
        s, tc = self.slime, _SIM.tomb
        if self.phase == BossPhase.TOMB:
            if self.track is not None and 0 <= self.tick < len(self.track):
                _, dx, cy, r = self.track[self.tick]
                if tc.smash_damage_ticks[0] <= self.tick < tc.smash_damage_ticks[1]:
                    return [(self.x + dx, cy, r, r, True)]
            return []  # the tombstone only hurts while smashing
        if self.track is not None and 0 <= self.tick < len(self.track):
            _, (dx, cy, r), fists = self.track[self.tick]
            d = self.facing
            out: list[Shape] = [(self.x + d * dx, cy, r, r, True)]
            out += [(self.x + d * fdx, fcy, fhw, fhh, rnd) for fdx, fcy, fhw, fhh, rnd in fists]
            return out
        return [(self.x, s.ground_y + s.body_offset_y + self.lift, s.body_radius, s.body_radius, True)]

    def step(self, dt: float) -> list[Shape]:
        st, s = self.state, self.slime
        if st == BossMove.JUMP:
            self.vy -= s.high_jump_gravity * dt
            self.lift += self.vy * dt
            self.x += self.vx * dt
            if abs(self.x) > s.max_x:
                self.x = math.copysign(s.max_x, self.x)
                self.vx = 0.0 if self.vy < 0.0 else -self.vx
            if self.lift <= 0.0:
                self.lift, self.vy, self.vx, self.t = 0.0, 0.0, 0.0, 0.0
                self.state = BossMove.IDLE
        elif st == BossMove.IDLE and self.phase != BossPhase.TOMB:
            self.t += dt
            if self.variant in ("low", "high") and self.t >= s.ground_delay:
                high = self.variant == "high"
                self.state = BossMove.JUMP
                self.vy = s.high_jump_vy if high else s.low_jump_vy
                self.vx = self.facing * (s.high_jump_vx if high else s.low_jump_vx)
                self.variant = ""
        elif st == BossMove.TOMB_MOVE:
            self.x = max(-_SIM.tomb.max_x, min(_SIM.tomb.max_x, self.x + self.vx * dt))
        if self.track is not None:
            self.tick += round(dt * spec.GAME_FPS)
            if self.tick >= len(self.track):
                self.track = None
                self.state = BossMove.TOMB_MOVE if self.phase == BossPhase.TOMB else BossMove.IDLE
        return self.shapes()


def _scenarios(o: np.ndarray, px: float, py: float) -> list[_BossSim]:
    """All continuations of Goopy's current move the planner must survive."""
    phase = BossPhase(int(np.argmax(o[_PHASE:_PHASE + 3])))
    state = BossMove(int(np.argmax(o[_STATE:_STATE + len(BossMove)])))
    hh = o[_BOSS + 4] * H
    bx = px + o[_BOSS] * W
    by = py + o[_BOSS + 1] * H
    vx, vy = o[_BOSS + 5] * V, o[_BOSS + 6] * V
    facing = 1 if o[_BOSS + 8] >= 0 else -1
    t = o[_BOSS + 7] * 4.0
    ticks = int(round(t * spec.GAME_FPS))
    slime = _SIM.big if phase == BossPhase.BIG else _SIM.small
    lift = max(0.0, by - slime.ground_y - slime.body_offset_y) if phase != BossPhase.TOMB else 0.0

    def mk(st: BossMove, variant: str = "", face: int = facing, tick: int = 0, x: float = bx) -> _BossSim:
        return _BossSim(phase, st, x, lift, vx, vy, face, tick, variant)

    if state == BossMove.TRANSFORM:
        if phase == BossPhase.BIG:  # growing: plan against the final size, standing still
            return [mk(BossMove.IDLE)]
        # Dying big Goopy keeps hurting until the dropping tombstone crushes it; the
        # drop itself is harmless.
        body_seen = any(o[_HAZ + i * spec.HAZARD_FEATURES] > 0.5 and int(np.argmax(
            o[_HAZ + i * spec.HAZARD_FEATURES + 8:_HAZ + (i + 1) * spec.HAZARD_FEATURES])) == HazardKind.BODY
            for i in range(spec.MAX_HAZARDS))
        return [_BossSim(BossPhase.BIG, BossMove.IDLE, bx, 0.0, 0.0, 0.0, facing, 0)] if body_seen else []
    if phase == BossPhase.TOMB:
        if state in (BossMove.TOMB_SLAM_WINDUP, BossMove.TOMB_SLAM):
            tick = ticks + (live_tracks.TOMB_SMASH_WINDUP_TICKS if state == BossMove.TOMB_SLAM else 0)
            tick = min(tick, len(live_tracks.TOMB_SMASH) - 1)
            x = bx - live_tracks.TOMB_SMASH[tick][1]
            return [mk(state, tick=tick, x=x)]
        return [mk(state)]
    if state in (BossMove.PUNCH_WINDUP, BossMove.PUNCH):
        track = live_tracks.BIG_PUNCH if phase == BossPhase.BIG else live_tracks.SMALL_PUNCH
        windup = live_tracks.BIG_PUNCH_WINDUP_TICKS if phase == BossPhase.BIG else live_tracks.SMALL_PUNCH_WINDUP_TICKS
        if state == BossMove.PUNCH:  # the track after the windup is the same for every variant
            ticks_in = {ticks + windup}
        else:
            # The observation can't tell the windup variants apart, so survive all of them:
            # the full live windup, the later big punches that skip its blink, and small
            # Goopy's turn-punch with 23 extra (body-less) ticks after the start anim.
            ticks_in = {ticks}
            if slime.repeat_punch_skip:
                ticks_in.add(ticks + slime.repeat_punch_skip)
            if slime.punch_turn_frames and ticks >= slime.punch_turn_at:
                ticks_in.add(ticks - slime.punch_turn_frames)
        out = []
        for tick in sorted(ticks_in):
            tick = min(tick, len(track) - 1)
            x = bx - facing * track[max(tick, 0)][1][0]  # body centre -> transform x
            out.append(mk(state, tick=tick, x=x))
        return out
    if state in (BossMove.IDLE, BossMove.INTRO):
        out = [mk(BossMove.IDLE, "low"), mk(BossMove.IDLE, "high")]
        d = 1 if px > bx else -1
        in_position = (slime.punch_min_x < bx < slime.punch_max_x if d > 0
                       else -slime.punch_max_x < bx < -slime.punch_min_x)
        if in_position:
            # A punch can start after the punch delay: model it as a windup that begins late.
            punch = mk(BossMove.PUNCH_WINDUP, face=d)
            punch.tick = -int(round(max(0.0, slime.punch_delay - t) * spec.GAME_FPS))
            out.append(punch)
        return out
    return [mk(state)]


# ------------------------------------------------------------------------------ player model
def _player_path(px: float, py: float, pvy: float, grounded: bool, can_dash: bool,
                 move: int, jump: bool, dash: float, tap: bool, duck: bool = False) -> list[tuple[float, float, int]]:
    """(x, feet y, pose) over the horizon for one candidate manoeuvre; pose 0 = airborne,
    1 = standing, 2 = ducking (holding down on the ground: no walking, low hurtbox).

    dash: delay in seconds before dashing (0 = now), or a negative value for no dash.
    tap: only move for one decision step (used to turn around).
    """
    if duck and grounded:
        return [(px, 0.0, 2)] * _STEPS
    pc = _SIM.player
    lim = _SIM.arena.half_width - pc.hurt_w * 0.5
    x, y, vy = px, py, pvy
    on_ground = grounded
    hold = 0.0
    rise = pc.jump_speed - pc.gravity * _SIM.dt  # held rise: a tick of gravity after each reset
    if not grounded and abs(vy - rise) < 1.0:  # still in the held, constant-vy rise
        hold = max(0.0, pc.jump_hold_max - y / rise)
    if jump and grounded:
        vy, on_ground, hold = rise, False, pc.jump_hold_max
    dash_t = pc.dash_time if dash == 0.0 and (grounded or can_dash) else 0.0
    dash_dir = move or 1
    air_dash_at = dash if dash > 0.0 else -1.0
    out = []
    for i in range(1, _STEPS + 1):
        t = i * _DT
        if 0.0 < air_dash_at <= t:
            dash_t, air_dash_at = pc.dash_time, -1.0
        if dash_t > 0.0:
            dash_t -= _DT
            x += dash_dir * pc.dash_speed * _DT
        else:
            if not tap or t <= _TAP:
                x += move * pc.run_speed * _DT
            if not on_ground:
                if hold > 0.0:
                    hold -= _DT
                else:
                    vy = max(-pc.max_fall_speed, vy - pc.gravity * _DT)
                y += vy * _DT
                if y <= 0.0:
                    y, vy, on_ground = 0.0, 0.0, True
        x = max(-lim, min(lim, x))
        out.append((x, y, 1 if on_ground else 0))
    return out


def _touches(x: float, y: float, pose: int, shape: Shape) -> bool:
    pc = _SIM.player
    if pose == 1:
        cy, phh = y + pc.hurt_cy, pc.hurt_h * 0.5 + _MARGIN
    elif pose == 2:
        cy, phh = y + pc.duck_hurt_cy, pc.duck_hurt_h * 0.5 + _MARGIN
    else:
        cy, phh = y + pc.air_hurt_cy, pc.air_hurt_h * 0.5 + _MARGIN
    phw = pc.hurt_w * 0.5 + _MARGIN
    hx, hy, hw, hh, rnd = shape
    if rnd:
        dx = max(0.0, abs(hx - x) - phw)
        dy = max(0.0, abs(hy - cy) - phh)
        return dx * dx + dy * dy < hw * hw
    return abs(hx - x) < hw + phw and abs(hy - cy) < hh + phh


def _time_to_hit(path: list[tuple[float, float, int]], boss: _BossSim) -> int:
    """Index of the first planner step where the player touches a hazard (or _STEPS)."""
    for i, (x, y, g) in enumerate(path):
        for shape in boss.step(_DT):
            if _touches(x, y, g, shape):
                return i
    return _STEPS


def heuristic_policy(obs: np.ndarray) -> np.ndarray:
    """Hand-written player: shoot constantly, face Goopy, dodge with a 1 s lookahead."""
    o = np.asarray(obs, dtype=np.float64)
    px, py, pvy = o[_P] * W, o[_P + 1] * H, o[_P + 3] * V
    grounded, can_dash, facing = o[_P + 4] > 0.5, o[_P + 6] > 0.5, int(np.sign(o[_P + 7]) or 1)
    state = BossMove(int(np.argmax(o[_STATE:_STATE + len(BossMove)])))
    dx = o[_BOSS] * W
    to_boss = 1 if dx >= 0 else -1
    face = to_boss if facing != to_boss else 0

    # Parry pink question marks when one is right above us (and Goopy is not).
    for i in range(spec.MAX_HAZARDS):
        b = _HAZ + i * spec.HAZARD_FEATURES
        if o[b] > 0.5 and o[b + 7] > 0.5:
            hx, hy = o[b + 1] * W, o[b + 2] * H
            if abs(hx) < 60 and 20 < hy < 260 and abs(dx) > 300 and (grounded or (pvy < 0 and abs(hy - 50) < 60)):
                return _act(int(np.sign(hx)), True, False, True, False)

    if state in (BossMove.INTRO, BossMove.DEAD):
        return _act(face, False, False, True, False)

    phase = BossPhase(int(np.argmax(o[_PHASE:_PHASE + 3])))
    tomb = phase == BossPhase.TOMB and state != BossMove.TRANSFORM

    # Candidate manoeuvres in order of preference: (move, jump, dash, tap).
    options: list[tuple] = []  # (move, jump, dash delay s or -1, tap, duck)
    if tomb:
        # The tombstone only hurts while smashing: stay under it and shoot straight up.
        if abs(dx) > 30.0:
            options.append((to_boss, False, -1.0, False))
    elif face:
        options.append((face, False, -1.0, True))  # turn around to shoot
    options += [(0, False, -1.0, False), (to_boss, False, -1.0, False), (-to_boss, False, -1.0, False)]
    if grounded:
        options.append((0, False, -1.0, False, True))  # duck under it
        options += [(0, True, -1.0, False), (-to_boss, True, -1.0, False), (to_boss, True, -1.0, False),
                    (to_boss, True, 0.28, False), (-to_boss, True, 0.28, False),
                    (-to_boss, False, 0.0, False), (to_boss, False, 0.0, False)]
    elif can_dash:
        # Air dash now or a little later (e.g. once high enough to clear Goopy).
        options += [(d, False, delay, False) for delay in (0.0, 0.067, 0.133, 0.2) for d in (-to_boss, to_boss)]

    # Commit to a manoeuvre only when doing nothing would get us hit soon; dodging too
    # early (e.g. jumping long before the fist arrives) just lands us in its path.
    winding_up = state == BossMove.PUNCH_WINDUP
    best, best_t = options[0], -1
    for k, opt in enumerate(options):
        move, jump, dash, tap, *rest = opt
        duck = bool(rest and rest[0])
        path = _player_path(px, py, pvy, grounded, can_dash, move, jump, dash, tap, duck)
        t_hit = min((_time_to_hit(path, sim) for sim in _scenarios(o, px, py)), default=_STEPS)
        # Punch windups are long and the punch covers a lot of ground: never wait them out.
        if t_hit >= _STEPS or (k < 2 and t_hit >= _REACT and not winding_up):
            best = opt
            break
        if t_hit > best_t:
            best, best_t = opt, t_hit

    move, jump, dash, _tap, *rest = best
    duck = bool(rest and rest[0])
    jump_btn = (jump and grounded) or (not grounded and pvy > 0)  # keep holding while rising
    ex = o[_P + 11] > 0.5 and move == 0 and facing == to_boss and grounded and abs(dx) > 200 and not tomb
    # The tombstone floats above standing shot height: aim up (straight up under it,
    # diagonally while moving) for the whole phase.
    aim = -1 if duck else (1 if tomb else 0)
    return _act(move, jump_btn, dash == 0.0, True, bool(ex), aim)
