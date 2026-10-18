"""Goopy Le Grande: three phases driven by one state machine (after ``SlimeLevelSlime`` /
``SlimeLevelTombstone``, see docs/GAME_CONSTANTS.md).

Phase 1 (SMALL) and phase 2 (BIG) share the slime logic: hop along the game's jump
pattern, turn at ±max_x, and after 5-8 jumps punch toward the player once standing in a
punch position. Phase changes are HP-fraction thresholds over the whole fight; the small
slime only transforms on a landing near the centre, the big one dies on landing, and
3.5 s later the tombstone drops on it. The tombstone patrols wall to wall with eased legs
and periodically smashes near the player; it only hurts during the smash.

Punches and the smash replay per-tick hitbox tracks measured live in the game
(``live_tracks.py``): every boss hitbox is a circle, except the punch arm boxes.

``x`` is the transform x; ``lift`` is the height above the rest height.
``y``/``half_w``/``half_h`` describe the body hitbox (centre and half extents).
"""

from __future__ import annotations

import math
import random

from cuphead_rl.spec import BossMove, BossPhase, Hazard, HazardKind
from cuphead_rl.sim import live_tracks
from cuphead_rl.sim.config import SimConfig, SlimeConfig

Body = tuple[float, float, float, float, bool]
HazardTuple = tuple[HazardKind, float, float, float, float, float, float, bool]

# Sub-stages of BossMove.TRANSFORM.
GROW = "grow"  # small -> big, eats the jellybean
DYING = "dying"  # big Goopy collapses (death animation + loop)
FALLING = "falling"  # tombstone drops from the sky onto him

# The bridge reports collider velocities as per-tick centre differences and takes a jump
# faster than this for a teleport (v = 0): StateReader.MaxSpeed.
TELEPORT_SPEED = 6000.0


def track_velocity(dx: float, dy: float, dt: float) -> tuple[float, float]:
    """Velocity of a track hitbox that moved (dx, dy) in one tick, as the bridge reports it."""
    vx, vy = dx / dt, dy / dt
    return (0.0, 0.0) if vx * vx + vy * vy > TELEPORT_SPEED * TELEPORT_SPEED else (vx, vy)


def parse_pattern(pattern: str) -> list[tuple[str, float]]:
    """Game jump-pattern string -> [(kind, ground_delay_after)], kind in H/L/R.

    ``D<x>`` sets the ground delay before the following jump. Entries are read by their
    first non-space character, as the game does (so the typo ``HL`` is a high jump).
    """
    out: list[tuple[str, float]] = []
    delay = -1.0
    for tok in pattern.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok[0] == "D":
            delay = float(tok[1:])
        else:
            out.append((tok[0], delay))
            delay = -1.0
    return out


def pattern_starts(pattern: str) -> list[tuple[int, bool]]:
    """Where jump_cr can start: one entry per RAW token of the pattern string (D tokens
    included, as ``Random.Range(0, pattern.Length)``), giving the index of the parsed jump
    it leads to and whether that jump's D delay is skipped (start right after the D)."""
    toks = [t.strip() for t in pattern.split(",") if t.strip()]
    out: list[tuple[int, bool]] = []
    j = 0
    for k, tok in enumerate(toks):
        if tok[0] == "D":
            out.append((j, False))
        else:
            out.append((j, k > 0 and toks[k - 1][0] == "D"))
            j += 1
    return [(i % j, skip) for i, skip in out]


class Goopy:
    def __init__(self, cfg: SimConfig, rng: random.Random, start_phase: BossPhase = BossPhase.SMALL) -> None:
        self.cfg = cfg
        self.rng = rng
        self.max_hp = cfg.total_hp
        t = cfg.transition
        self.phase = BossPhase(start_phase)
        start_frac = (1.0, t.big_trigger_frac, t.tomb_trigger_frac)[self.phase]
        self.hp = self.max_hp * start_frac
        self.state = BossMove.INTRO
        self.state_time = 0.0
        self.transform = ""
        self.x = cfg.arena.boss_start_x
        self.lift = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.facing = -1
        self.wants_transform = False
        self.wants_death = False
        self._gravity = 0.0
        # slime
        self.pattern: list[tuple[str, float]] = []
        self.pattern_i = 0
        self.skip_delay = False  # started right after a D token: its delay is not applied
        self.punch_dir = 0  # punch decided at the last landing (jump_cr), 0 = none
        self.ground_hold = 0.0  # extra time on the ground before the next jump
        self.jumps_to_punch = 0
        self.punches_done = 0
        self.track: tuple | None = None  # live hitbox track of the current punch/smash
        self.track_tick = 0
        self.attack_dir = 1
        # tomb
        self.leg_from = self.leg_to = self.leg_time = 0.0
        self.leg_dir = 1
        self.wants_smash = False
        self.smash_offset = 0.0
        self.attack_timer = 0.0
        self.attack_i = 0
        self.target_x: float | None = None
        if self.phase == BossPhase.TOMB:
            self._begin(BossMove.TRANSFORM)
            self._start_fall()
        else:
            self._enter_slime_form()

    # ------------------------------------------------------------------ geometry
    @property
    def slime(self) -> SlimeConfig:
        return self.cfg.big if self.phase == BossPhase.BIG else self.cfg.small

    def _is_slime_body(self) -> bool:
        return self.phase != BossPhase.TOMB or (self.state == BossMove.TRANSFORM and self.transform == DYING)

    def _radius(self) -> float:
        s, b = self.cfg.small, self.cfg.big
        if self.state == BossMove.TRANSFORM and self.transform == GROW:
            t = self.cfg.transition
            k = min(1.0, max(0.0, (self.state_time - t.grow_start) / (t.grow_end - t.grow_start)))
            return s.body_radius + (b.body_radius - s.body_radius) * k
        return s.body_radius if self.phase == BossPhase.SMALL else b.body_radius

    def body(self) -> Body:
        """Body hitbox (x, centre y, half_w, half_h, round), computed once per call.

        Callers that need several of these per frame should call this once instead of
        the individual properties.
        """
        if self.track is not None:
            x, cy, r = self._track_body(self.track_tick)
            return x, cy, r, r, True
        if self._is_slime_body():
            s, b = self.cfg.small, self.cfg.big
            r = self._radius()
            k = (r - s.body_radius) / (b.body_radius - s.body_radius)
            y = s.ground_y + s.body_offset_y + (b.body_offset_y - s.body_offset_y) * k + self.lift
            return self.x, y, r, r, True
        tc = self.cfg.tomb
        if self.tomb_sliding:
            return (self.x + self.leg_dir * tc.body_dx_moving, tc.rest_y + tc.body_offset_y_moving,
                    tc.body_radius, tc.body_radius, True)
        return self.x, tc.rest_y + self.lift + tc.body_offset_y, tc.body_radius, tc.body_radius, True

    def _track_body(self, i: int) -> tuple[float, float, float]:
        """(x, centre y, radius) of the body circle in frame ``i`` of the current track."""
        fr = self.track[i]
        if self.phase == BossPhase.TOMB:
            _, dx, cy, r = fr
            return self.x + dx, cy, r
        dx, cy, r = fr[1]
        return self.x + self.attack_dir * dx, cy, r

    def velocity(self) -> tuple[float, float]:
        """Body velocity as the game reports it. While a live track plays (the punch windup
        and lunge, the smash), the collider moves frame by frame, not by ``vx``/``vy``: its
        per-tick centre difference, 0 on a collider just switched on (track start, hidden
        turn frames) and on jumps faster than TELEPORT_SPEED (the lunge itself)."""
        if self.track is None:
            return self.vx, self.vy
        if not self.track_tick:
            return 0.0, 0.0
        x0, y0, r0 = self._track_body(self.track_tick - 1)
        x1, y1, r1 = self._track_body(self.track_tick)
        if r0 <= 0.0 or r1 <= 0.0:
            return 0.0, 0.0
        return track_velocity(x1 - x0, y1 - y0, self.cfg.dt)

    def hit_body(self, body: Body | None = None) -> Body:
        """What the player's shots collide with: the body, except that while the tombstone
        drops, big Goopy (still there, collider on) is what stands in the line of fire."""
        if self.crushed_slime_alive:
            b = self.cfg.big
            return self.x, b.ground_y + b.body_offset_y, b.body_radius, b.body_radius, True
        return body if body is not None else self.body()

    @property
    def smash_damage(self) -> bool:
        """The tombstone only deals damage during the crash of its smash."""
        return (self.track is not None and self.phase == BossPhase.TOMB
                and self.cfg.tomb.smash_damage_ticks[0] <= self.track_tick < self.cfg.tomb.smash_damage_ticks[1])

    @property
    def is_round(self) -> bool:
        """Every Goopy body hitbox is a circle (slimes and tombstone alike)."""
        return True

    @property
    def half_w(self) -> float:
        return self.body()[2]

    @property
    def half_h(self) -> float:
        return self.body()[3]

    @property
    def y(self) -> float:
        return self.body()[1]

    @property
    def bottom(self) -> float:
        _, y, _, hh, _ = self.body()
        return y - hh

    @property
    def crushed_slime_alive(self) -> bool:
        """While the tombstone drops, big Goopy's body is still there (and still hurts)
        until the tomb passes ``crush_slime_y`` (SlimeLevelTombstone.crush_slime_cr)."""
        return (self.state == BossMove.TRANSFORM and self.transform == FALLING
                and self.cfg.tomb.rest_y + self.lift > self.cfg.tomb.crush_slime_y)

    @property
    def vulnerable(self) -> bool:
        """Goopy takes damage whenever a collider is up, also while transforming or dying:
        every hit goes through AbstractLevelProperties.DealDamage on the one HP pool.
        Exceptions: small Goopy's turn before a punch (no collider) and the tombstone
        while underground in its smash (damage receiver off)."""
        if self.track is not None:
            if self.phase == BossPhase.TOMB:
                lo, hi = self.cfg.tomb.smash_invuln_ticks
                return not lo <= self.track_tick < hi
            return self.track[self.track_tick][1][2] > 0.0
        return self.state != BossMove.DEAD

    @property
    def dead(self) -> bool:
        return self.state == BossMove.DEAD

    @property
    def phase_hp_frac(self) -> float:
        t = self.cfg.transition
        bounds = (1.0, t.big_trigger_frac, t.tomb_trigger_frac, 0.0)
        hi, lo = bounds[self.phase] * self.max_hp, bounds[self.phase + 1] * self.max_hp
        return max(0.0, min(1.0, (self.hp - lo) / (hi - lo)))

    def _begin(self, state: BossMove) -> None:
        self.state = state
        self.state_time = 0.0

    # ------------------------------------------------------------------ damage
    def damage(self, amount: float) -> float:
        """Apply damage; returns the amount removed. Excess carries across phases."""
        if not self.vulnerable:
            return 0.0
        dealt = min(amount, self.hp)
        self.hp -= dealt
        frac = self.hp / self.max_hp
        t = self.cfg.transition
        if self.hp <= 1e-9:
            self.hp = 0.0
            self.track, self.vx = None, 0.0
            self._begin(BossMove.DEAD)
        elif self.phase == BossPhase.SMALL and frac < t.big_trigger_frac:
            self.wants_transform = True
        elif self.phase == BossPhase.BIG and frac < t.tomb_trigger_frac and not self.wants_death:
            self.wants_death = True
            # DeathTransform: airborne -> dies on landing; on the ground it dies at once, even
            # mid-punch (Die stops every coroutine). Still growing: dies once grown.
            if self.state not in (BossMove.JUMP, BossMove.TRANSFORM):
                self._die()
        return dealt

    def _die(self) -> None:
        """Big Goopy collapses. The phase stays BIG (the visible entity) until the
        tombstone appears (spec.BossPhase)."""
        self.phase = BossPhase.BIG
        self.track, self.vx, self.vy = None, 0.0, 0.0
        self._begin(BossMove.TRANSFORM)
        self.transform = DYING

    def _start_fall(self) -> None:
        tc = self.cfg.tomb
        self.phase = BossPhase.TOMB
        self.transform = FALLING
        self.attack_i = self.rng.randrange(len(tc.attack_offsets))  # StartIntro: random offset start
        self.x = max(-tc.max_x, min(tc.max_x, self.x))
        self.lift = tc.drop_start_y - tc.rest_y
        self.vy = -self.lift / tc.drop_time
        self.vx = 0.0

    def _enter_slime_form(self) -> None:
        s = self.slime
        self.pattern = parse_pattern(s.jump_pattern)
        self._random_pattern_start()
        self.ground_hold = 0.0
        self.jumps_to_punch = s.first_punch_jumps or self.rng.randint(s.num_jumps_min, s.num_jumps_max)
        self.punches_done = 0
        self.x = max(-s.max_x, min(s.max_x, self.x))

    # ------------------------------------------------------------------ update
    def step(self, dt: float, player_x: float) -> None:
        self.state_time += dt
        st = self.state
        if st == BossMove.DEAD:
            return
        if st == BossMove.INTRO:
            if self.state_time >= self.cfg.transition.intro_time:
                self._begin(BossMove.IDLE)
            return
        if st == BossMove.TRANSFORM:
            self._step_transform(dt)
        elif self.phase == BossPhase.TOMB:
            self._step_tomb(dt, player_x)
        else:
            self._step_slime(dt, player_x)

    def _step_transform(self, dt: float) -> None:
        tc = self.cfg.tomb
        if self.transform == GROW:
            if self.state_time >= self.cfg.transition.grow_time:
                self.transform = ""
                self._enter_slime_form()
                self._begin(BossMove.IDLE)
                if self.wants_death:
                    self._die()
        elif self.transform == DYING:
            if self.state_time >= tc.death_anim + tc.death_to_drop:
                self._start_fall()
        elif self.transform == FALLING:
            self.lift += self.vy * dt
            if self.lift <= 0.0:
                self.lift = self.vy = 0.0
                self.transform = ""
                self._start_tomb(tc.land_hold)

    # ---------------------------------------------------------------- slime (1 & 2)
    def _in_punch_position(self, d: int) -> bool:
        """inPunchPosition: toward the right inside (min, max), toward the left inside the
        mirrored (-max, -min)."""
        s = self.slime
        if d > 0:
            return s.punch_min_x < self.x < s.punch_max_x
        return -s.punch_max_x < self.x < -s.punch_min_x

    def _random_pattern_start(self) -> None:
        starts = pattern_starts(self.slime.jump_pattern)
        self.pattern_i, self.skip_delay = starts[self.rng.randrange(len(starts))]

    def _step_slime(self, dt: float, player_x: float) -> None:
        s = self.slime
        st = self.state
        if self.track is not None:
            # Replay the live punch track: windup frames, the punch, then the start of
            # Punch_End, where punch_cr already calls StartJump (Goopy counts as idle).
            self.track_tick += 1
            if self.track_tick >= len(self.track):
                self.track = None
                if st != BossMove.IDLE:
                    self._end_punch()
            elif st == BossMove.PUNCH_WINDUP and not self.track[self.track_tick][0]:
                self._begin(BossMove.PUNCH)
            elif st == BossMove.PUNCH and self.track_tick >= len(self.track) - s.punch_end_frames:
                self._end_punch()
            return
        if st == BossMove.IDLE:
            if self.punch_dir:  # decided on landing
                if self.state_time >= s.punch_delay:
                    self._start_punch(self.punch_dir)
                return
            # A D<x> pattern entry sets the delay before the jump that follows it.
            kind, delay = self.pattern[self.pattern_i % len(self.pattern)]
            if self.skip_delay:
                delay = -1.0
            if self.state_time < self.ground_hold + (delay if delay >= 0.0 else s.ground_delay) + s.squish_time:
                return
            self.pattern_i += 1
            if kind == "R":
                kind = self.rng.choice("HL")
            high = kind == "H"
            self.vy = s.high_jump_vy if high else s.low_jump_vy
            self.vx = self.facing * (s.high_jump_vx if high else s.low_jump_vx)
            self._gravity = s.high_jump_gravity if high else s.low_jump_gravity
            self.ground_hold = 0.0
            self.skip_delay = False
            self._begin(BossMove.JUMP)
        elif st == BossMove.JUMP:
            # Semi-implicit Euler, as jump_cr: vy -= g dt; pos += v dt.
            self.vy -= self._gravity * dt
            self.lift += self.vy * dt
            self.x += self.vx * dt
            if abs(self.x) > s.max_x:
                self.x = math.copysign(s.max_x, self.x)
                self.facing = -self.facing
                # Hitting the bound on the way down kills the horizontal speed.
                self.vx = 0.0 if self.vy < 0.0 else -self.vx
            if self.lift <= 0.0:
                self.lift = self.vy = self.vx = 0.0
                self.jumps_to_punch -= 1
                self._begin(BossMove.IDLE)
                self._on_landing(player_x)

    def _end_punch(self) -> None:
        """StartJump: a fresh jump_cr (new jump count, random pattern start), whose first
        squish waits for the Punch_End anim."""
        s = self.slime
        self.punches_done += 1
        self.jumps_to_punch = self.rng.randint(s.num_jumps_min, s.num_jumps_max)
        self._random_pattern_start()
        self.ground_hold = s.punch_end_time
        self._begin(BossMove.IDLE)

    def _start_punch(self, d: int) -> None:
        """Replay the live punch track toward ``d``. Turning to punch plays the longer
        Punch_Start_Turn anim; only a form's first punch plays the big blink windup."""
        s = self.slime
        track = live_tracks.BIG_PUNCH if self.phase == BossPhase.BIG else live_tracks.SMALL_PUNCH
        if self.punches_done and s.repeat_punch_skip:
            track = track[s.repeat_punch_skip:]
        if d != self.facing and s.punch_turn_frames:
            at = s.punch_turn_at
            hold = (True, (0.0, 0.0, 0.0), ()) if s.punch_turn_hidden else track[at]
            track = track[:at] + (hold,) * s.punch_turn_frames + track[at:]
        self.attack_dir = self.facing = d
        self.punch_dir = 0
        self.track, self.track_tick = track, 0
        self._begin(BossMove.PUNCH_WINDUP)

    def _on_landing(self, player_x: float) -> None:
        if self.phase == BossPhase.SMALL and self.wants_transform \
                and abs(self.x) < self.cfg.transition.transform_max_abs_x:
            self.phase = BossPhase.BIG
            self._begin(BossMove.TRANSFORM)
            self.transform = GROW
        elif self.phase == BossPhase.BIG and self.wants_death:
            self._die()
        elif self.jumps_to_punch <= 0:
            # jump_cr decides once, on the landing frame, from where the player is now
            # (Punch() picks the side the same frame). Big Goopy's first punch comes after a
            # fixed jump count, wherever it stands.
            s = self.slime
            d = 1 if player_x > self.x else -1
            first_big = self.phase == BossPhase.BIG and self.punches_done == 0 and s.first_punch_jumps
            if first_big or self._in_punch_position(d):
                self.punch_dir = d

    # ---------------------------------------------------------------- tomb (3)
    def _start_tomb(self, hold: float, elapsed: float = 0.0) -> None:
        """SlimeLevelTombstone.StartMove: arm the smash timer (started ``elapsed`` ago) and
        slide toward a random wall after ``hold``."""
        tc = self.cfg.tomb
        self.attack_timer = self.rng.uniform(tc.attack_delay_min, tc.attack_delay_max) - elapsed
        self.target_x = None
        self.wants_smash = False
        off = tc.attack_offsets[self.attack_i % len(tc.attack_offsets)]
        self.attack_i += 1
        self.smash_offset = off
        self._new_leg(self.rng.choice((-1, 1)), hold)
        self._begin(BossMove.TOMB_MOVE)

    def _new_leg(self, d: int, hold: float) -> None:
        self.leg_from, self.leg_to, self.leg_time = self.x, d * self.cfg.tomb.max_x, -hold
        self.leg_dir = d

    @property
    def tomb_sliding(self) -> bool:
        return self.state == BossMove.TOMB_MOVE and self.leg_time > 0.0

    def _step_tomb(self, dt: float, player_x: float) -> None:
        tc = self.cfg.tomb
        st = self.state
        if st == BossMove.TOMB_MOVE:
            self.attack_timer -= dt
            if self.attack_timer <= 0.0:
                self.wants_smash = True
            self.leg_time += dt
            if self.leg_time <= 0.0:  # waiting for the move/turn anim
                self.vx = 0.0
                return
            # easeInOutSine leg from leg_from to leg_to; duration |dx| / move_speed.
            dur = max(abs(self.leg_to - self.leg_from) / tc.move_speed, dt)
            k = min(1.0, self.leg_time / dur)
            prev = self.x
            self.x = self.leg_from + (self.leg_to - self.leg_from) * (0.5 - 0.5 * math.cos(math.pi * k))
            self.vx = (self.x - prev) / dt
            if self.target_x is not None:
                # The smash leg: stop once past the player (+ offset), or at the wall, and
                # snap onto the clamped target (move_cr).
                self.target_x = player_x + self.smash_offset
                if (self.x - self.target_x) * self.leg_dir > 0.0 or k >= 1.0:
                    self.x, self.vx = max(-tc.max_x, min(tc.max_x, self.target_x)), 0.0
                    self.track, self.track_tick = live_tracks.TOMB_SMASH, 0
                    self._begin(BossMove.TOMB_SLAM_WINDUP)
            elif k >= 1.0:
                # The smash is only decided between legs (while (!wantsToSmash) in move_cr).
                if self.wants_smash:
                    self.target_x = player_x + self.smash_offset
                self._new_leg(-self.leg_dir, tc.turn_hold)
        elif st in (BossMove.TOMB_SLAM_WINDUP, BossMove.TOMB_SLAM):
            # Replay the live smash track (anticipation, then the crash and recovery).
            # smash_cr calls StartMove right after the anticipation; the slide waits for
            # the smash anim to end.
            self.track_tick += 1
            if self.track_tick >= len(self.track):
                self.track = None
                self._start_tomb(tc.start_hold, (len(live_tracks.TOMB_SMASH) - live_tracks.TOMB_SMASH_WINDUP_TICKS) * dt)
            elif st == BossMove.TOMB_SLAM_WINDUP and not self.track[self.track_tick][0]:
                self._begin(BossMove.TOMB_SLAM)

    # ------------------------------------------------------------------ hazards
    def hazard_tuples(self, body: Body | None = None) -> list[HazardTuple]:
        """What damages the player right now: (kind, x, y, half_w, half_h, vx, vy, round).

        Raw tuples keep the 60 Hz collision loop allocation-light; ``round`` marks a
        circular hitbox (radius = half_w). ``body`` may pass a precomputed ``self.body()``.
        """
        if self.state == BossMove.DEAD:
            return []
        x, y, hw, hh, rnd = body if body is not None else self.body()
        out: list[HazardTuple] = []
        if hw <= 0.0:
            pass  # no collider (small Goopy turning before a punch)
        elif self._is_slime_body():
            # Slime bodies always deal contact damage, also while transforming or dying
            # (SlimeLevelSlime.OnCollisionPlayer has no state check). Small Goopy's body *is*
            # the punch while it lunges (spec.HazardKind.FIST).
            lunge = self.state == BossMove.PUNCH and self.phase == BossPhase.SMALL
            out.append((HazardKind.FIST if lunge else HazardKind.BODY, x, y, hw, hh, *self.velocity(), True))
        elif self.smash_damage:  # the tombstone only hurts while smashing
            # The bridge samples the TOMB collider only while it deals damage: its first frame
            # has no earlier sample, so it reads v = 0.
            first = self.track_tick == self.cfg.tomb.smash_damage_ticks[0]
            out.append((HazardKind.TOMB, x, y, hw, hh, *((0.0, 0.0) if first else self.velocity()), True))
        if self.crushed_slime_alive:
            b = self.cfg.big
            out.append((HazardKind.BODY, x, b.ground_y + b.body_offset_y, b.body_radius, b.body_radius,
                        0.0, 0.0, True))
        if self.track is not None and self.phase != BossPhase.TOMB:
            out += self.fist_tuples()
        return out

    def qmark_positions(self) -> list[tuple[int, float, float]]:
        """(index, x, y) of the pink question marks shown during the transformation."""
        if not (self.state == BossMove.TRANSFORM and self.transform == GROW):
            return []
        t = self.cfg.transition
        if not t.qmark_on <= self.state_time < t.qmark_off:
            return []
        k = min(1.0, max(0.0, (self.state_time - t.grow_start) / (t.grow_end - t.grow_start)))
        a, b = t.qmark_offsets_start, t.qmark_offsets_end
        return [(i, self.x + a[2 * i] + (b[2 * i] - a[2 * i]) * k, a[2 * i + 1] + (b[2 * i + 1] - a[2 * i + 1]) * k)
                for i in range(len(a) // 2)]

    def fist_tuples(self) -> list[HazardTuple]:
        """The punch hitboxes of the current track frame (thrown circle / arm boxes)."""
        if self.track is None or self.phase == BossPhase.TOMB:
            return []
        d = self.attack_dir
        fists = self.track[self.track_tick][2]
        prev = self.track[self.track_tick - 1][2] if self.track_tick else ()
        out: list[HazardTuple] = []
        for i, (dx, cy, hw, hh, rnd) in enumerate(fists):
            vx, vy = track_velocity((dx - prev[i][0]) * d, cy - prev[i][1], self.cfg.dt) if i < len(prev) else (0.0, 0.0)
            out.append((HazardKind.FIST, self.x + d * dx, cy, hw, hh, vx, vy, rnd))
        return out

    def hazards(self, body: Body | None = None) -> list[Hazard]:
        return [Hazard(k, x, y, hw, hh, vx, vy) for k, x, y, hw, hh, vx, vy, _ in self.hazard_tuples(body)]
