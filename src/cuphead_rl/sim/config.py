"""Every tunable of the simulator, in one place.

Units: spec units (1 unit = 1 px of the 1280x720 game camera), seconds, units/second.
Coordinates follow ``spec``: x_spec = x_unity, y_spec = y_unity + 250 (Level.Ground = -250).

Defaults are the Regular-difficulty values decompiled from the real game
(``data/goopy_constants.json``, documented in ``docs/GAME_CONSTANTS.md``). Fields marked
``# est`` are not statically extractable (animation-driven hitboxes and timings) and remain
estimates; ``sim/calibration.py`` maps every game-backed field to its JSON key and
``scripts/calibrate_sim.py`` checks the defaults against the JSON.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PlayerConfig:
    hurt_w: float = 50.0  # standing hurtbox 50 x 105, centre 62 above the feet
    hurt_h: float = 105.0
    hurt_cy: float = 62.0
    air_hurt_h: float = 50.0  # airborne hurtbox 50 x 50, centre 57 above the feet (live)
    air_hurt_cy: float = 57.0
    duck_hurt_h: float = 35.0  # ducking hurtbox 50 x 35, centre 25 above the feet (live)
    duck_hurt_cy: float = 25.0
    run_speed: float = 490.0
    gravity: float = 5991.78
    max_fall_speed: float = 1620.0
    jump_speed: float = 1223.1  # vy is held constant while jump is held ...
    jump_hold_min: float = 0.01  # ... for at least this long (taps)
    jump_hold_max: float = 0.16  # ... and at most this long
    dash_speed: float = 1100.0
    dash_time: float = 0.3  # 330 units; no i-frames while dashing
    ground_dash_cooldown: float = 0.1
    hit_invuln: float = 2.0
    hit_stun: float = 0.3  # no control after a hit ...
    hit_knockback: float = 300.0  # ... pushed away from the hazard ...
    hit_jump_speed: float = 972.0  # ... and knocked up
    parry_bounce_speed: float = 1620.0
    parry_window: float = 0.2  # est: parry hitbox lifetime after the in-air press
    parry_invuln: float = 0.25  # est
    # EX (LevelPlayerMotor.StartEx / exMove_cr): input and gravity off for the whole anim. The
    # shot leaves after ex_freeze (OnExFired); then the player slides back at ex_knockback
    # (unless aimed straight up/down) until the anim ends. Lockstep logs: 16 + 18 ticks.
    ex_freeze: float = 16 / 60
    ex_knockback_time: float = 18 / 60
    ex_knockback: float = 230.0  # exKnockback
    input_buffer_time: float = 0.134  # a blocked jump/dash/EX press stays live this long


@dataclass
class WeaponConfig:
    pea_damage: float = 4.0
    pea_interval_frames: int = 8  # 0.11 s nominal, quantised to 8 ticks = 0.1333 s (30 DPS)
    pea_speed: float = 2250.0
    pea_y_offsets: tuple[float, ...] = (0.0, 20.0, 40.0, 20.0)  # cycled per shot
    pea_half_w: float = 5.0  # live Collider2D bounds
    pea_half_h: float = 5.0
    # Spawn points per pose: LevelPlayerWeaponManager.ProjectilePosition (in player.py).
    ex_damage: float = 8.334  # per hit
    ex_max_damage: float = 25.0
    ex_hit_distance: float = 80.0  # an overlapping EX hits again every 80 units travelled
    ex_hit_freeze: float = 0.05  # the EX pauses this long on every hit
    ex_speed: float = 1500.0
    ex_half: float = 36.0  # est
    # Super meter, in cards (game: 10 meter = 1 card, 0.0625 meter per damage, +10 per parry).
    meter_per_damage: float = 0.00625
    meter_per_parry: float = 1.0
    meter_from_ex: bool = True


@dataclass
class SlimeConfig:
    """One Goopy form (phase 1 small / phase 2 big)."""

    ground_y: float  # y of the slime's transform when standing (Unity -247 -> 3)
    body_radius: float  # circular hurt/hit box
    body_offset_y: float  # circle centre above the transform
    max_x: float  # turn-around bound for the transform x
    punch_min_x: float  # punching toward -x is allowed when punch_min_x < x < punch_max_x (mirrored)
    punch_max_x: float
    ground_delay: float  # pause on the ground between jumps
    high_jump_vy: float
    high_jump_vx: float
    high_jump_gravity: float
    low_jump_vy: float
    low_jump_vx: float
    low_jump_gravity: float
    num_jumps_min: int  # jumps between punches: uniform int in [min, max]
    num_jumps_max: int
    first_punch_jumps: int  # jumps before this form's first punch (0 = random like the rest)
    jump_pattern: str  # game pattern string: HJ / LJ / RJ (random) / D<x> (next ground delay = x)
    # Punch timing/hitboxes come from the live per-tick tracks (live_tracks.py); these two
    # game constants are contained in them and kept for the calibration check.
    punch_pre_hold: float
    punch_main_hold: float
    # Animation time on the ground the game adds to ground_delay before every jump: the
    # Jump_Squish_Start anim plays before the delay starts and Jump_Squish_End after it
    # (jump_cr waits for both). Measured on the live trace: landing -> next take-off is
    # 34 ticks for small (delay 12) and 36 for big (delay 18).
    squish_time: float = 0.0
    punch_delay: float = 0.0  # landing -> start of the live punch track (its windup)
    # Punching away from the facing side plays the longer *_Start_Turn anim: extra windup
    # ticks inserted into the live track. Small Goopy sinks out of sight meanwhile (no
    # collider, no damage); big Goopy stays put and hittable.
    punch_turn_frames: int = 0
    punch_turn_at: int = 0  # track tick where those frames go in
    punch_turn_hidden: bool = False
    repeat_punch_skip: int = 0  # windup ticks of the live track only the form's first punch plays
    punch_end_time: float = 0.0  # Punch_End anim, before the next jump's squish
    punch_end_frames: int = 0  # ... of which the live track's last frames show the start


def _small() -> SlimeConfig:
    return SlimeConfig(
        ground_y=3.0, body_radius=80.0, body_offset_y=75.0, max_x=560.0,
        punch_min_x=-1000.0, punch_max_x=200.0, ground_delay=0.2,
        high_jump_vy=2350.0, high_jump_vx=840.0, high_jump_gravity=6500.0,
        low_jump_vy=2000.0, low_jump_vx=780.0, low_jump_gravity=6500.0,
        num_jumps_min=5, num_jumps_max=8, first_punch_jumps=0,
        jump_pattern="HJ,LJ,HJ,LJ,D.5, LJ,RJ,HJ,LJ,HJ,LJ,LJ,HJ,RJ,LJ,LJ",
        punch_pre_hold=0.65, punch_main_hold=0.4,
        # trace: Punch_Start_Turn 38 ticks vs Punch_Start 15
        squish_time=22 / 60, punch_delay=13 / 60, punch_turn_frames=23, punch_turn_at=24, punch_turn_hidden=True,
        punch_end_time=12 / 60, punch_end_frames=12,
    )


def _big() -> SlimeConfig:
    return SlimeConfig(
        ground_y=3.0, body_radius=160.0, body_offset_y=130.0, max_x=500.0,
        punch_min_x=-425.0, punch_max_x=200.0, ground_delay=0.3,
        high_jump_vy=2850.0, high_jump_vx=820.0, high_jump_gravity=7500.0,
        low_jump_vy=2250.0, low_jump_vx=760.0, low_jump_gravity=7500.0,
        num_jumps_min=5, num_jumps_max=8, first_punch_jumps=2,
        jump_pattern="HJ,HJ,LJ,HJ,RJ,LJ,D.5, LJ,HL,LJ,RJ,HJ,LJ,LJ,HJ,LJ,LJ",
        punch_pre_hold=0.65, punch_main_hold=0.4,
        # The live big track starts at the landing and was recorded on a first, non-turning
        # punch: Punch_Blink_Start 64 ticks (trace: Punch_Blink_Start_Turn 75, so a turn adds
        # 11). Later punches play Punch_Start instead: 18 ticks when turning (trace), so
        # 7 without (est: 18 - 11).
        squish_time=18 / 60, punch_delay=0.0, repeat_punch_skip=57,
        punch_turn_frames=11, punch_turn_at=13,
        punch_end_time=28 / 60, punch_end_frames=13,
    )


@dataclass
class TombConfig:
    rest_y: float = 170.0  # transform y at rest (Unity -80)
    drop_start_y: float = 800.0  # Unity 550
    drop_time: float = 0.2  # linear drop onto big Goopy
    crush_slime_y: float = 320.0  # big Goopy explodes once the dropping tomb passes Unity y 70
    death_to_drop: float = 3.5  # big Goopy's death loop before the drop
    death_anim: float = 0.6  # est: death animation before the loop starts
    max_x: float = 500.0
    move_speed: float = 950.0  # average speed per eased leg
    # Pauses before a leg's tween starts (move_cr waits for the Move_Left/Right anim), from
    # the live trace: Land anim + Move_Start after the drop, Move_Start after a smash,
    # the wall turn between legs.
    land_hold: float = 125 / 60
    start_hold: float = 10 / 60
    turn_hold: float = 20 / 60
    attack_delay_min: float = 2.0
    attack_delay_max: float = 4.0
    attack_offsets: tuple[float, ...] = (-130.0, 70.0, 0.0, 130.0, -70.0, 0.0)
    anticipation_hold: float = 0.35  # part of the live smash track's 30-tick windup
    # One circle (live Collider2D bounds): r 100 at +170 over the transform at rest, +145
    # while sliding (310-340 y_spec: above standing shots, so aim up). The smash replays
    # the live track (live_tracks.TOMB_SMASH), where the circle swells to r ~250 and drops
    # through the floor.
    body_radius: float = 100.0
    body_offset_y: float = 170.0
    body_offset_y_moving: float = 145.0
    body_dx_moving: float = 40.0  # the circle leans 40 forward while sliding (live trace)
    # Smash track ticks with dealDamage set (anim events; stand_shoot trace: the TOMB hazard
    # is on from 40 to 57 ticks after the anticipation starts, both smashes).
    smash_damage_ticks: tuple[int, int] = (40, 58)
    # Smash track ticks with the damage receiver off (DisableDamageReceiver anim event,
    # live trace: 8 to 30 ticks into the crash, while the stone is underground).
    smash_invuln_ticks: tuple[int, int] = (38, 60)


@dataclass
class TransitionConfig:
    intro_time: float = 10 / 60  # IntroContinue starts jump_cr at once; intro anim ends at tick 10 (trace)
    big_trigger_frac: float = 0.76  # phase 2 starts when HP fraction drops below this
    tomb_trigger_frac: float = 0.31  # phase 3 starts below this
    transform_max_abs_x: float = 350.0  # small Goopy only transforms on a landing with |x| < this
    # Small -> big transformation, measured on the game trace (stand_shoot, ticks from the
    # start of anim "small_slime_transform"): 390 ticks in all; the small r80 body stays
    # until tick 253, grows to the big r160 body by tick 290; the three pink question marks
    # are on from tick 151 to 370. Goopy takes damage throughout.
    grow_time: float = 6.5
    grow_start: float = 253 / 60
    grow_end: float = 290 / 60
    qmark_on: float = 151 / 60
    qmark_off: float = 370 / 60
    qmark_offsets_start: tuple[float, ...] = (-156.0, 200.0, 0.0, 316.0, 156.0, 200.0)  # (dx, y) pairs
    qmark_offsets_end: tuple[float, ...] = (-209.0, 229.0, 0.0, 406.0, 209.0, 229.0)  # after the growth
    qmark_harmful: bool = False  # live trace: no damage dealer on them
    qmark_radius: float = 28.0


@dataclass
class ArenaConfig:
    half_width: float = 658.0  # walls at Unity x = ±658 (the camera shows ±640)
    player_start_x: float = -410.0
    boss_start_x: float = 369.0


@dataclass
class SimConfig:
    total_hp: float = 1200.0
    player: PlayerConfig = field(default_factory=PlayerConfig)
    weapon: WeaponConfig = field(default_factory=WeaponConfig)
    small: SlimeConfig = field(default_factory=_small)
    big: SlimeConfig = field(default_factory=_big)
    tomb: TombConfig = field(default_factory=TombConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    dt: float = 1.0 / 60.0

    @property
    def phase_hps(self) -> tuple[float, float, float]:
        """Nominal HP per phase (phase 1 can run longer: the transform waits for a landing)."""
        t = self.transition
        hp = self.total_hp
        return (hp * (1 - t.big_trigger_frac), hp * (t.big_trigger_frac - t.tomb_trigger_frac),
                hp * t.tomb_trigger_frac)

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> SimConfig:
        """Defaults with ``overrides`` (a possibly partial nested dict) applied."""
        return _apply(cls(), overrides)

    @classmethod
    def from_json(cls, path: str | Path) -> SimConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_game_constants(cls, path: str | Path = "data/goopy_constants.json",
                            live_path: str | Path | None = "data/live_hitboxes.json") -> SimConfig:
        """Defaults with every mapped field taken from the decompiled-constants JSON and the
        live hitbox measurements (which win where both define a field)."""
        from cuphead_rl.sim.calibration import game_overrides

        live = json.loads(Path(live_path).read_text()) if live_path and Path(live_path).exists() else None
        return cls.from_dict(game_overrides(json.loads(Path(path).read_text()), live))

    # ---------------------------------------------------------------- randomization
    def randomized(self, rng: np.random.Generator, scale: float = 0.15) -> SimConfig:
        """A copy with boss timings/speeds jittered by U(1-scale, 1+scale), 0 <= scale < 0.5.

        Hitbox sizes get two thirds of the jitter; HP, counts, positions/bounds and the
        player/weapon (identical in sim and game) are left alone. ``*_min``/``*_max`` pairs
        share one factor (so they can't invert), and both slime forms share the body-radius
        factor (so big Goopy stays bigger than small Goopy).
        """
        if not 0.0 <= scale < 0.5:
            raise ValueError(f"randomization scale must be in [0, 0.5), got {scale}")
        shared: dict[str, float] = {}

        def factor(key: str, s: float) -> float:
            if key not in shared:
                shared[key] = float(rng.uniform(1.0 - s, 1.0 + s))
            return shared[key]

        out = dataclasses.replace(self)
        for name in ("small", "big", "tomb", "transition"):
            sub = getattr(self, name)
            changes: dict[str, float] = {}
            for f in dataclasses.fields(sub):
                v = getattr(sub, f.name)
                if not isinstance(v, float) or f.name in _NO_JITTER or f.name.endswith(("_x", "_y", "_frac")):
                    continue
                s = scale * (2.0 / 3.0) if ("radius" in f.name or "half" in f.name) else scale
                if f.name == "body_radius":
                    key = "body_radius"  # shared across forms
                elif f.name.endswith(("_min", "_max")):
                    key = f"{name}.{f.name[:-4]}"  # shared by the pair
                else:
                    key = f"{name}.{f.name}"
                changes[f.name] = v * factor(key, s)
            if name == "transition" and "grow_time" in changes:
                # The growth and question-mark windows are points on the same timeline.
                k = changes["grow_time"] / sub.grow_time
                for f_name in _GROW_TIMELINE:
                    changes[f_name] = getattr(sub, f_name) * k
            setattr(out, name, dataclasses.replace(sub, **changes))
        return out


_GROW_TIMELINE = ("grow_start", "grow_end", "qmark_on", "qmark_off")  # scaled with grow_time
_NO_JITTER = {"body_dx_moving", "grow_start", "grow_end", "qmark_on", "qmark_off", "body_offset_y", "body_offset_y_moving", "rest_y", "drop_start_y"}


def _apply(obj: Any, overrides: dict[str, Any]) -> Any:
    known = {f.name: f for f in dataclasses.fields(obj)}
    changes: dict[str, Any] = {}
    for k, v in overrides.items():
        if k not in known:
            raise KeyError(f"unknown config key {type(obj).__name__}.{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur):
            changes[k] = _apply(cur, v)
        elif isinstance(cur, bool):
            changes[k] = bool(v)
        elif isinstance(cur, float):
            changes[k] = float(v)
        elif isinstance(cur, int):
            changes[k] = int(v)
        elif isinstance(cur, tuple):
            changes[k] = tuple(float(x) for x in v)
        else:
            changes[k] = v
    return dataclasses.replace(obj, **changes)
