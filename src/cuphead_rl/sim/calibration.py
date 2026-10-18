"""Mapping from SimConfig fields to the decompiled game constants (``data/goopy_constants.json``).

Every entry is ``(sim field path, json key, transform)``. ``transform`` converts the game
value to sim units (mostly the Unity -> spec y shift). Fields not listed here are
estimates (marked ``# est`` in ``config.py``).
"""

from __future__ import annotations

from typing import Any, Callable

GROUND_Y_UNITY = -250.0


def _y(v: float) -> float:
    """Unity world y -> spec y."""
    return float(v) - GROUND_Y_UNITY


def _same(v: Any) -> Any:
    return v


def _tuple(v: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in v)


Mapping = tuple[str, str, Callable[[Any], Any]]

MAPPING: list[Mapping] = [
    ("total_hp", "boss.total_hp", float),
    ("dt", "engine.fixed_dt", float),
    ("transition.big_trigger_frac", "boss.phase_big_trigger_frac", float),
    ("transition.tomb_trigger_frac", "boss.phase_tomb_trigger_frac", float),
    # boss.first_pattern_delay (1 s) only delays SlimeLevel.slimePattern_cr, which never
    # gates jump_cr: intro_time is measured on the live trace instead.
    ("transition.transform_max_abs_x", "small.transform_max_abs_x", float),
    ("transition.qmark_radius", "small.question_mark_radius", float),
    ("arena.half_width", "arena.right_x_unity", float),
    ("arena.player_start_x", "arena.player_spawn_x", float),
    ("arena.boss_start_x", "small.spawn_x", float),
    # player
    ("player.run_speed", "player.move_speed", float),
    ("player.jump_speed", "player.jump_vy", float),
    ("player.jump_hold_min", "player.jump_hold_min", float),
    ("player.jump_hold_max", "player.jump_hold_max", float),
    ("player.gravity", "player.gravity", float),
    ("player.max_fall_speed", "player.max_fall_speed", float),
    ("player.dash_speed", "player.dash_speed", float),
    ("player.dash_time", "player.dash_time", float),
    ("player.ground_dash_cooldown", "player.dash_ground_cooldown", float),
    ("player.hit_invuln", "player.hit_iframes", float),
    ("player.hit_stun", "player.hit_stun_time", float),
    ("player.hit_knockback", "player.hit_knockback", float),
    ("player.hit_jump_speed", "player.hit_jump_vy", float),
    ("player.parry_bounce_speed", "player.parry_bounce_vy", float),
    ("player.input_buffer_time", "player.input_buffer_time", float),
    ("player.hurt_w", "player.collider_w", float),
    ("player.hurt_h", "player.collider_h", float),
    ("player.hurt_cy", "player.collider_center_y", float),
    ("player.air_hurt_h", "player.collider_air_h", float),
    ("player.air_hurt_cy", "player.collider_air_center_y", float),
    # weapons / meter (game meter: 10 per card)
    ("weapon.pea_damage", "peashooter.damage", float),
    ("weapon.pea_interval_frames", "peashooter.fire_interval_frames", int),
    ("weapon.pea_speed", "peashooter.bullet_speed", float),
    ("weapon.pea_y_offsets", "peashooter.y_offsets", _tuple),
    ("weapon.ex_damage", "peashooter_ex.damage_per_hit", float),
    ("weapon.ex_max_damage", "peashooter_ex.max_damage", float),
    ("weapon.ex_speed", "peashooter_ex.speed", float),
    ("weapon.ex_hit_freeze", "peashooter_ex.hit_freeze", float),
    ("weapon.ex_hit_distance", "peashooter_ex.damage_distance", float),
    ("weapon.meter_per_damage", "super.meter_per_damage", lambda v: float(v) / 10.0),
    ("weapon.meter_per_parry", "super.cards_per_parry", float),
    ("weapon.meter_from_ex", "hidden.meter_from_ex_damage", bool),
    # tombstone
    ("tomb.rest_y", "tomb.ground_y_unity", _y),
    ("tomb.drop_start_y", "tomb.drop_start_y_unity", _y),
    ("tomb.drop_time", "tomb.drop_time", float),
    ("tomb.death_to_drop", "big.death_to_tombstone_delay", float),
    ("tomb.max_x", "tomb.max_x", float),
    ("tomb.move_speed", "tomb.move_speed", float),
    ("tomb.attack_delay_min", "tomb.attack_delay_min", float),
    ("tomb.attack_delay_max", "tomb.attack_delay_max", float),
    ("tomb.attack_offsets", "tomb.attack_offsets", _tuple),
    ("tomb.anticipation_hold", "tomb.anticipation_hold", float),
    ("tomb.body_radius", "tomb.body_radius", float),
    ("tomb.body_offset_y", "tomb.body_offset_y", float),
]

for _form in ("small", "big"):
    MAPPING += [
        (f"{_form}.max_x", f"{_form}.max_x", float),
        (f"{_form}.punch_min_x", f"{_form}.punch_min_x", float),
        (f"{_form}.punch_max_x", f"{_form}.punch_max_x", float),
        (f"{_form}.ground_delay", f"{_form}.ground_delay", float),
        (f"{_form}.num_jumps_min", f"{_form}.num_jumps_min", int),
        (f"{_form}.num_jumps_max", f"{_form}.num_jumps_max", int),
        (f"{_form}.jump_pattern", f"{_form}.jump_pattern", str),
        (f"{_form}.punch_pre_hold", f"{_form}.punch_pre_hold", float),
        (f"{_form}.punch_main_hold", f"{_form}.punch_main_hold", float),
        (f"{_form}.body_offset_y", f"{_form}.body_offset_y", float),
    ]
    for _h in ("high", "low"):
        for _k in ("vy", "vx", "gravity"):
            MAPPING.append((f"{_form}.{_h}_jump_{_k}", f"{_form}.{_h}_jump_{_k}", float))
MAPPING += [
    ("small.ground_y", "small.ground_y_unity", _y),
    ("big.ground_y", "small.ground_y_unity", _y),  # the big slime spawns at the small one's position
    ("small.body_radius", "small.body_radius", float),
    ("big.body_radius", "big.body_radius_outer", float),  # outer circle contains the inner one
    ("big.first_punch_jumps", "big.initial_jumps_before_first_punch", int),
]

# Values measured live by the bridge (data/live_hitboxes.json): (sim field, top-level key,
# sub-key or index, transform). Where a field is also in MAPPING, the live value wins.
LIVE_MAPPING: list[tuple[str, str, Any, Callable[[Any], Any]]] = [
    ("player.hurt_w", "player.hurtbox.standing", "half_w", lambda v: 2.0 * float(v)),
    ("player.hurt_h", "player.hurtbox.standing", "half_h", lambda v: 2.0 * float(v)),
    ("player.hurt_cy", "player.hurtbox.standing", "center_dy_from_feet", float),
    ("player.air_hurt_h", "player.hurtbox.airborne", "half_h", lambda v: 2.0 * float(v)),
    ("player.air_hurt_cy", "player.hurtbox.airborne", "center_dy_from_feet", float),
    ("player.duck_hurt_h", "player.hurtbox.ducking", "half_h", lambda v: 2.0 * float(v)),
    ("player.duck_hurt_cy", "player.hurtbox.ducking", "center_dy_from_feet", float),
    ("weapon.pea_half_w", "peashooter.bullet_half_size", 0, float),
    ("weapon.pea_half_h", "peashooter.bullet_half_size", 1, float),
]
LIVE_FIELDS = {path for path, *_ in LIVE_MAPPING}


def live_value(live: dict[str, Any], key: str, sub: Any) -> Any:
    return live[key][sub]


# Game facts the sim encodes as behaviour rather than as a number (checked by calibrate_sim).
EXPECTED_FLAGS = {
    "player.dash_iframes": False,
    "peashooter.fires_while_dashing": False,
    "player.dash_air_count": 1,
    "player.max_hp": 3,
}


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def game_overrides(constants: dict[str, Any], live: dict[str, Any] | None = None) -> dict[str, Any]:
    """Nested override dict for ``SimConfig.from_dict`` built from the constants JSON (and
    the live hitbox measurements, which take precedence)."""
    values: dict[str, Any] = {path: fn(constants[key]) for path, key, fn in MAPPING if key in constants}
    for path, key, sub, fn in LIVE_MAPPING if live else ():
        if key in live:
            values[path] = fn(live_value(live, key, sub))
    out: dict[str, Any] = {}
    for path, value in values.items():
        node = out
        *parents, leaf = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out
