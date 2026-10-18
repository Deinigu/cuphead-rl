"""Compare SimConfig defaults with the decompiled game constants.

    python -m uv run python scripts/calibrate_sim.py [--json data/goopy_constants.json]
                                                     [--write-overrides configs/sim_game.json]

Prints every mapped field (sim default vs game value), flags mismatches, lists JSON keys
the sim does not map, and checks behavioural flags. Exit code 1 if anything mismatches.
``--write-overrides`` writes a SimConfig override file built from the JSON, so a new
bridge dump can be used immediately via ``SimConfig.from_json`` before the defaults are
updated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from cuphead_rl.sim.calibration import (EXPECTED_FLAGS, LIVE_FIELDS, LIVE_MAPPING, MAPPING, game_overrides,
                                        get_path, live_value)
from cuphead_rl.sim.config import SimConfig

# JSON keys the sim deliberately does not consume, with the reason.
UNMAPPED_OK = {
    "engine.fps": "implied by engine.fixed_dt",
    "arena.ground_y_unity": "coordinate origin (y_spec = y_unity + 250)",
    "arena.ceiling_y_unity": "nothing in the fight reaches it",
    "arena.left_x_unity": "symmetric with arena.right_x_unity",
    "arena.camera_half_width": "renderer/spec use spec.ARENA_HALF_WIDTH = 640",
    "arena.player_spawn_y_unity": "player spawns on the ground",
    "boss.phase_small_hp": "derived from total_hp and the trigger fractions",
    "boss.phase_big_hp": "derived",
    "boss.phase_tomb_hp": "derived",
    "boss.hit_pause_on_super": "supers are not in the action space",
    "boss.first_pattern_delay": "only delays SlimeLevel.slimePattern_cr, which never gates jump_cr",
    "big.body_radius_inner": "contained in the outer r=160 circle",
    "tomb.move_ease": "implemented as easeInOutSine legs in goopy.py",
    "tomb.tiny_health": "tiny slimes never spawn (SlimeLevelSlime.TINIES = false)",
    "tomb.tiny_run_time": "tiny slimes never spawn",
    "tomb.tiny_melt_delay": "tiny slimes never spawn",
    "tomb.tiny_time_until_unmelt": "tiny slimes never spawn",
    "player.jump_power_raw": "normalised form of player.jump_vy",
    "player.jump_apex_full_hold": "checked by tests (sim 298 vs 283; airtime exact)",
    "player.jump_airtime_full_hold": "checked by tests",
    "player.jump_apex_tap": "checked by tests",
    "player.jump_airtime_tap": "checked by tests",
    "player.coyote_time": "no ledges in this arena",
    "player.collider_duck_h": "ducking is not in the action space",
    "player.collider_duck_center_y": "ducking is not in the action space",
    "player.dash_distance": "= dash_speed * dash_time",
    "player.revive_iframes": "single player: no revives",
    "player.ex_knockback": "EX recoil not modelled (player freezes during EX)",
    "peashooter.fire_interval_nominal": "the real cadence is fire_interval_frames",
    "peashooter.fire_interval": "= fire_interval_frames / 60",
    "peashooter.dps": "= damage / fire_interval",
    "super.meter_max": "spec.SUPER_METER_MAX = 5 cards",
    "super.ex_cost": "1 card",
    "super.meter_per_card": "sim meter is in cards",
    "super.meter_per_parry": "mapped as super.cards_per_parry",
    "super.damage_per_card": "= 1 / meter_per_damage",
    "hidden.coop_damage_multiplier": "single player",
    "hidden.charm_health_up_weapon_debuff": "not in the RL loadout",
}


def _equal(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=1e-6)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/goopy_constants.json")
    ap.add_argument("--live", default="data/live_hitboxes.json")
    ap.add_argument("--write-overrides", default=None)
    args = ap.parse_args()
    constants = json.loads(Path(args.json).read_text())
    live_path = Path(args.live)
    live = json.loads(live_path.read_text()) if live_path.exists() else None
    cfg = SimConfig()

    bad = 0
    print(f"{'sim field':38s} {'sim default':>22s} {'game':>22s}  json key")
    for path, key, fn in MAPPING:
        if key not in constants:
            print(f"{path:38s} {'':>22s} {'MISSING':>22s}  {key}")
            bad += 1
            continue
        sim, game = get_path(cfg, path), fn(constants[key])
        ok = _equal(sim, game)
        superseded = live is not None and path in LIVE_FIELDS
        bad += not ok and not superseded
        flag = "" if ok else ("   (superseded by live measurement)" if superseded else "   <-- MISMATCH")
        print(f"{path:38s} {str(sim)[:22]:>22s} {str(game)[:22]:>22s}  {key}{flag}")

    if live is not None:
        print(f"\nlive measurements ({live_path}):")
        for path, key, sub, fn in LIVE_MAPPING:
            sim, game = get_path(cfg, path), fn(live_value(live, key, sub))
            ok = _equal(sim, game)
            bad += not ok
            print(f"{path:38s} {str(sim)[:22]:>22s} {str(game)[:22]:>22s}  {key}[{sub}]{'' if ok else '   <-- MISMATCH'}")
        # The per-tick punch/smash tracks must be regenerated from the same file.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from extract_live_tracks import OUT, render

        fresh = render(live) == OUT.read_text()
        bad += not fresh
        print(f"live_tracks.py {'up to date' if fresh else 'STALE: run scripts/extract_live_tracks.py'}")

    print("\nbehavioural flags:")
    for key, expected in EXPECTED_FLAGS.items():
        got = constants.get(key)
        ok = got == expected
        bad += not ok
        print(f"  {key:36s} game={got!s:6s} sim assumes {expected}{'' if ok else '   <-- MISMATCH'}")

    mapped = {k for _, k, _ in MAPPING} | set(EXPECTED_FLAGS)
    unmapped = [k for k in constants if not k.startswith("_meta") and k not in mapped]
    unexplained = [k for k in unmapped if not UNMAPPED_OK.get(k)]
    print(f"\nunmapped json keys: {len(unmapped)} ({len(unexplained)} unexplained)")
    for k in unexplained:
        print(f"  UNEXPLAINED: {k} = {constants[k]}")
    bad += len(unexplained)

    if args.write_overrides:
        Path(args.write_overrides).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_overrides).write_text(json.dumps(game_overrides(constants, live), indent=2))
        print(f"\nwrote {args.write_overrides}")
    print("\nOK: defaults match the game" if bad == 0 else f"\n{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
