"""Player + boss mechanics of the simulator (no gym wrapper), checked against the
decompiled game constants (docs/GAME_CONSTANTS.md)."""

import pytest

from cuphead_rl.sim.config import SimConfig
from cuphead_rl.sim.goopy import parse_pattern
from cuphead_rl.sim.player import FrameInput, Player
from cuphead_rl.sim.world import QuestionMark, World
from cuphead_rl.spec import BossMove, BossPhase, HazardKind


def run_jump(hold_frames: int, cfg: SimConfig) -> tuple[float, float]:
    """(apex, airtime) of a jump with the button held for ``hold_frames`` ticks."""
    p = Player(cfg, 0.0)
    apex = 0.0
    for f in range(200):
        p.step(FrameInput(jump_held=f < hold_frames, jump_pressed=f == 0), cfg.dt, [])
        apex = max(apex, p.y)
        if p.grounded and f > 0:
            return apex, (f + 1) * cfg.dt
    raise AssertionError("never landed")


def test_jump_matches_game_tick_replay():
    cfg = SimConfig()
    tap_apex, tap_air = run_jump(1, cfg)
    full_apex, full_air = run_jump(120, cfg)
    # GAME_CONSTANTS: tap apex 133.5 / 0.433 s, full hold apex 283.3 / 0.667 s.
    assert tap_air == pytest.approx(0.433, abs=0.017)
    assert full_air == pytest.approx(0.667, abs=0.017)
    assert tap_apex == pytest.approx(133.5, abs=0.1)  # dodge_shoot trace: one-tick taps reach 133.52
    assert full_apex == pytest.approx(283.3, abs=0.1)
    mid_apex, _ = run_jump(5, cfg)
    assert tap_apex < mid_apex < full_apex  # height grows with how long jump is held


def test_held_jump_rises_at_jump_speed_minus_one_tick_of_gravity():
    """Bridge measurement (data/check_velocities_after.json, per-tick velocities): with jump held the
    player rises at 1123.2 u/s (HandleFalling runs after HandleJumping's reset), for 10 ticks,
    and the whole arc matches the game tick for tick."""
    import json
    from pathlib import Path

    rows = json.loads((Path(__file__).parents[1] / "data" / "check_velocities_after.json").read_text())["rows"]
    game = {r["tick"]: (r["player_y"], r["player_vy_f1"]) for r in rows if r["tick"] <= 68}
    cfg = SimConfig()
    p = Player(cfg, 0.0)
    for t in range(1, 69):
        k = t - 1  # tools/check_velocities.py: idle 32 ticks, then jump held for 20
        p.step(FrameInput(jump_held=32 <= k < 52, jump_pressed=k == 32), cfg.dt, [])
        if t in game:
            assert p.y == pytest.approx(game[t][0], abs=0.01)
            assert p.vy == pytest.approx(game[t][1], abs=0.01)
    assert game[40][1] == pytest.approx(cfg.player.jump_speed - cfg.player.gravity * cfg.dt, abs=0.01)


def test_ground_and_air_dash_distance():
    cfg = SimConfig()
    p = Player(cfg, -300.0)
    p.step(FrameInput(dash_pressed=True), cfg.dt, [])
    for _ in range(40):
        p.step(FrameInput(), cfg.dt, [])
    assert p.x + 300.0 == pytest.approx(330.0, abs=cfg.player.dash_speed * cfg.dt + 1e-6)

    # Air dash: holds altitude and only one per airtime.
    p = Player(cfg, 0.0)
    p.step(FrameInput(jump_held=True, jump_pressed=True), cfg.dt, [])
    for _ in range(5):
        p.step(FrameInput(jump_held=True), cfg.dt, [])
    p.step(FrameInput(move=-1, dash_pressed=True), cfg.dt, [])
    assert p.dashing and not p.can_air_dash and p.vy == 0.0
    y0 = p.y
    p.step(FrameInput(), cfg.dt, [])
    assert p.y == y0
    while p.dashing:
        p.step(FrameInput(), cfg.dt, [])
    for _ in range(10):  # outlast the input buffer
        p.step(FrameInput(), cfg.dt, [])
    p.step(FrameInput(dash_pressed=True), cfg.dt, [])
    assert not p.dashing  # second air dash refused


def test_dash_press_during_dash_is_buffered():
    cfg = SimConfig()
    p = Player(cfg, -300.0)
    p.step(FrameInput(dash_pressed=True), cfg.dt, [])
    frames = round(cfg.player.dash_time / cfg.dt)
    for i in range(frames - 2):
        p.step(FrameInput(), cfg.dt, [])
    p.step(FrameInput(dash_pressed=True), cfg.dt, [])  # pressed while still dashing
    dashed_again = False
    for _ in range(20):
        p.step(FrameInput(), cfg.dt, [])
        dashed_again |= p.dashing and p.dash_timer > cfg.player.dash_time - 2 * cfg.dt
    # Ground dash cooldown (0.1 s) is shorter than the buffer (0.134 s), so it fires.
    assert dashed_again


def test_peashooter_fires_every_8_ticks_and_cycles_lanes():
    cfg = SimConfig()
    p = Player(cfg, 0.0)
    bullets = []
    for _ in range(8 * 4):
        p.step(FrameInput(shoot=True), cfg.dt, bullets)
    assert len(bullets) == 4
    assert [round(b.y - bullets[0].y) for b in bullets] == [0, 20, 40, 20]
    # No shooting (and no cooldown progress) while dashing.
    p = Player(cfg, 0.0)
    bullets = []
    p.step(FrameInput(shoot=True), cfg.dt, bullets)
    n = len(bullets)
    p.step(FrameInput(shoot=True, dash_pressed=True), cfg.dt, bullets)
    while p.dashing:
        p.step(FrameInput(shoot=True), cfg.dt, bullets)
    assert len(bullets) == n


def test_bullets_travel_and_damage_boss():
    cfg = SimConfig()
    w = World(cfg, seed=0)
    hp0 = w.boss.hp
    w.step(FrameInput(move=1, shoot=True))
    assert len(w.bullets) == 1 and w.boss.hp == hp0  # projectile in flight, no instant hit
    for _ in range(60):
        w.step(FrameInput(shoot=True))
    assert w.boss.hp < hp0
    assert w.damage_dealt == pytest.approx(hp0 - w.boss.hp)
    assert (hp0 - w.boss.hp) % cfg.weapon.pea_damage == pytest.approx(0.0)
    assert w.player.meter == pytest.approx(w.damage_dealt / 160.0)  # 160 damage per card


def test_ex_costs_a_card_and_deals_25():
    cfg = SimConfig()
    cfg.transition.intro_time = 100.0  # keep Goopy still (and hittable) at its start position
    w = World(cfg, seed=0)
    w.player.meter = 1.0
    w.step(FrameInput(move=1, ex_pressed=True))
    assert w.player.meter == 0.0
    for _ in range(round(cfg.player.ex_freeze * 60)):  # the shot leaves after the freeze
        assert not any(b.ex for b in w.bullets)
        w.step(FrameInput())
    assert any(b.ex for b in w.bullets)
    for _ in range(120):
        w.step(FrameInput())
    assert w.damage_dealt == pytest.approx(cfg.weapon.ex_max_damage, abs=0.01)


def test_parse_jump_pattern_quirks():
    pat = parse_pattern("HJ,LJ,D.5, LJ,HL,RJ")
    assert pat == [("H", -1.0), ("L", -1.0), ("L", 0.5), ("H", -1.0), ("R", -1.0)]


def _force_land(w: World, max_frames: int = 600) -> None:
    for _ in range(max_frames):
        w.step(FrameInput())
        if w.boss.state != BossMove.JUMP and w.boss.state != BossMove.INTRO:
            if w.boss.state == BossMove.TRANSFORM:
                return


def test_phase_transitions_and_win():
    cfg = SimConfig()
    w = World(cfg, seed=1)
    b = w.boss
    w.player.invuln = 1e9
    assert b.phase == BossPhase.SMALL and b.max_hp == 1200.0
    b.damage(1200 * (1 - 0.76) + 10)  # past the threshold: wants to transform...
    assert b.phase == BossPhase.SMALL and b.wants_transform  # ...but only on a landing near the centre
    for _ in range(1200):
        w.step(FrameInput())
        if b.state == BossMove.TRANSFORM:
            break
    assert b.phase == BossPhase.BIG and abs(b.x) < cfg.transition.transform_max_abs_x
    assert b.vulnerable and b.damage(10.0) == 10.0  # hittable while transforming (live trace)
    saw_qmarks = False
    for _ in range(int(cfg.transition.grow_time * 60) + 2):
        w.step(FrameInput())
        saw_qmarks |= bool(w.qmarks)
    assert saw_qmarks  # parryable question marks appeared during the transformation
    assert b.state != BossMove.TRANSFORM and b.half_w == pytest.approx(cfg.big.body_radius)
    assert b.jumps_to_punch == cfg.big.first_punch_jumps
    assert not w.qmarks

    hp_before = b.hp
    b.damage(hp_before - 1200 * 0.31 + 20)  # excess damage carries into the tomb phase
    assert b.hp == pytest.approx(1200 * 0.31 - 20)
    for _ in range(600):
        w.step(FrameInput())
        if b.state == BossMove.TRANSFORM:
            break
    assert b.transform == "dying" and b.phase == BossPhase.BIG  # still the big slime on screen
    for _ in range(600):
        w.step(FrameInput())
        if b.phase == BossPhase.TOMB:
            break
    assert b.transform == "falling"  # the phase flips when the tombstone appears
    for _ in range(600):
        w.step(FrameInput())
        if b.state == BossMove.TOMB_MOVE:
            break
    assert b.state == BossMove.TOMB_MOVE and b.vulnerable
    assert abs(w.time) > cfg.tomb.death_to_drop  # waited out the death loop
    assert not b.hazards()  # the tombstone only hurts while smashing

    b.damage(b.hp)
    w.step(FrameInput())
    assert b.state == BossMove.DEAD and w.won and w.done and b.hp == 0.0


def test_slime_lands_after_phase_change_mid_jump():
    """Review finding: Goopy must not freeze mid-air when a phase threshold hits during a hop."""
    cfg = SimConfig()
    w = World(cfg, seed=0, start_phase=BossPhase.BIG)
    w.player.invuln = 1e9
    b = w.boss
    for _ in range(600):
        w.step(FrameInput())
        if b.state == BossMove.JUMP and b.lift > 100:
            break
    b.damage(b.hp - 1200 * 0.31 + 1)
    assert b.state == BossMove.JUMP  # dies on landing, keeps falling meanwhile
    for _ in range(120):
        w.step(FrameInput())
    assert b.lift == 0.0 and b.transform == 'dying'


def test_tomb_smash_targets_player_and_hurts_only_while_smashing():
    cfg = SimConfig()
    w = World(cfg, seed=0, start_phase=BossPhase.TOMB)
    b = w.boss
    w.player.invuln = 1e9
    saw_windup = False
    for _ in range(60 * 12):
        w.step(FrameInput())
        if b.state == BossMove.TOMB_SLAM_WINDUP and not saw_windup:
            saw_windup = True
            target = b.x
            assert any(abs(target - (w.player.x + off)) < 1.0 or abs(target) == cfg.tomb.max_x
                       for off in cfg.tomb.attack_offsets)
            assert not b.hazards()
        if b.state == BossMove.TOMB_SLAM:
            break
    assert saw_windup
    # The live smash: the circle swells to r ~250 and drops through the floor; it only
    # hurts while swollen, and then the tombstone slides again.
    hurt_frames, reached_ground = 0, False
    while b.state in (BossMove.TOMB_SLAM_WINDUP, BossMove.TOMB_SLAM):
        hz = b.hazards()
        if hz:
            hurt_frames += 1
            reached_ground |= any(h.y - h.half_h < 0.0 for h in hz)
        w.step(FrameInput())
    assert 10 <= hurt_frames <= 30 and reached_ground
    assert b.state == BossMove.TOMB_MOVE and not b.hazards()


def test_loss_at_zero_hp():
    w = World(SimConfig(), seed=0)
    for _ in range(3):
        w.player.invuln = 0.0
        w._hit(w.boss.x)
    assert w.player.hp == 0 and w.lost and w.done


def test_contact_damage_iframes_and_knockback():
    cfg = SimConfig()
    w = World(cfg, seed=0)
    w.player.x = w.boss.x - 60  # stand inside Goopy, left of centre
    w.step(FrameInput())
    p = w.player
    assert p.hp == 2 and p.invuln == pytest.approx(cfg.player.hit_invuln, abs=0.02)
    assert p.stun_frames > 0 and p.vx < 0 and p.vy > 0  # knocked away and up
    w.step(FrameInput())
    assert p.hp == 2  # i-frames


def test_hurtbox_shrinks_in_the_air():
    cfg = SimConfig()
    p = Player(cfg, 0.0)
    assert p.box()[3] * 2 == pytest.approx(cfg.player.hurt_h)
    p.step(FrameInput(jump_held=True, jump_pressed=True), cfg.dt, [])
    assert p.box()[3] * 2 == pytest.approx(cfg.player.air_hurt_h)


def test_parry_question_mark_gives_card_once_per_airtime():
    cfg = SimConfig()
    w = World(cfg, seed=0)
    w.qmarks.append(QuestionMark(x=w.player.x, y=150.0))
    w.step(FrameInput(jump_held=True, jump_pressed=True))
    for _ in range(4):
        w.step(FrameInput(jump_held=True))
    w.step(FrameInput(jump_held=True, jump_pressed=True))  # press again in the air = parry
    for _ in range(20):
        w.step(FrameInput())
        if w.parries:
            break
    assert w.parries == 1 and w.player.meter == pytest.approx(cfg.weapon.meter_per_parry)
    assert not w.qmarks


def test_parry_cannot_be_rearmed_in_the_same_airtime():
    cfg = SimConfig()
    p = Player(cfg, 0.0)
    p.step(FrameInput(jump_held=True, jump_pressed=True), cfg.dt, [])
    p.step(FrameInput(jump_pressed=True), cfg.dt, [])
    assert p.parrying
    for _ in range(int(cfg.player.parry_window * 60) + 1):
        p.step(FrameInput(), cfg.dt, [])
    assert not p.parrying
    p.step(FrameInput(jump_pressed=True), cfg.dt, [])
    assert not p.parrying  # second press in the same airtime does nothing


def test_hit_stun_and_ex_freeze_last_exact_frames():
    """Review finding: float timers made these last one frame too long."""
    cfg = SimConfig()
    p = Player(cfg, 0.0)
    p.take_hit(source_x=100.0)
    frames = 0
    while p.stun_frames:
        p.step(FrameInput(), cfg.dt, [])
        frames += 1
    assert frames == round(cfg.player.hit_stun * 60) == 18

    p = Player(cfg, 0.0)
    p.meter = 1.0
    bullets = []
    p.step(FrameInput(ex_pressed=True), cfg.dt, bullets)  # the EX starts this frame
    frames = 0
    while p.ex_frames:
        p.step(FrameInput(move=1), cfg.dt, bullets)
        frames += 1
    assert frames + 1 == round((cfg.player.ex_freeze + cfg.player.ex_knockback_time) * 60) == 34
    p.step(FrameInput(move=1), cfg.dt, bullets)
    assert p.vx > 0  # control is back on the next frame


def test_ground_ex_freezes_then_slides_back_as_in_the_lockstep_logs():
    """LevelPlayerMotor.StartEx/exMove_cr: no control for the whole EX; frozen until the shot
    leaves (OnExFired), then pushed back at exKnockback = 230 u/s. Lockstep logs (4-tick
    decisions, walking right then EX): 0, 0, 0, 0, -15.3 x 4, then 2 slide + 2 walk ticks."""
    cfg = SimConfig()
    p = Player(cfg, 0.0)
    p.meter = 1.0
    bullets = []
    p.step(FrameInput(move=1), cfg.dt, bullets)  # facing right
    x0, per_decision = p.x, []
    for d in range(9):
        for t in range(4):
            p.step(FrameInput(move=1, ex_pressed=d == 0 and t == 0), cfg.dt, bullets)
            if d * 4 + t < 16:
                assert not bullets  # the shot leaves after the freeze
        per_decision.append(round(p.x - x0, 2))
        x0 = p.x
    assert per_decision[:4] == [0.0] * 4
    assert per_decision[4:8] == [pytest.approx(-15.33, abs=0.01)] * 4
    assert per_decision[8] == pytest.approx(-2 * 230 / 60 + 2 * 490 / 60, abs=0.01)  # log: +8.7
    assert len(bullets) == 1 and bullets[0].ex and bullets[0].vx > 0

    p = Player(cfg, 0.0)  # straight up: exDelay_cr, frozen without sliding
    p.meter = 1.0
    p.step(FrameInput(aim=1, ex_pressed=True), cfg.dt, [])
    while p.ex_frames:
        p.step(FrameInput(move=1, aim=1), cfg.dt, [])
    assert p.x == 0.0


def test_dying_big_goopy_hurts_until_crushed():
    """SlimeLevelSlime.OnCollisionPlayer has no state check: the dying slime keeps its
    contact damage until the dropping tombstone passes y_unity 70 and it explodes."""
    cfg = SimConfig()
    w = World(cfg, seed=0, start_phase=BossPhase.BIG)
    w.player.invuln = 1e9
    b = w.boss
    for _ in range(300):
        w.step(FrameInput())
        if b.state == BossMove.IDLE and b.lift == 0.0:
            break
    b.damage(b.hp - 1200 * 0.31 + 1)
    assert b.transform == "dying"
    assert any(h.kind == HazardKind.BODY for h in b.hazards())
    seen_falling_body = False
    while b.state == BossMove.TRANSFORM:
        w.step(FrameInput())
        kinds = [h.kind for h in b.hazards()]
        if b.transform == "falling":
            assert HazardKind.TOMB not in kinds  # the drop itself does not hurt
            if b.crushed_slime_alive:
                seen_falling_body = True
                assert HazardKind.BODY in kinds
            else:
                assert HazardKind.BODY not in kinds
    assert seen_falling_body


def test_tomb_needs_aiming_up():
    """Live data: the resting tombstone circle (r 100 at y 310-340) is above standing shots."""
    cfg = SimConfig()
    for aim, expect_damage in ((0, False), (1, True)):
        w = World(cfg, seed=0, start_phase=BossPhase.TOMB)
        b = w.boss
        w.player.invuln = 1e9
        while b.state != BossMove.TOMB_MOVE:
            w.step(FrameInput())
        hp0 = b.hp
        for _ in range(90):
            w.player.x = b.x - 27.0  # stand under it; an up-shot leaves 27 u ahead of the cup
            w.player.facing = 1
            w.step(FrameInput(shoot=True, aim=aim))
            if b.state != BossMove.TOMB_MOVE:
                break
        assert (b.hp < hp0) == expect_damage


def test_aim_and_duck_shot_directions():
    cfg = SimConfig()
    p = Player(cfg, 0.0)
    bullets = []
    p.step(FrameInput(shoot=True, aim=1), cfg.dt, bullets)  # standing still, aiming up
    b = bullets[-1]
    assert b.vx == 0.0 and b.vy == pytest.approx(cfg.weapon.pea_speed)
    assert (b.x, b.y) == pytest.approx((27.0, 158.0))
    for _ in range(8):
        p.step(FrameInput(move=1, shoot=True, aim=1), cfg.dt, bullets)  # running + up: diagonal
    b = bullets[-1]
    assert b.vx == pytest.approx(b.vy) and b.vx > 0
    assert p.vx > 0  # aiming up does not stop walking
    for _ in range(8):
        p.step(FrameInput(move=-1, shoot=True, aim=-1), cfg.dt, bullets)  # duck: no walking
    b = bullets[-1]
    assert p.ducking and p.vx == 0.0 and p.facing == -1
    assert b.vy == 0.0 and b.vx < 0 and b.y - p.y in (24.0, 44.0, 64.0)
    cx, cy, hw, hh = p.box()
    assert (cy - p.y, 2 * hh) == pytest.approx((cfg.player.duck_hurt_cy, cfg.player.duck_hurt_h))
    # In the air, down shoots straight down.
    p = Player(cfg, 0.0)
    p.step(FrameInput(jump_held=True, jump_pressed=True), cfg.dt, [])
    bullets = []
    p.step(FrameInput(shoot=True, aim=-1, jump_held=True), cfg.dt, bullets)
    assert bullets[-1].vy < 0 and bullets[-1].vx == 0.0 and not p.ducking


def test_punches_replay_live_tracks():
    from cuphead_rl.sim import live_tracks

    cfg = SimConfig()
    for phase, track, windup, reach in ((BossPhase.SMALL, live_tracks.SMALL_PUNCH, 64, 390.0),
                                        (BossPhase.BIG, live_tracks.BIG_PUNCH, 118, 550.0)):
        w = World(cfg, seed=0, start_phase=phase)
        b = w.boss
        w.player.invuln = 1e9
        b.jumps_to_punch = 0
        b.x, b.facing = 150.0, -1
        b.punch_dir = -1  # as decided on a landing with the player on the left
        w.player.x = -300.0
        for _ in range(600):
            w.step(FrameInput())
            if b.state == BossMove.PUNCH_WINDUP:
                break
        windup_frames = 0
        while b.state == BossMove.PUNCH_WINDUP:
            w.step(FrameInput())
            windup_frames += 1
        assert windup_frames == windup  # ticks 0..windup-1 of the live timeline
        furthest = 0.0
        while b.state == BossMove.PUNCH:
            shapes = [(b.body()[0], b.body()[2])] + [(h.x, h.half_w) for h in b.hazards()]
            furthest = max(furthest, max(150.0 - x for x, _ in shapes))
            w.step(FrameInput())
        assert furthest == pytest.approx(reach, abs=1.0)
        assert b.state == BossMove.IDLE



# --------------------------------------------------------------------------------------
# Contract semantics (spec.py d435b30): one test per rule.
# --------------------------------------------------------------------------------------
def _run_until(w, cond, frames=3000, inp=None):
    for _ in range(frames):
        if cond():
            return True
        w.step(inp or FrameInput())
    return cond()


def test_contract_hazards_are_damaging_or_parryable():
    """Every hazard either damages the player this tick or is parryable."""
    from cuphead_rl.spec import HazardKind as K

    cfg = SimConfig()
    for phase in (BossPhase.SMALL, BossPhase.BIG, BossPhase.TOMB):
        w = World(cfg, seed=3, start_phase=phase)
        w.player.invuln = 1e9
        for _ in range(1500):
            w.step(FrameInput())
            for h in w.hazards():
                assert h.parryable == (h.kind == K.OTHER)
                if not h.parryable:
                    assert h.kind in (K.BODY, K.FIST, K.TOMB)


def test_contract_tomb_hazard_only_while_slam_damages():
    from cuphead_rl.spec import HazardKind as K

    w = World(SimConfig(), seed=0, start_phase=BossPhase.TOMB)
    w.player.invuln = 1e9
    b = w.boss
    seen_tomb = 0
    for _ in range(60 * 15):
        w.step(FrameInput())
        tomb = [h for h in w.hazards() if h.kind == K.TOMB]
        assert bool(tomb) == b.smash_damage
        assert not tomb or b.state == BossMove.TOMB_SLAM
        seen_tomb += bool(tomb)
    assert seen_tomb > 0


def test_tomb_smash_damage_window_matches_the_trace():
    """stand_shoot trace: the TOMB hazard (dealDamage) is on from 40 to 57 ticks after the
    smash anticipation starts, in both smashes. It used to follow the circle radius (37-59)."""
    import json
    from pathlib import Path

    rows = [json.loads(s) for s in (Path(__file__).parents[1] / "data" / "traces" / "stand_shoot.jsonl").open()][1:]
    windows, start, on = [], None, []
    for r in rows:
        st = r["state"]["boss"]["state"]
        if st == BossMove.TOMB_SLAM_WINDUP and start is None:
            start = r["tick"]
        if start is not None and any(h["kind"] == HazardKind.TOMB for h in r["state"]["hazards"]):
            on.append(r["tick"] - start)
        if start is not None and st not in (BossMove.TOMB_SLAM_WINDUP, BossMove.TOMB_SLAM):
            windows.append((on[0], on[-1] + 1))
            start, on = None, []
    assert windows and set(windows) == {(40, 58)}

    w = World(SimConfig(), seed=0, start_phase=BossPhase.TOMB)
    w.player.invuln = 1e9
    b = w.boss
    assert _run_until(w, lambda: b.state == BossMove.TOMB_SLAM_WINDUP, frames=60 * 15)
    sim_on = []
    while b.track is not None:
        if any(h.kind == HazardKind.TOMB for h in w.hazards()):
            sim_on.append(b.track_tick)
        w.step(FrameInput())
    assert (sim_on[0], sim_on[-1] + 1) == windows[0] and len(sim_on) == 18


def test_contract_question_marks_are_other_and_parryable():
    from cuphead_rl.spec import HazardKind as K

    w = World(SimConfig(), seed=1, start_phase=BossPhase.SMALL)
    w.player.invuln = 1e9
    b = w.boss
    b.damage(1200 * 0.25)
    assert _run_until(w, lambda: bool(w.qmarks))
    qs = [h for h in w.hazards() if h.parryable]
    assert len(qs) == 3 and all(h.kind == K.OTHER for h in qs)


def test_contract_phase_is_the_visible_entity():
    """Phase flips at transform start (small->big) and when the tomb appears, not at the
    HP threshold."""
    w = World(SimConfig(), seed=1)
    w.player.invuln = 1e9
    b = w.boss
    b.damage(1200 * 0.25)  # below 76 %
    assert b.phase == BossPhase.SMALL  # still small until it lands near the centre
    assert _run_until(w, lambda: b.state == BossMove.TRANSFORM)
    assert b.phase == BossPhase.BIG
    assert _run_until(w, lambda: b.state == BossMove.IDLE)
    b.damage(b.hp - 1200 * 0.30)  # below 31 %
    assert _run_until(w, lambda: b.transform == "dying")
    assert b.phase == BossPhase.BIG  # big Goopy is still what you see
    assert _run_until(w, lambda: b.transform == "falling")
    assert b.phase == BossPhase.TOMB


def test_contract_time_counts_from_episode_start():
    from cuphead_rl.sim import GoopyEnv

    env = GoopyEnv(start_phase=2)
    _, info = env.reset(seed=0)
    assert info["time"] == 0.0
    _, _, _, _, info = env.step([1, 0, 0, 0, 0, 1])
    assert info["time"] == pytest.approx(4 / 60)


def test_contract_body_hazard_while_dying_and_damage_during_transforms():
    from cuphead_rl.spec import HazardKind as K

    w = World(SimConfig(), seed=0, start_phase=BossPhase.BIG)
    w.player.invuln = 1e9
    b = w.boss
    assert _run_until(w, lambda: b.state == BossMove.IDLE)
    b.damage(b.hp - 1200 * 0.30)
    assert _run_until(w, lambda: b.transform == "dying")
    assert any(h.kind == K.BODY for h in w.hazards())
    hp = b.hp
    assert b.vulnerable and b.damage(4.0) == 4.0 and b.hp == hp - 4.0  # hittable while dying
    assert _run_until(w, lambda: b.transform == "falling")
    assert b.damage(4.0) == 4.0  # and the dropping tombstone too


def test_contract_lunge_circle_is_a_fist():
    from cuphead_rl.spec import HazardKind as K

    w = World(SimConfig(), seed=0)
    w.player.invuln = 1e9
    b = w.boss
    b.jumps_to_punch = 0
    b.x = 150.0
    w.player.x = -300.0
    assert _run_until(w, lambda: b.state == BossMove.PUNCH)
    kinds = {h.kind for h in w.hazards()}
    assert K.FIST in kinds and K.BODY not in kinds


def test_punch_track_reports_the_body_velocity_the_game_shows():
    """While a punch track plays, the boss and its body hazard move at the per-tick centre
    difference, as the bridge measures them (was 0: the policy never saw the lunge coming
    in the sim, but saw ~4800 u/s in the game and stopped ducking; docs/SIM_VS_GAME.md)."""
    from cuphead_rl.sim import live_tracks

    cfg = SimConfig()
    w = World(cfg, seed=0)
    w.player.invuln = 1e9
    b = w.boss
    b.jumps_to_punch = 0
    b.x, b.facing = 150.0, -1
    b.punch_dir = -1
    w.player.x = -300.0
    assert _run_until(w, lambda: b.state == BossMove.PUNCH)
    trk = live_tracks.SMALL_PUNCH
    seen = {}
    while b.track is not None and b.track_tick < 72:
        i = b.track_tick
        s = w.state()
        (dx0, cy0, _), (dx1, cy1, _) = trk[i - 1][1], trk[i][1]
        want = (-(dx1 - dx0) / cfg.dt, (cy1 - cy0) / cfg.dt)
        if want[0] ** 2 + want[1] ** 2 > 6000.0 ** 2:
            want = (0.0, 0.0)  # the lunge's jump: a teleport for the bridge
        assert (s.boss.vx, s.boss.vy) == pytest.approx(want, abs=1e-6)
        lunge = [h for h in s.hazards if h.half_w == s.boss.half_w and h.x == s.boss.x]
        assert (lunge[0].vx, lunge[0].vy) == pytest.approx(want, abs=1e-6)
        seen[i] = want
        w.step(FrameInput())
    assert seen[65] == pytest.approx((-4800.0, 1440.0))  # first PUNCH frame, toward -x
    assert seen[69] == (0.0, 0.0)
    # Outside tracks the boss velocity is the jump's, as before.
    assert _run_until(w, lambda: b.state == BossMove.JUMP)
    assert w.state().boss.vx == b.vx != 0.0


# --------------------------------------------------------------------------------------
# Timing fixes from the sim-vs-game comparison (docs/SIM_VS_GAME.md).
# --------------------------------------------------------------------------------------
def test_ground_time_includes_squish_anims():
    """Landing -> next take-off = squish start + ground delay + squish end: 34 ticks for
    small Goopy and 36 for big Goopy on the live trace."""
    for phase, ticks in ((BossPhase.SMALL, 34), (BossPhase.BIG, 36)):
        cfg = SimConfig()
        cfg.small.jump_pattern = cfg.big.jump_pattern = "HJ"  # no D<x> delays
        w = World(cfg, seed=0, start_phase=phase)
        w.player.invuln = 1e9
        b = w.boss
        b.jumps_to_punch = 99
        assert _run_until(w, lambda: b.state == BossMove.JUMP)
        assert _run_until(w, lambda: b.state == BossMove.IDLE)
        n = 0
        while b.state == BossMove.IDLE:
            w.step(FrameInput())
            n += 1
        assert abs(n - ticks) <= 1


def test_pattern_delay_applies_before_the_following_jump():
    cfg = SimConfig()
    cfg.small.jump_pattern = "D1.0, HJ"
    w = World(cfg, seed=0)
    w.player.invuln = 1e9
    b = w.boss
    b.jumps_to_punch = 99
    b.pattern_i, b.skip_delay = 0, False  # jump_cr started on the D token
    assert _run_until(w, lambda: b.state == BossMove.JUMP)
    assert b.state_time == 0.0 and w.time == pytest.approx(cfg.transition.intro_time + 1.0 + cfg.small.squish_time, abs=0.02)


def test_pattern_start_is_drawn_over_raw_tokens():
    """jump_cr: i = Random.Range(0, pattern.Length) over the raw tokens, D included; starting
    right after a D skips its delay."""
    from cuphead_rl.sim.goopy import pattern_starts

    assert pattern_starts("HJ,LJ,D.5, LJ,HL") == [(0, False), (1, False), (2, False), (2, True), (3, False)]


def test_big_goopy_dies_at_once_when_punching():
    """DeathTransform on the ground calls Die, which stops the punch coroutine."""
    w = World(SimConfig(), seed=0, start_phase=BossPhase.BIG)
    w.player.invuln = 1e9
    b = w.boss
    b.jumps_to_punch = 0
    b.punches_done = 1
    assert _run_until(w, lambda: b.state == BossMove.PUNCH_WINDUP)
    b.damage(b.hp - 1200 * 0.30)
    assert b.state == BossMove.TRANSFORM and b.transform == "dying" and b.track is None


def test_small_turn_punch_sinks_out_of_reach():
    """Punch_Start_Turn: 23 ticks without a collider (no hazard, no damage taken)."""
    from cuphead_rl.sim import live_tracks

    cfg = SimConfig()
    w = World(cfg, seed=0)
    w.player.invuln = 1e9
    b = w.boss
    assert _run_until(w, lambda: b.state == BossMove.IDLE and b.lift == 0.0 and b.state_time == 0.0)
    b.jumps_to_punch = 0
    b.x, b.facing = -100.0, 1
    w.player.x = -400.0  # behind: Goopy turns to punch
    assert _run_until(w, lambda: b.state == BossMove.PUNCH_WINDUP)
    gone = 0
    while b.state == BossMove.PUNCH_WINDUP:
        if not b.vulnerable:
            gone += 1
            assert not [h for h in b.hazards() if h.kind == HazardKind.BODY]
            assert b.damage(4.0) == 0.0
        w.step(FrameInput())
    assert gone == cfg.small.punch_turn_frames
    assert len(b.track) == len(live_tracks.SMALL_PUNCH) + gone


def test_punch_end_counts_as_idle_and_later_big_punches_skip_the_blink():
    """Big windups (trace): first punch 64 blink ticks (75 when turning), later ones the
    shorter Punch_Start (18 when turning); a turn keeps big Goopy's body and hittability."""
    from cuphead_rl.sim import live_tracks

    cfg = SimConfig()
    w = World(cfg, seed=0, start_phase=BossPhase.BIG)
    w.player.invuln = 1e9
    b = w.boss
    lens = []
    for turn in (False, False, True):
        b._begin(BossMove.IDLE)
        b.x, b.facing = 0.0, 1
        b.punch_dir = -1 if turn else 1
        assert _run_until(w, lambda: b.state == BossMove.PUNCH_WINDUP)
        lens.append(len(b.track))
        while b.state == BossMove.PUNCH_WINDUP:
            assert b.vulnerable and b.half_w == cfg.big.body_radius
            w.step(FrameInput())
        assert _run_until(w, lambda: b.state == BossMove.IDLE)
        assert b.track is not None  # the Punch_End tail still plays
    full = len(live_tracks.BIG_PUNCH)
    assert lens == [full, full - cfg.big.repeat_punch_skip, full - cfg.big.repeat_punch_skip + cfg.big.punch_turn_frames]
    windup_later_turn = live_tracks.BIG_PUNCH_WINDUP_TICKS - cfg.big.repeat_punch_skip + cfg.big.punch_turn_frames
    assert windup_later_turn - 13 - 41 == 18  # squish 13, pre-hold 41 around Punch_Start_Turn


def test_tomb_smashes_only_after_finishing_a_leg():
    cfg = SimConfig()
    w = World(cfg, seed=4, start_phase=BossPhase.TOMB)
    w.player.invuln = 1e9
    b = w.boss
    for _ in range(60 * 30):
        w.player.x = 0.0
        prev = b.state
        armed_mid_leg = b.state == BossMove.TOMB_MOVE and b.wants_smash and b.target_x is None
        w.step(FrameInput())
        if prev == BossMove.TOMB_MOVE and b.state == BossMove.TOMB_SLAM_WINDUP:
            assert not armed_mid_leg or b.target_x is not None
        if armed_mid_leg and b.state == BossMove.TOMB_MOVE and b.leg_time > 0.0:
            assert abs(b.leg_to) == cfg.tomb.max_x  # keeps sliding to the wall
    # Wall-to-wall leg with the turn pause: 1.4 s on the live trace.
    leg = cfg.tomb.turn_hold + 2 * cfg.tomb.max_x / cfg.tomb.move_speed
    assert leg == pytest.approx(1.4, abs=0.05)


def test_tomb_cannot_be_hurt_underground_in_its_smash():
    cfg = SimConfig()
    w = World(cfg, seed=0, start_phase=BossPhase.TOMB)
    w.player.invuln = 1e9
    b = w.boss
    assert _run_until(w, lambda: b.state == BossMove.TOMB_SLAM_WINDUP, frames=60 * 20)
    off = []
    while b.state in (BossMove.TOMB_SLAM_WINDUP, BossMove.TOMB_SLAM):
        off.append(not b.vulnerable)
        w.step(FrameInput())
    lo, hi = cfg.tomb.smash_invuln_ticks
    assert [i for i, o in enumerate(off) if o] == list(range(lo, hi))


def test_shots_hit_big_goopy_while_the_tomb_drops():
    w = World(SimConfig(), seed=0, start_phase=BossPhase.BIG)
    w.player.invuln = 1e9
    b = w.boss
    assert _run_until(w, lambda: b.state == BossMove.IDLE)
    b.damage(b.hp - 1200 * 0.30)
    assert _run_until(w, lambda: b.transform == "falling")
    hb = b.hit_body()
    assert b.crushed_slime_alive and hb[2] == SimConfig().big.body_radius and hb[1] < b.y


def test_punch_position_matches_the_game_traces():
    """inPunchPosition: right inside (min, max), left inside (-max, -min). Both small
    punches of the re-recorded traces: stand_shoot tick 631 (Goopy 393, player -410) and
    dodge_shoot tick 547 (Goopy -560, player -462)."""
    for x, px, d in ((393.0, -410.0, -1), (-560.0, -462.0, 1), (300.0, 500.0, 0), (-393.0, -410.0, 0)):
        w = World(SimConfig(), seed=0)
        b = w.boss
        b._begin(BossMove.IDLE)
        b.jumps_to_punch, b.x = 0, x
        b._on_landing(px)
        assert b.punch_dir == d, (x, px)


def test_punch_is_decided_on_the_landing_frame_only():
    """The player crossing sides while Goopy stands on the ground doesn't trigger a punch."""
    w = World(SimConfig(), seed=0)
    w.player.invuln = 1e9
    b = w.boss
    b._begin(BossMove.IDLE)
    b.jumps_to_punch, b.x = 0, 300.0
    b._on_landing(500.0)  # player on the right: 300 is outside (-1000, 200)
    assert b.punch_dir == 0
    w.player.x = -500.0  # now on the left, where (-200, 1000) would allow the punch
    for _ in range(120):
        w.step(FrameInput())
        assert b.state not in (BossMove.PUNCH_WINDUP, BossMove.PUNCH)
        if b.state == BossMove.JUMP:
            break
    assert b.state == BossMove.JUMP


def test_state_reports_the_body_after_a_hit_cuts_a_punch_short():
    """A pea that drops big Goopy below 31 % mid-punch ends the punch the same step: the
    exported body must be the slime body, not the punch frame."""
    from cuphead_rl.sim.player import Bullet

    w = World(SimConfig(), seed=0, start_phase=BossPhase.BIG)
    w.player.invuln = 1e9
    b = w.boss
    b._begin(BossMove.IDLE)
    b.jumps_to_punch, b.x, b.facing, b.punch_dir = 0, 150.0, -1, -1
    w.player.x = -300.0
    assert _run_until(w, lambda: b.state == BossMove.PUNCH)
    # A punch frame whose body sits elsewhere than the resting slime (the live big punch
    # keeps the body in place, which would hide a stale geometry).
    b.track, b.track_tick = ((False, (100.0, 250.0, 60.0), ()),) * 20, 0
    b.hp = 1200 * 0.31 + 2.0
    bx, by, *_ = b.body()
    w.bullets.append(Bullet(bx, by, 0.0, 5.0, 5.0, 4.0))
    w.step(FrameInput())
    assert b.transform == "dying" and b.track is None
    s = w.state().boss
    assert (s.x, s.y, s.half_w) == pytest.approx(b.body()[:3])
