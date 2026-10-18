import numpy as np

from cuphead_rl import spec
from cuphead_rl.spec import BossInfo, BossMove, BossPhase, GameState, Hazard, HazardKind, PlayerState


def make_state(**kw) -> GameState:
    s = GameState(
        player=PlayerState(x=-300.0, y=0.0),
        boss=BossInfo(phase=BossPhase.SMALL, state=BossMove.IDLE, x=300.0, y=60.0,
                      half_w=60.0, half_h=60.0, hp=1000.0, max_hp=1000.0),
    )
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_observation_shape_and_padding():
    s = make_state(hazards=[Hazard(HazardKind.FIST, x=0.0, y=40.0, half_w=50, half_h=20, vx=-900)])
    obs = spec.encode_observation(s)
    assert obs.shape == (spec.OBS_SIZE,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_hazards_sorted_nearest_first():
    far = Hazard(HazardKind.BODY, x=500.0, y=0.0, half_w=10, half_h=10)
    near = Hazard(HazardKind.FIST, x=-250.0, y=0.0, half_w=10, half_h=10)
    obs = spec.encode_observation(make_state(hazards=[far, near]))
    base = spec.PLAYER_FEATURES + spec.BOSS_FEATURES
    assert obs[base] == 1.0
    assert np.isclose(obs[base + 1], 50.0 / spec.ARENA_HALF_WIDTH)


def test_reward_damage_hit_and_win():
    prev = make_state()
    curr = make_state(damage_dealt=10.0)  # 1 % of 1000 HP
    curr.player.hp = 2
    r = spec.compute_reward(prev, curr)
    cfg = spec.RewardConfig()
    assert np.isclose(r, cfg.damage + cfg.hit_taken + cfg.time)
    won = make_state(won=True)
    assert spec.compute_reward(prev, won) > 40


def test_decode_action():
    a = spec.decode_action([0, 1, 0, 1, 1, 2])
    assert a.move == -1 and a.jump and not a.dash and a.shoot and a.ex and a.aim == 1
    assert spec.decode_action([1, 0, 0, 0, 0, 0]).aim == -1
    assert len(spec.ACTION_NVEC) == 6
