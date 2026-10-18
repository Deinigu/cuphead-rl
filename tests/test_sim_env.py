"""GoopyEnv API, determinism, curricula, config IO, rendering and video."""

import json
import pickle

import numpy as np
import pytest

from cuphead_rl import spec
from cuphead_rl.sim import GoopyEnv, SimConfig
from cuphead_rl.sim.policies import heuristic_policy, random_policy


def rollout(env: GoopyEnv, seed: int, n: int = 300):
    rng = np.random.default_rng(123)
    obs, _ = env.reset(seed=seed)
    out = [obs]
    rewards = []
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(rng.integers(0, spec.ACTION_NVEC))
        out.append(obs)
        rewards.append(r)
        if term or trunc:
            break
    return np.stack(out), np.asarray(rewards)


def test_spaces_and_obs_shape():
    env = GoopyEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (spec.OBS_SIZE,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert tuple(env.action_space.nvec) == spec.ACTION_NVEC
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    for key in ("phase", "boss_hp_frac", "player_hp", "won", "damage_dealt", "time", "parries"):
        assert key in info
    assert info["time"] == pytest.approx(4 / 60)


@pytest.mark.parametrize("randomize", [False, True])
def test_determinism_same_seed_same_actions(randomize):
    a = rollout(GoopyEnv(randomize=randomize), seed=7)
    b = rollout(GoopyEnv(randomize=randomize), seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_different_seeds_differ():
    env = GoopyEnv()
    seqs = set()
    for seed in range(6):
        env.reset(seed=seed)
        for _ in range(150):
            env.step([1, 0, 0, 0, 0, 1])
        seqs.add(round(env.world.boss.x, 3))
    assert len(seqs) > 1


def test_start_phase_constructor_and_options():
    t = SimConfig().transition
    env = GoopyEnv(start_phase=1)
    _, info = env.reset(seed=0)
    assert info["phase"] == 1
    assert info["boss_hp_frac"] == pytest.approx(t.big_trigger_frac)
    _, info = env.reset(seed=0, options={"start_phase": 2})
    assert info["phase"] == 2
    assert info["boss_hp_frac"] == pytest.approx(t.tomb_trigger_frac)
    _, info = env.reset(seed=0, options={"start_phase": 0})
    assert info["phase"] == 0 and info["boss_hp_frac"] == 1.0


def test_win_terminates_with_win_reward():
    env = GoopyEnv(start_phase=2)
    env.reset(seed=0)
    while env.world.boss.state != spec.BossMove.TOMB_MOVE:
        env.step([1, 0, 0, 0, 0, 1])
    env.world.boss.damage(env.world.boss.hp)
    obs, r, term, trunc, info = env.step([1, 0, 0, 0, 0, 1])
    assert term and not trunc and info["won"]
    assert r >= spec.RewardConfig().win + spec.RewardConfig().time - 1e-9  # damage was applied outside a step


def test_env_rejects_episodes_longer_than_spec():
    with pytest.raises(AssertionError):
        GoopyEnv(max_seconds=spec.MAX_EPISODE_SECONDS + 1)


def test_loss_terminates_with_loss_reward():
    env = GoopyEnv()
    env.reset(seed=0)
    w = env.world
    w.player.hp = 1
    w.player.x = w.boss.x  # walk into Goopy
    w.player.invuln = 0.0
    obs, r, term, trunc, info = env.step([1, 0, 0, 0, 0, 1])
    assert term and info["player_hp"] == 0 and not info["won"]
    assert r < spec.RewardConfig().loss


def test_truncation_at_max_seconds():
    env = GoopyEnv(max_seconds=1.0)
    env.reset(seed=0)
    env.world.player.invuln = 1e9  # can't die
    trunc = False
    for _ in range(100):
        _, _, term, trunc, info = env.step([1, 0, 0, 0, 0, 1])
        if term or trunc:
            break
    assert trunc and info["time"] == pytest.approx(1.0, abs=4 / 60)


def test_config_json_roundtrip_and_partial_override(tmp_path):
    cfg = SimConfig()
    path = tmp_path / "full.json"
    cfg.to_json(path)
    assert SimConfig.from_json(path) == cfg
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"total_hp": 1000, "small": {"max_x": 500}, "player": {"run_speed": 500.5},
                                   "tomb": {"attack_offsets": [0, 10]}}))
    c2 = SimConfig.from_json(partial)
    assert c2.total_hp == 1000.0 and c2.small.max_x == 500.0 and c2.player.run_speed == 500.5
    assert c2.tomb.attack_offsets == (0.0, 10.0) and c2.big == cfg.big
    with pytest.raises(KeyError):
        SimConfig.from_dict({"small": {"not_a_field": 1}})


def test_randomized_config_jitters_but_keeps_hp():
    cfg = SimConfig()
    r = cfg.randomized(np.random.default_rng(0), 0.15)
    assert r.total_hp == cfg.total_hp and r.transition.big_trigger_frac == cfg.transition.big_trigger_frac
    assert r.small.max_x == cfg.small.max_x and r.small.jump_pattern == cfg.small.jump_pattern
    assert r.small.punch_pre_hold != cfg.small.punch_pre_hold
    assert 0.85 <= r.tomb.move_speed / cfg.tomb.move_speed <= 1.15
    assert r.player == cfg.player


def test_env_is_picklable_before_render():
    env = GoopyEnv()
    env.reset(seed=0)
    clone = pickle.loads(pickle.dumps(env))
    assert clone.world.boss.hp == env.world.boss.hp


def test_random_policy_rarely_gets_far_and_heuristic_makes_progress():
    env = GoopyEnv()
    obs, _ = env.reset(seed=3)
    while True:
        obs, _, term, trunc, info = env.step(random_policy(obs, np.random.default_rng(0)))
        if term or trunc:
            break
    assert not info["won"]
    obs, _ = env.reset(seed=3, options={"start_phase": 2})
    while True:
        obs, _, term, trunc, info = env.step(heuristic_policy(obs))
        if term or trunc:
            break
    assert info["damage_dealt"] > 50


def test_render_frame_and_overlay():
    env = GoopyEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step([2, 1, 0, 1, 0, 1])
    frame = env.render({"experiment": "test", "value": 1.234, "probs": [0.1, 0.5, 0.4]})
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
    assert frame.std() > 10  # not a blank image
    env._renderer.show_hitboxes = True
    env.render()


def test_render_character_is_cosmetic():
    """Mugman (default) and Cuphead differ only in looks: same frame size, same state."""
    frames = []
    for ch in ("mugman", "cuphead"):
        env = GoopyEnv(render_mode="rgb_array", character=ch)
        env.reset(seed=0)
        env.step([2, 0, 0, 1, 0, 1])
        frames.append(env.render())
    assert GoopyEnv().character == "mugman"
    assert frames[0].shape == frames[1].shape and (frames[0] != frames[1]).any()
    with pytest.raises(ValueError):
        GoopyEnv(render_mode="rgb_array", character="elder_kettle").render()


def test_render_structured_agent_overlay():
    """The training pipeline's AgentPolicy.overlay() dict renders without errors."""
    env = GoopyEnv()
    env.reset(seed=0)
    overlay = {
        "title": "ppo_baseline | 12.50M steps | first_win | a very long experiment name that must shrink",
        "value": 87.5,  # outside the gauge range: clipped
        "action": "> JUMP SHOOT",
        "probs": {"move": [0.2, 0.1, 0.7], "jump": [0.4, 0.6], "dash": [0.9, 0.1],
                  "shoot": [0.05, 0.95], "ex": [1.0, 0.0], "aim": [0.1, 0.3, 0.6]},
        "seed": 3,
    }
    frame = env.render(overlay)
    assert frame.shape == (720, 1280, 3)
    env.render({"value": -50.0, "probs": {"move": np.array([0.3, 0.3, 0.4])}})
    env.close()


def test_record_episode_writes_mp4(tmp_path):
    from cuphead_rl.sim.video import record_episode

    env = GoopyEnv(frame_skip=4)
    calls = []
    path = tmp_path / "ep.mp4"
    stats = record_episode(env, lambda o: np.array([1, 0, 0, 1, 0, 1]), path, fps=60,
                           overlay_fn=lambda: calls.append(1) or {"n": len(calls)}, max_seconds=0.5, seed=0)
    assert path.exists() and path.stat().st_size > 1000
    # every simulation frame is rendered (+1 reset frame, +1 s held on the last frame)
    assert stats["frames"] == pytest.approx(0.5 * 60 + 1, abs=4)
    assert len(calls) >= stats["frames"]
    for key in ("won", "phase", "boss_hp_frac", "player_hp", "time", "total_reward"):
        assert key in stats
    assert env.frame_callback is None


def test_randomized_scale_bounds_and_pairs():
    cfg = SimConfig()
    with pytest.raises(ValueError):
        cfg.randomized(np.random.default_rng(0), 0.5)
    for seed in range(200):
        r = cfg.randomized(np.random.default_rng(seed), 0.49)
        assert r.tomb.attack_delay_min <= r.tomb.attack_delay_max
        assert r.big.body_radius > r.small.body_radius
        t = r.transition  # the transformation timeline scales as one
        assert t.grow_start < t.grow_end <= t.grow_time and t.qmark_on < t.qmark_off <= t.grow_time
        assert t.grow_end / t.grow_time == pytest.approx(cfg.transition.grow_end / cfg.transition.grow_time)


def test_fast_action_decode_matches_spec():
    from cuphead_rl.sim.env import _decode

    for a in np.ndindex(*spec.ACTION_NVEC):
        arr = np.array(a, dtype=np.int64)
        assert _decode(arr) == spec.decode_action(arr)
    assert _decode([2, 1, 0, 1, 0, 1]) == spec.decode_action([2, 1, 0, 1, 0, 1])
