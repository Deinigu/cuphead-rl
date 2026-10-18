"""End-to-end smoke test of the training pipeline against a tiny fake env (no simulator needed)."""

import json

import gymnasium as gym
import numpy as np
import pytest

from cuphead_rl import spec
from cuphead_rl.train import callbacks, envs, evaluate, train
from cuphead_rl.train.agent import AgentPolicy, load_model
from cuphead_rl.train.config import ExperimentConfig, PPOConfig


class FakeEnv(gym.Env):
    """Reward for pressing RIGHT; 'wins' after 30 steps of doing so."""

    observation_space = gym.spaces.Box(-5, 5, (spec.OBS_SIZE,), np.float32)
    action_space = gym.spaces.MultiDiscrete(spec.ACTION_NVEC)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.phase = (options or {}).get("start_phase", 0)
        self.t, self.progress = 0, 0
        return np.zeros(spec.OBS_SIZE, np.float32), self._info()

    def _info(self):
        return dict(won=self.progress >= 30, phase=self.phase, boss_hp_frac=1 - self.progress / 30,
                    player_hp=3, damage_dealt=self.progress, time=self.t / 15, parries=0)

    def step(self, action):
        self.t += 1
        good = int(action[0]) == 2
        self.progress += good
        obs = np.full(spec.OBS_SIZE, self.progress / 30, np.float32)
        info = self._info()
        return obs, float(good), info["won"], self.t >= 60, info


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "make_base_env", lambda cfg, **kw: FakeEnv())
    monkeypatch.setattr(evaluate, "make_base_env", lambda cfg, **kw: FakeEnv())
    monkeypatch.setattr(callbacks, "record_video", lambda *a, **kw: {"stub": True})
    return ExperimentConfig(name="t", n_envs=1, total_timesteps=2048, eval_every=1024, eval_episodes=2,
                            out_dir=str(tmp_path), ppo=PPOConfig(n_steps=256, batch_size=64))


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nppo:\n  ent_coef: 0.02\n")
    c = ExperimentConfig.load(p, ["ppo.gamma=0.9", "curriculum.enabled=true"])
    assert c.ppo.ent_coef == 0.02 and c.ppo.gamma == 0.9 and c.curriculum.enabled
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"nope": 1})


def test_train_eval_roundtrip(cfg):
    venv = envs.make_vec_env(cfg)
    model = train.build_model(cfg, venv)
    cb = callbacks.EvalVideoCallback(cfg)
    model.learn(cfg.total_timesteps, callback=cb)
    venv.close()
    lines = (cfg.run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 2 and "win_rate" in json.loads(lines[-1])
    assert (cfg.run_dir / "best.zip").exists()
    assert (cfg.run_dir / "videos" / "index.jsonl").exists()  # "untrained" video always recorded

    policy = AgentPolicy(load_model(cfg.run_dir / "best.zip"))
    stats = evaluate.evaluate(policy, cfg, episodes=2)
    assert 0.0 <= stats["win_rate"] <= 1.0
    ov = policy.overlay("t")
    assert "value" in ov and len(ov["probs"]["move"]) == 3 and len(ov["probs"]["aim"]) == 3


def test_frame_stack_policy(cfg):
    cfg.frame_stack = 3
    venv = envs.make_vec_env(cfg)
    model = train.build_model(cfg, venv)
    policy = AgentPolicy(model, frame_stack=3)
    a = policy(np.zeros(spec.OBS_SIZE, np.float32))
    assert a.shape == (len(spec.ACTION_NVEC),)
    venv.close()


def test_adaptive_curriculum_promotes():
    cur = callbacks.AdaptiveCurriculum(promote_at=0.5)
    assert cur.stage == 2 and cur.probs == [0.0, 0.0, 1.0]

    class V:
        def env_method(self, name, probs):
            self.called = (name, probs)

    v = V()
    cur.update({"win_rate": 0.1, "curriculum": {"win_rate": 0.8}}, v)
    assert cur.stage == 1 and v.called[0] == "set_probs"
    assert np.isclose(sum(cur.probs), 1.0) and cur.probs[1] == 0.7


def test_phase_curriculum_wrapper():
    w = envs.PhaseCurriculum(FakeEnv(), [0, 0, 1])
    w.reset()
    assert w.env.phase == 2
    w.set_probs([1, 0, 0])
    w.reset()
    assert w.env.phase == 0


def test_init_from_uses_new_config_and_env_count(cfg):
    venv = envs.make_vec_env(cfg)
    model = train.build_model(cfg, venv)
    model.save(cfg.run_dir / "seed.zip")
    venv.close()
    cfg2 = ExperimentConfig(name="ft", n_envs=2, out_dir=cfg.out_dir, init_from=str(cfg.run_dir / "seed.zip"),
                            ppo=PPOConfig(n_steps=128, batch_size=64, learning_rate=1e-5, ent_coef=0.0))
    venv2 = envs.make_vec_env(cfg2)
    ft = train.build_model(cfg2, venv2)
    assert ft.n_envs == 2 and ft.n_steps == 128 and ft.ent_coef == 0.0
    assert ft.lr_schedule(1.0) == 1e-5
    ft.learn(256)
    venv2.close()


def test_resume_continues_timeline(cfg, tmp_path, monkeypatch):
    import sys
    import yaml

    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump({**cfg.to_dict(), "name": "res", "total_timesteps": 1024}))
    monkeypatch.setattr(sys, "argv", ["train", str(cfg_path)])
    train.main()
    run = cfg.run_dir.parent / "res"
    first = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert first[0]["steps"] < 1024

    monkeypatch.setattr(sys, "argv", ["train", "--resume", str(run), "-o", "total_timesteps=3072"])
    train.main()
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r["steps"] for r in rows]
    assert steps == sorted(steps) and steps[-1] >= 2048 and len(rows) > len(first)
    assert [r["eval"] for r in rows] == list(range(len(rows)))
    tags = [json.loads(x)["tags"] for x in (run / "videos" / "index.jsonl").read_text().splitlines()]
    assert sum("untrained" in t for t in tags) == 1

    with pytest.raises(SystemExit):  # starting fresh into an existing run dir is refused
        monkeypatch.setattr(sys, "argv", ["train", str(cfg_path)])
        train.main()


def test_resume_fine_tune_counts_budget_from_init_model(cfg, tmp_path, monkeypatch):
    import sys
    import yaml

    venv = envs.make_vec_env(cfg)
    seed_model = train.build_model(cfg, venv)
    seed_model.learn(2048)
    seed_model.save(tmp_path / "seed.zip")
    venv.close()
    base = seed_model.num_timesteps
    cfg_path = tmp_path / "ft.yaml"
    cfg_path.write_text(yaml.safe_dump({**cfg.to_dict(), "name": "ft", "total_timesteps": 1024,
                                        "init_from": str(tmp_path / "seed.zip")}))
    monkeypatch.setattr(sys, "argv", ["train", str(cfg_path)])
    train.main()
    run = cfg.run_dir.parent / "ft"
    monkeypatch.setattr(sys, "argv", ["train", "--resume", str(run), "-o", "total_timesteps=3072"])
    train.main()
    # 3072 steps on top of the init model, not 3072 in total (which it had already passed).
    assert load_model(run / "final.zip").num_timesteps >= base + 3072


def test_game_backend_uses_training_episodes(cfg, monkeypatch):
    called = []
    monkeypatch.setattr(callbacks, "evaluate", lambda *a, **k: called.append(1))
    cfg.backend = "game"
    venv = envs.make_vec_env(cfg)
    model = train.build_model(cfg, venv)
    model.learn(cfg.total_timesteps, callback=callbacks.EvalVideoCallback(cfg))
    venv.close()
    assert not called  # never opens a second game connection
    rows = [json.loads(x) for x in (cfg.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows and all(r["source"] == "training_episodes" for r in rows)
    assert all(r["reward"] <= 30 for r in rows)  # unscaled return: at most 30 rewarded steps
    assert (cfg.run_dir / "best.zip").exists()
