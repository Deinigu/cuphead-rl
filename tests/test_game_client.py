"""Client/env logic of the real-game backend, tested against a fake bridge (no game needed)."""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

from cuphead_rl import spec
from cuphead_rl.game.client import BridgeClient, BridgeError, to_game_state
from cuphead_rl.game.env import EXPECTED_LOADOUT, CupheadGameEnv

FPS = spec.GAME_FPS


def state_msg(tick: int = 0, t: float = 0.0, seq: int = -1, act_tick: int = 0, **over) -> dict:
    m = {
        "type": "state", "tick": tick, "frame": tick, "mode": "lockstep", "in_fight": True, "started": True,
        "no_hit": False, "character": "mugman", "seq": seq, "act_tick": act_tick, "time": t,
        "player": {"x": -410.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "grounded": True, "dashing": False,
                   "can_dash": True, "facing": 1, "hp": 3, "invuln": 0.0, "super_meter": 0.0},
        "boss": {"phase": 0, "state": 1, "x": 369.0, "y": 78.0, "half_w": 80.0, "half_h": 80.0, "vx": 0.0,
                 "vy": 0.0, "state_time": 0.0, "facing": -1, "hp": 1200.0, "max_hp": 1200.0,
                 "phase_hp_frac": 1.0, "vulnerable": True, "entity": "small", "anim": ""},
        "hazards": [{"kind": 0, "x": 369.0, "y": 78.0, "half_w": 80.0, "half_h": 80.0, "vx": 0.0, "vy": 0.0,
                     "parryable": False}],
        "damage_dealt": 0.0, "won": False, "lost": False, "parries": 0,
    }
    m.update(over)
    return m


class FakeBridge:
    """Speaks the plugin's protocol on a random port. Lockstep semantics: every act advances
    ``frames`` ticks; ``paused`` freezes in-game time; ``reset_time`` is the fight time at reset."""

    def __init__(self, spec_version: int = spec.SPEC_VERSION, loadout: dict | None = None,
                 reject: bool = False, backlog: int = 0, ignore_character: bool = False):
        self.spec_version = spec_version
        self.loadout = EXPECTED_LOADOUT if loadout is None else loadout
        self.reject = reject
        self.reject_first = 0  # reject this many connections, then accept (reconnect race)
        self.backlog = backlog
        self.paused = False
        self.reset_time = 0.0
        self.tick, self.t, self.frames, self.seq = 0, 0.0, 4, -1
        self.live_ticks = 0  # ticks that advanced the fight clock since reset
        self.lockstep = True
        self.no_hit = False
        self.character = "mugman"
        self.save_guard = ("OnlineInterfaceSteam.SaveCloudData,PlayerData.ClearSlot,PlayerData.Save,"
                           "PlayerData.set_CurrentSaveFileIndex")
        self.ignore_character = ignore_character  # an outdated plugin
        self.received: list[dict] = []
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _state(self, **over) -> dict:
        return state_msg(self.tick, self.t, self.seq, self.act_tick, no_hit=self.no_hit, character=self.character, **over)

    act_tick = 0

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        f = conn.makefile("rwb")

        def send(m: dict) -> None:
            f.write((json.dumps(m) + "\n").encode())
            f.flush()

        if self.reject_first > 0:
            self.reject_first -= 1
            self.reject_now = True
        else:
            self.reject_now = False
        if self.reject or self.reject_now:
            # like the plugin: say why, half-close, let the client's hello arrive, then close (no RST)
            send({"type": "error", "message": "another client is already connected"})
            conn.shutdown(socket.SHUT_WR)
            conn.settimeout(1.0)
            try:
                while conn.recv(4096):
                    pass
            except OSError:
                pass
            conn.close()
            return
        for line in f:
            m = json.loads(line)
            self.received.append(m)
            cmd = m["cmd"]
            if cmd == "hello":
                send({"type": "hello", "spec_version": self.spec_version, "no_hit": self.no_hit, "loadout": self.loadout,
                      "save_slot": 2, "save_guard": self.save_guard})
            elif cmd == "config":
                self.frames = m.get("frames", self.frames)
                self.lockstep = m.get("lockstep", self.lockstep)
                send({"type": "config", **m})
            elif cmd == "invincible":
                self.no_hit = m["on"]
                send(self._state())
            elif cmd == "reset":
                self.no_hit = False
                if not self.ignore_character:
                    self.character = m.get("character", self.character)
                self.tick, self.t, self.live_ticks = 0, self.reset_time, 0
                send({"type": "reset_done", "loadout": self.loadout, "no_hit": False})
                send(self._state())
            elif cmd == "act":
                self.seq, self.act_tick = m["seq"], self.tick
                if self.lockstep:
                    self._advance(m.get("frames", self.frames))
                    send(self._state())
                else:  # free-run: stale backlog, a state from before the action, then progress
                    for i in range(self.backlog):
                        send(state_msg(i, 0.0, -1, 0))
                    send(state_msg(self.tick, self.t, self.seq - 1, 0))
                    for _ in range(self.frames):
                        self._advance(1)
                        send(self._state())

    def _advance(self, n: int) -> None:
        self.tick += n
        if not self.paused:
            self.live_ticks += n
            self.t = self.reset_time + self.live_ticks / FPS

    def close(self) -> None:
        self.srv.close()


@pytest.fixture
def fake():
    fb = FakeBridge()
    yield fb
    fb.close()


def test_to_game_state_roundtrip():
    s = to_game_state(state_msg(t=1.5))
    assert s.time == 1.5 and s.boss.phase == spec.BossPhase.SMALL and s.hazards[0].kind == spec.HazardKind.BODY
    assert spec.encode_observation(s).shape == (spec.OBS_SIZE,)


def test_spec_version_mismatch_raises():
    fb = FakeBridge(spec_version=spec.SPEC_VERSION - 1)
    with pytest.raises(BridgeError, match="spec v"):
        BridgeClient(port=fb.port, timeout=5).connect()
    fb.close()


def test_second_client_rejection_raises():
    fb = FakeBridge(reject=True)
    with pytest.raises(BridgeError, match="already connected"):
        BridgeClient(port=fb.port, timeout=5).connect()
    fb.close()


def test_step_free_skips_backlog_and_waits_for_own_action():
    fb = FakeBridge(backlog=50)
    c = BridgeClient(port=fb.port, timeout=5)
    c.connect()
    c.configure(lockstep=False, frames=4)
    for _ in range(3):
        m = c.step_free(spec.Action(move=1), 4)
        assert m["seq"] == c._seq
        assert m["tick"] >= m["act_tick"] + 4
    c.close()
    fb.close()


def test_act_carries_seq_and_aim(fake):
    c = BridgeClient(port=fake.port, timeout=5)
    c.connect()
    c.step_lockstep(spec.Action(aim=-1), 4)
    c.step_lockstep(spec.Action(aim=1), 4)
    acts = [m for m in fake.received if m["cmd"] == "act"]
    assert [a["seq"] for a in acts] == [1, 2]
    assert [a["move_y"] for a in acts] == [-1, 1]
    c.close()


def make_env(fb: FakeBridge, **kw) -> CupheadGameEnv:
    return CupheadGameEnv(port=fb.port, connect_wait=0, **kw)


def test_env_time_starts_at_zero_and_truncates(fake):
    fake.reset_time = 12.25  # e.g. the fight clock after a start_phase skip
    env = make_env(fake, frame_skip=4, max_seconds=1.0)
    try:
        obs, info = env.reset(seed=0)
        assert info["time"] == pytest.approx(0.0)
        assert obs[-1] == pytest.approx(0.0)  # time feature uses max_seconds
        truncated, steps = False, 0
        while not truncated:
            obs, r, term, truncated, info = env.step(env.action_space.sample())
            steps += 1
        assert steps == 15  # 15 * 4 ticks = 1.0 s
        assert info["time"] == pytest.approx(1.0)
        assert obs[-1] == pytest.approx(1.0)
        assert set(info) >= {"won", "phase", "boss_hp_frac", "player_hp", "damage_dealt", "time", "parries"}
    finally:
        env.close()


def test_env_turns_no_hit_off_by_default(fake):
    fake.no_hit = True  # left on by some earlier tool
    env = make_env(fake)
    try:
        assert {"cmd": "invincible", "on": False} in fake.received
    finally:
        env.close()


def test_env_rejects_unknown_kwargs_and_long_episodes(fake):
    with pytest.raises(TypeError):
        make_env(fake, max_episode_seconds=10)
    with pytest.raises(ValueError):
        make_env(fake, max_seconds=spec.MAX_EPISODE_SECONDS + 1)


def test_env_single_instance_per_process(fake):
    env = make_env(fake)
    try:
        with pytest.raises(RuntimeError, match="one CupheadGameEnv"):
            make_env(fake)
    finally:
        env.close()
    make_env(fake).close()  # allowed again once closed


def test_env_asserts_loadout():
    fb = FakeBridge(loadout={"primary": "level_weapon_spread", "secondary": "None", "charm": "None", "super": "None"})
    env = make_env(fb)
    try:
        with pytest.raises(BridgeError, match="loadout"):
            env.reset(seed=0)
    finally:
        env.close()
        fb.close()


def test_env_detects_paused_game(fake):
    env = make_env(fake)
    try:
        env.reset(seed=0)
        fake.paused = True
        with pytest.raises(RuntimeError, match="paused"):
            for _ in range(100):
                env.step(np.ones(len(spec.ACTION_NVEC), dtype=int))
    finally:
        env.close()


def test_client_reads_have_timeouts():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    threading.Thread(target=lambda: (srv.accept(), time.sleep(5)), daemon=True).start()
    c = BridgeClient(port=srv.getsockname()[1], timeout=0.5)
    t0 = time.monotonic()
    with pytest.raises((TimeoutError, socket.timeout)):
        c.connect()
    assert time.monotonic() - t0 < 3
    srv.close()


def test_env_plays_mugman_by_default(fake):
    fake.character = "cuphead"  # whatever the last client chose
    env = make_env(fake)
    try:
        env.reset(seed=0)
        assert [m.get("character") for m in fake.received if m["cmd"] == "reset"] == ["mugman"]
        assert env.last_msg["character"] == "mugman"
        env.close()
        env = make_env(fake, character="cuphead")
        env.reset(seed=0)
        assert env.last_msg["character"] == "cuphead"
    finally:
        env.close()


def test_env_character_validated(fake):
    with pytest.raises(ValueError, match="character"):
        make_env(fake, character="chalice")
    fb = FakeBridge(ignore_character=True)
    fb.character = "cuphead"
    env = make_env(fb)
    try:
        with pytest.raises(BridgeError, match="mugman"):
            env.reset(seed=0)
    finally:
        env.close()
        fb.close()


def test_env_refuses_without_save_guard():
    fb = FakeBridge()
    fb.save_guard = "PlayerData.Save"  # an old plugin, or a failed Harmony patch
    with pytest.raises(BridgeError, match="save guard"):
        make_env(fb)
    fb.close()


def test_client_refuses_without_save_guard():
    fb = FakeBridge()
    fb.save_guard = "OnlineInterfaceSteam.SaveCloudData,PlayerData.Save,PlayerData.ClearSlot"
    c = BridgeClient(port=fb.port, timeout=5)
    with pytest.raises(BridgeError, match="set_CurrentSaveFileIndex"):
        c.connect()
    assert c.sock is None
    assert BridgeClient(port=fb.port, timeout=5).connect(require_save_guard=False)["type"] == "hello"
    fb.close()


def test_client_retries_once_on_reconnect_race():
    fb = FakeBridge()
    fb.reject_first = 1
    c = BridgeClient(port=fb.port, timeout=5)
    assert c.connect()["type"] == "hello"
    c.close()
    fb.reject_first = 2  # a second client that really is concurrent still fails
    with pytest.raises(BridgeError, match="already connected"):
        BridgeClient(port=fb.port, timeout=5).connect()
    fb.close()
