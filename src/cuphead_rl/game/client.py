"""Socket client for the CupheadRLBridge BepInEx plugin (see docs/GAME_BRIDGE.md).

Protocol: newline-delimited JSON over 127.0.0.1:47800. The client sends commands
(``{"cmd": ...}``); the plugin sends ``state`` messages (streamed every frame in free-run mode,
or once per action in lockstep mode) plus replies (``hello``, ``config``, ``reset_done``, ``error``).
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

from cuphead_rl import spec

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47800


class BridgeError(RuntimeError):
    pass


def to_game_state(msg: dict[str, Any]) -> spec.GameState:
    """Convert a bridge ``state`` message (already in spec coordinates) into a spec.GameState."""
    p, b = msg["player"], msg["boss"]
    player = spec.PlayerState(
        x=p["x"],
        y=p["y"],
        vx=p["vx"],
        vy=p["vy"],
        grounded=p["grounded"],
        dashing=p["dashing"],
        can_dash=p["can_dash"],
        facing=p["facing"],
        hp=p["hp"],
        invuln=p["invuln"],
        super_meter=p["super_meter"],
    )
    boss = spec.BossInfo(
        phase=spec.BossPhase(b["phase"]),
        state=spec.BossMove(b["state"]),
        x=b["x"],
        y=b["y"],
        half_w=b["half_w"],
        half_h=b["half_h"],
        vx=b["vx"],
        vy=b["vy"],
        state_time=b["state_time"],
        facing=b["facing"],
        hp=b["hp"],
        max_hp=b["max_hp"],
        phase_hp_frac=b["phase_hp_frac"],
        vulnerable=b["vulnerable"],
    )
    hazards = [
        spec.Hazard(
            kind=spec.HazardKind(h["kind"]),
            x=h["x"],
            y=h["y"],
            half_w=h["half_w"],
            half_h=h["half_h"],
            vx=h["vx"],
            vy=h["vy"],
            parryable=h["parryable"],
        )
        for h in msg["hazards"]
    ]
    return spec.GameState(
        player=player,
        boss=boss,
        hazards=hazards,
        time=msg["time"],
        damage_dealt=msg["damage_dealt"],
        won=msg["won"],
        lost=msg["lost"],
        parries=msg["parries"],
    )


# mod/CupheadRLBridge/SaveGuard.cs: the user's save slots 0/1 may only be touched by the game
# while these four patches are live (hello.save_guard is read from Harmony at hello time).
SAVE_GUARDS = frozenset({
    "OnlineInterfaceSteam.SaveCloudData", "PlayerData.ClearSlot", "PlayerData.Save", "PlayerData.set_CurrentSaveFileIndex",
})
BRIDGE_SAVE_SLOT = 2


def check_save_guard(hello: dict[str, Any]) -> None:
    """Refuse to drive the game unless the plugin's save guard is complete."""
    patched = {g for g in (hello.get("save_guard") or "").split(",") if g}
    missing = SAVE_GUARDS - patched
    if missing or hello.get("save_slot") != BRIDGE_SAVE_SLOT:
        raise BridgeError(
            f"plugin save guard incomplete (save_slot={hello.get('save_slot')}, missing={sorted(missing)}): refusing to "
            "run so the user's save slots 0/1 cannot be written; rebuild/reinstall mod/CupheadRLBridge"
        )


class BridgeClient:
    """Thin blocking client. Not thread-safe."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 30.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock: socket.socket | None = None
        self._buf = b""
        self.last_state: dict[str, Any] | None = None
        self.last_reset: dict[str, Any] | None = None
        self.hello: dict[str, Any] | None = None
        self._seq = 0

    # ------------------------------------------------------------------ connection
    def connect(self, wait: float = 0.0, require_save_guard: bool = True) -> dict[str, Any]:
        """Connect (retrying for up to ``wait`` seconds while the game boots) and say hello.

        Raises unless the plugin reports every save guard installed and slot 2 in use
        (``require_save_guard=False`` only for tests of the guard itself). An "already connected"
        rejection is retried once: the previous client may still be being dropped."""
        try:
            try:
                hello = self._connect(wait)
            except BridgeError as e:
                if "already connected" not in str(e):
                    raise
                self.close()
                time.sleep(1.0)
                hello = self._connect(0.0)
            if require_save_guard:
                check_save_guard(hello)
        except BaseException:
            self.close()
            raise
        return hello

    def _connect(self, wait: float) -> dict[str, Any]:
        self._buf = b""
        deadline = time.monotonic() + wait
        while True:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.send({"cmd": "hello"})
            hello = self.wait_for("hello")  # a second client gets an "error" line instead (raises)
        except ConnectionError as e:
            self.close()
            raise BridgeError(f"bridge dropped the connection during hello (another client already connected?): {e}") from e
        if hello.get("spec_version") != spec.SPEC_VERSION:
            self.close()
            raise BridgeError(
                f"bridge implements spec v{hello.get('spec_version')}, this code expects v{spec.SPEC_VERSION}: "
                "rebuild/reinstall mod/CupheadRLBridge"
            )
        self.hello = hello
        return hello

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ framing
    def send(self, msg: dict[str, Any]) -> None:
        if self.sock is None:
            raise BridgeError("not connected")
        self.sock.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode())

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        """Return the next message, blocking up to ``timeout`` seconds."""
        assert self.sock is not None
        self.sock.settimeout(self.timeout if timeout is None else timeout)
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise BridgeError("bridge closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        msg = json.loads(line)
        if msg.get("type") == "state":
            self.last_state = msg
        elif msg.get("type") == "error":
            raise BridgeError(msg.get("message"))
        return msg

    def drain(self) -> dict[str, Any] | None:
        """Consume everything already received; return the newest state (or None)."""
        assert self.sock is not None
        newest = None
        self.sock.setblocking(False)
        try:
            while True:
                try:
                    chunk = self.sock.recv(1 << 16)
                except (BlockingIOError, socket.timeout):
                    break
                if not chunk:
                    raise BridgeError("bridge closed the connection")
                self._buf += chunk
        finally:
            self.sock.setblocking(True)
        while b"\n" in self._buf:
            msg = self.recv()
            if msg.get("type") == "state":
                newest = msg
        return newest

    def wait_for(self, type_: str, timeout: float | None = None, pred=None) -> dict[str, Any]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no '{type_}' message from the bridge")
            msg = self.recv(timeout=left)
            if msg.get("type") == type_ and (pred is None or pred(msg)):
                return msg

    # ------------------------------------------------------------------ commands
    def configure(self, **cfg: Any) -> dict[str, Any]:
        """Keys: stream_every (frames), lockstep (bool), frames (ticks/action), time_scale, control."""
        self.send({"cmd": "config", **cfg})
        return self.wait_for("config")

    def reset(self, timeout: float = 120.0, seed: int | None = None, character: str | None = None) -> dict[str, Any]:
        """(Re)load the Goopy fight; returns the first state once the fight has started.

        ``seed`` seeds UnityEngine.Random when the level starts (reproducible boss patterns in
        lockstep mode). ``character`` ("mugman" | "cuphead") is player one for this and later
        loads; None keeps the current one (the plugin starts with Mugman)."""
        msg: dict[str, Any] = {"cmd": "reset"}
        if seed is not None:
            msg["seed"] = int(seed)
        if character is not None:
            msg["character"] = character
        self.send(msg)
        self.last_reset = self.wait_for("reset_done", timeout=timeout)
        return self.wait_for("state", timeout=timeout, pred=lambda m: m.get("in_fight"))

    def damage(self, amount: float) -> dict[str, Any]:
        """Debug/curriculum: damage the boss as if shot; returns the state right after."""
        self.send({"cmd": "damage", "amount": float(amount)})
        return self.wait_for("state")

    def invincible(self, on: bool = True) -> dict[str, Any]:
        """Debug: make the player immune to damage (for hitbox sampling, not for training)."""
        self.send({"cmd": "invincible", "on": bool(on)})
        return self.wait_for("state")

    def set_meter(self, cards: float) -> dict[str, Any]:
        """Debug: set the player's super meter in cards (0..5)."""
        self.send({"cmd": "set_meter", "cards": float(cards)})
        return self.wait_for("state")

    def request_state(self) -> dict[str, Any]:
        self.send({"cmd": "state"})
        return self.wait_for("state")

    def act(self, action: spec.Action, frames: int | None = None, move_y: int | None = None) -> int:
        """Send an action; returns its sequence id (echoed back in every later state as ``seq``)."""
        self._seq += 1
        msg = {
            "cmd": "act",
            "seq": self._seq,
            "move": action.move,
            "jump": action.jump,
            "dash": action.dash,
            "shoot": action.shoot,
            "ex": action.ex,
        }
        if frames is not None:
            msg["frames"] = int(frames)
        # vertical stick: spec v2 Action.aim (-1 duck, +1 aim up / diagonal while moving)
        msg["move_y"] = int(action.aim if move_y is None else move_y)
        self.send(msg)
        return self._seq

    def step_lockstep(self, action: spec.Action, frames: int, move_y: int | None = None) -> dict[str, Any]:
        """Lockstep mode: apply ``action`` for ``frames`` ticks and return the resulting state."""
        self.act(action, frames, move_y)
        return self.wait_for("state")

    def step_free(self, action: spec.Action, frames: int, timeout: float = 5.0) -> dict[str, Any]:
        """Free-run mode: apply ``action`` and return the first state at least ``frames`` ticks
        after the tick the game applied it.

        Streamed states that piled up while the agent was thinking are discarded first, and the
        plugin echoes the action's sequence id, so the returned state always reflects this action
        (no growing backlog)."""
        self.drain()
        seq = self.act(action)
        return self.wait_for(
            "state",
            timeout=timeout,
            pred=lambda m: not m.get("in_fight") or (m.get("seq") == seq and m["tick"] >= m["act_tick"] + frames),
        )
