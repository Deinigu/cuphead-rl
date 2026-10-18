"""Cuphead: movement, jump, dash, parry, peashooter and EX.

The player owns its own physics; projectiles it fires are appended to the world's
bullet list. ``y`` is the feet position (0 = standing on the ground). Numbers and update
order follow ``LevelPlayerMotor`` / ``AbstractLevelWeapon`` (see docs/GAME_CONSTANTS.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cuphead_rl import spec
from cuphead_rl.sim.config import SimConfig

_D = math.sqrt(0.5)
# Bullet spawn offsets (x forward, y up from the feet) and unit directions per pose, from
# LevelPlayerWeaponManager.ProjectilePosition / GetCurrentPose / UpdateAim (Cuphead, not
# Chalice). Keys: (grounded, ducking, moving, aim).
_SHOTS: dict[tuple[bool, bool, bool, int], tuple[float, float, float, float]] = {
    # grounded: Forward / Up / Forward_R (running) / Up_D_R (running, aiming up) / Duck
    (True, False, False, 0): (78.0, 64.0, 1.0, 0.0),
    (True, False, False, 1): (27.0, 158.0, 0.0, 1.0),
    (True, False, True, 0): (70.0, 46.0, 1.0, 0.0),
    (True, False, True, 1): (73.0, 107.0, _D, _D),
    (True, True, False, -1): (102.0, 24.0, 1.0, 0.0),
    (True, True, True, -1): (102.0, 24.0, 1.0, 0.0),
    # airborne (Jump pose): 8-way
    (False, False, False, 0): (78.0, 64.0, 1.0, 0.0),
    (False, False, True, 0): (78.0, 64.0, 1.0, 0.0),
    (False, False, False, 1): (0.0, 158.0, 0.0, 1.0),
    (False, False, True, 1): (71.0, 107.0, _D, _D),
    (False, False, False, -1): (0.0, -11.0, 0.0, -1.0),
    (False, False, True, -1): (71.0, 20.0, _D, -_D),
}


@dataclass(slots=True)
class Bullet:
    x: float
    y: float
    vx: float
    half_w: float
    half_h: float
    damage: float
    ex: bool = False
    vy: float = 0.0
    damage_left: float = 0.0  # EX: total damage it can still deal
    next_hit_dist: float = 0.0  # EX: distance travelled before it may hit again
    travelled: float = 0.0
    freeze: float = 0.0  # EX: pauses briefly on every hit


@dataclass(slots=True)
class FrameInput:
    """Inputs for one simulation frame (edges already resolved by the env)."""

    move: int = 0
    jump_held: bool = False
    jump_pressed: bool = False
    dash_pressed: bool = False
    shoot: bool = False
    ex_pressed: bool = False
    aim: int = 0  # -1 down (duck on the ground), 0, +1 up


class Player:
    def __init__(self, cfg: SimConfig, x: float) -> None:
        self.cfg = cfg
        self.pc = cfg.player
        self.wc = cfg.weapon
        self.half_w = self.pc.hurt_w * 0.5
        self.x_limit = cfg.arena.half_width - self.half_w
        self.x = x
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.grounded = True
        self.facing = 1
        self.hp = spec.PLAYER_MAX_HP
        self.invuln = 0.0
        self.meter = 0.0
        self.can_air_dash = True
        self.dash_timer = 0.0
        self.dash_dir = 1
        self.dash_cooldown = 0.0
        self.jumping = False  # in the held, constant-vy part of a jump
        self.jump_timer = 0.0
        self.parry_timer = 0.0
        self.parry_used = False  # one parry attempt per airtime (re-armed on landing/success)
        # Fixed-length effects count whole frames (float timers drift a frame long).
        dt = cfg.dt
        self.ex_frames = 0
        self.stun_frames = 0
        self._ex_freeze_frames = round(self.pc.ex_freeze / dt)
        self._ex_knock_frames = round(self.pc.ex_knockback_time / dt)
        self._ex_shot: tuple[float, float, float, float] | None = None  # pose of the EX in progress
        self._ex_vx = 0.0  # knockback once it is fired
        self._stun_frames = round(self.pc.hit_stun / dt)
        self._buffer_frames = round(self.pc.input_buffer_time / dt)
        # Input buffer (frames): a press that can't act yet (dashing, stunned, EX...) stays live briefly.
        self.jump_buf = 0
        self.dash_buf = 0
        self.ex_buf = 0
        self.shot_ticks = self.wc.pea_interval_frames  # ticks since the last shot; the first press fires at once
        self.shot_count = 0
        self.shooting = False  # for the renderer
        self.ducking = False
        self.aim = 0
        self.moving = False

    # ------------------------------------------------------------------ helpers
    @property
    def dashing(self) -> bool:
        return self.dash_timer > 0.0

    @property
    def parrying(self) -> bool:
        return self.parry_timer > 0.0

    def box(self) -> tuple[float, float, float, float]:
        """Hurtbox (cx, cy, half_w, half_h): tall standing, short ducking, small in the air."""
        pc = self.pc
        if self.grounded:
            if self.ducking:
                return self.x, self.y + pc.duck_hurt_cy, self.half_w, pc.duck_hurt_h * 0.5
            return self.x, self.y + pc.hurt_cy, self.half_w, pc.hurt_h * 0.5
        return self.x, self.y + pc.air_hurt_cy, self.half_w, pc.air_hurt_h * 0.5

    def add_meter(self, cards: float) -> None:
        self.meter = min(spec.SUPER_METER_MAX, self.meter + cards)

    # ------------------------------------------------------------------ update
    def step(self, inp: FrameInput, dt: float, bullets: list[Bullet]) -> None:
        # Hot path (runs every frame): plain comparisons instead of max() / properties.
        pc, wc = self.pc, self.wc
        if self.invuln > 0.0:
            self.invuln = self.invuln - dt if self.invuln > dt else 0.0
        if self.parry_timer > 0.0:
            self.parry_timer = self.parry_timer - dt if self.parry_timer > dt else 0.0
        if self.dash_cooldown > 0.0:
            self.dash_cooldown = self.dash_cooldown - dt if self.dash_cooldown > dt else 0.0
        buf = self._buffer_frames
        self.jump_buf = buf if inp.jump_pressed else (self.jump_buf - 1 if self.jump_buf else 0)
        self.dash_buf = buf if inp.dash_pressed else (self.dash_buf - 1 if self.dash_buf else 0)
        self.ex_buf = buf if inp.ex_pressed else (self.ex_buf - 1 if self.ex_buf else 0)

        if self.stun_frames:  # knocked back: no control
            self.stun_frames -= 1
            self.shooting = False
            self._fall(dt)
            self._move(dt)
            return

        dashing = self.dash_timer > 0.0
        if inp.move != 0 and not dashing and not self.ex_frames:
            self.facing = inp.move
        self.aim = inp.aim
        self.moving = inp.move != 0
        # Down on the ground = duck (LevelPlayerMotor.Ducking): no walking, low hurtbox.
        self.ducking = inp.aim < 0 and self.grounded and not dashing

        # EX: costs one card; no control and no gravity for the whole anim (ex_frames counts
        # the ticks left after this one). Frozen for ex_freeze, then the shot leaves and the
        # player slides back for ex_knockback_time (LevelPlayerMotor.exMove_cr).
        if self.ex_frames:
            self.ex_frames -= 1
            self.shooting = False
            self.vy = 0.0
            if self.ex_frames < self._ex_knock_frames:
                if self._ex_shot is not None:  # OnExFired
                    ox, oy, dx, dy = self._ex_shot
                    self._ex_shot = None
                    bullets.append(Bullet(
                        x=self.x + ox, y=self.y + oy, vx=dx * wc.ex_speed, vy=dy * wc.ex_speed,
                        half_w=wc.ex_half, half_h=wc.ex_half,
                        damage=wc.ex_damage, ex=True, damage_left=wc.ex_max_damage,
                    ))
                self.vx = self._ex_vx
                self._move(dt)
            else:
                self.vx = 0.0
            return
        if self.ex_buf and self.meter >= 1.0 and not dashing:
            self.ex_buf = 0
            self.meter -= 1.0
            self.ex_frames = self._ex_freeze_frames + self._ex_knock_frames - 1
            self.dash_timer = 0.0
            self.jumping = False
            self._ex_shot = self._shot_pose()
            # Straight up/down poses (exDelay_cr) don't slide; forward and diagonals do.
            vertical = self.aim != 0 and not self.moving
            self._ex_vx = 0.0 if vertical else -self.facing * pc.ex_knockback
            self.vx = self.vy = 0.0
            return

        # Dash: fixed speed, no gravity, one in the air per airtime. No i-frames.
        if self.dash_buf and not dashing:
            if self.grounded and self.dash_cooldown <= 0.0:
                self.dash_buf = 0
                self._start_dash(inp.move)
                dashing = True
            elif not self.grounded and self.can_air_dash:
                self.dash_buf = 0
                self.can_air_dash = False
                self._start_dash(inp.move)
                dashing = True

        if dashing:
            self.dash_timer -= dt
            self.vx = self.dash_dir * pc.dash_speed
            self.vy = 0.0
            self.jumping = False
            if self.dash_timer <= 1e-9:
                self.dash_timer = 0.0
                if self.grounded:
                    self.dash_cooldown = pc.ground_dash_cooldown
        else:
            self.vx = 0.0 if self.ducking else inp.move * pc.run_speed
            if self.jump_buf:
                if self.grounded:
                    self.jump_buf = 0
                    self.vy = pc.jump_speed
                    self.grounded = False
                    self.jumping = True
                    self.jump_timer = 0.0
                elif inp.jump_pressed and not self.parry_used:
                    self.jump_buf = 0
                    self.parry_used = True
                    self.parry_timer = pc.parry_window
            # Variable height: HandleJumping resets vy to jump_speed on every jumping tick,
            # then decides whether the next tick still jumps (held, at least hold_min, at most
            # hold_max), and HandleFalling applies a tick of gravity after it: the player rises
            # at jump_speed - gravity * dt (1123.2 u/s, measured through the bridge) for 2 ticks on a one-tick
            # tap and 10 on a full hold (traces: tap apex 133.5, full hold 283.3).
            if self.jumping:
                self.vy = pc.jump_speed
                self.jump_timer += dt
                held = inp.jump_held or self.jump_timer <= pc.jump_hold_min
                if not (held and self.jump_timer < pc.jump_hold_max + 1e-9):
                    self.jumping = False
            if not self.grounded:  # gravity (inlined _fall)
                vy = self.vy - pc.gravity * dt
                self.vy = vy if vy > -pc.max_fall_speed else -pc.max_fall_speed
        self._move(dt)

        # Peashooter: 8-tick cadence, real projectiles, paused while dashing.
        dashing = self.dash_timer > 0.0
        self.shooting = inp.shoot and not dashing
        if not dashing:
            self.shot_ticks += 1
        if self.shooting and self.shot_ticks >= wc.pea_interval_frames:
            self.shot_ticks = 0
            lane = wc.pea_y_offsets[self.shot_count % len(wc.pea_y_offsets)]
            self.shot_count += 1
            ox, oy, dx, dy = self._shot_pose()
            # Lanes are spread perpendicular to the shot (upward for horizontal shots).
            px, py = (-dy, dx) if dx >= 0.0 else (dy, -dx)
            bullets.append(Bullet(
                x=self.x + ox + px * lane, y=self.y + oy + py * lane,
                vx=dx * wc.pea_speed, vy=dy * wc.pea_speed, half_w=wc.pea_half_w, half_h=wc.pea_half_h,
                damage=wc.pea_damage,
            ))

    def _shot_pose(self) -> tuple[float, float, float, float]:
        """(spawn dx, spawn dy, dir x, dir y) for the current pose, mirrored by facing."""
        aim = self.aim
        duck = self.ducking and self.grounded
        if self.grounded and aim < 0 and not duck:
            aim = 0
        ox, oy, dx, dy = _SHOTS[(self.grounded, duck, self.moving, aim)]
        f = self.facing
        return ox * f, oy, dx * f, dy

    def _fall(self, dt: float) -> None:
        if not self.grounded:
            self.vy = max(-self.pc.max_fall_speed, self.vy - self.pc.gravity * dt)

    def _move(self, dt: float) -> None:
        self.x += self.vx * dt
        if self.x > self.x_limit:
            self.x = self.x_limit
        elif self.x < -self.x_limit:
            self.x = -self.x_limit
        if not self.grounded:
            self.y += self.vy * dt
            if self.y <= 0.0:
                self.y = 0.0
                self.vy = 0.0
                self.grounded = True
                self.jumping = False
                self.can_air_dash = True
                self.parry_timer = 0.0
                self.parry_used = False

    def _start_dash(self, move: int) -> None:
        self.dash_dir = move if move != 0 else self.facing
        self.facing = self.dash_dir
        self.dash_timer = self.pc.dash_time
        self.parry_timer = 0.0

    def parry_success(self) -> None:
        self.parry_timer = 0.0
        self.parry_used = False
        self.vy = self.pc.parry_bounce_speed
        self.jumping = False
        self.can_air_dash = True
        self.invuln = max(self.invuln, self.pc.parry_invuln)
        self.add_meter(self.wc.meter_per_parry)

    def take_hit(self, source_x: float) -> None:
        pc = self.pc
        self.hp -= 1
        self.invuln = pc.hit_invuln
        self.stun_frames = self._stun_frames
        self.ducking = False
        self.dash_timer = 0.0
        self.ex_frames = 0
        self._ex_shot = None  # a hit aborts the EX before it fires (ResetSuperAndEx)
        self.jumping = False
        self.parry_timer = 0.0
        self.vx = pc.hit_knockback * (1.0 if self.x >= source_x else -1.0)
        self.vy = pc.hit_jump_speed
        self.grounded = False

    def to_spec(self) -> spec.PlayerState:
        return spec.PlayerState(
            x=self.x, y=self.y, vx=self.vx, vy=self.vy, grounded=self.grounded,
            dashing=self.dashing, can_dash=self.can_air_dash, facing=self.facing,
            hp=self.hp, invuln=self.invuln, super_meter=self.meter,
        )
