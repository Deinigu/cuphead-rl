"""The fight: player + Goopy + projectiles + parryables, advanced one 60 Hz frame at a time."""

from __future__ import annotations

import random
from dataclasses import dataclass

from cuphead_rl import spec
from cuphead_rl.spec import BossPhase, GameState, Hazard, HazardKind
from cuphead_rl.sim.config import SimConfig
from cuphead_rl.sim.goopy import Goopy
from cuphead_rl.sim.player import Bullet, FrameInput, Player


@dataclass(slots=True)
class QuestionMark:
    """Pink '?' spawned during the phase 1 -> 2 transformation. Parry it for a card."""

    x: float
    y: float
    idx: int = -1  # index in Goopy's transformation timeline (-1: placed by hand)


def overlap(ax: float, ay: float, ahw: float, ahh: float, bx: float, by: float, bhw: float, bhh: float) -> bool:
    """Axis-aligned boxes given as centre + half extents."""
    return abs(ax - bx) < ahw + bhw and abs(ay - by) < ahh + bhh


def circle_box(cx: float, cy: float, r: float, bx: float, by: float, bhw: float, bhh: float) -> bool:
    """Circle (centre, radius) vs axis-aligned box (centre, half extents)."""
    dx = abs(cx - bx) - bhw
    dy = abs(cy - by) - bhh
    dx = dx if dx > 0.0 else 0.0
    dy = dy if dy > 0.0 else 0.0
    return dx * dx + dy * dy < r * r


class World:
    def __init__(self, cfg: SimConfig, seed: int = 0, start_phase: BossPhase | int | None = None) -> None:
        self.cfg = cfg
        self.dt = cfg.dt
        self.rng = random.Random(seed)
        phase = BossPhase(start_phase) if start_phase is not None else BossPhase.SMALL
        self.player = Player(cfg, cfg.arena.player_start_x)
        self.boss = Goopy(cfg, self.rng, phase)
        self.bullets: list[Bullet] = []
        self.qmarks: list[QuestionMark] = []
        self.time = 0.0
        self.frame = 0
        self.damage_dealt = 0.0
        self.parries = 0
        self.hits_taken = 0
        self.won = False
        self.lost = False
        self._parried_qmarks: set[int] = set()
        self._body = None  # boss body geometry of the last frame (reused by state())

    @property
    def done(self) -> bool:
        return self.won or self.lost

    def step(self, inp: FrameInput) -> None:
        if self.won or self.lost:
            return
        dt = self.dt
        cfg = self.cfg
        wc = cfg.weapon
        p, b = self.player, self.boss
        self.frame += 1
        self.time = self.frame * dt  # no float drift over long episodes

        p.step(inp, dt, self.bullets)
        b.step(dt, p.x)

        # Boss geometry, read once per frame (the boss doesn't move while bullets resolve).
        body = self._body = b.body()
        bx, by, bhw, bhh, round_body = b.hit_body(body)
        track0, state0 = b.track, b.state
        r2 = bhw * bhw
        dead = b.dead
        vulnerable = b.vulnerable

        # Projectiles vs boss (the hottest loop when shooting: keep it flat).
        lim = cfg.arena.half_width + 120.0
        top = spec.ARENA_HEIGHT + 120.0
        alive: list[Bullet] = []
        append = alive.append
        for bl in self.bullets:
            if bl.ex:
                if bl.freeze > 0.0:
                    bl.freeze -= dt
                else:
                    bl.x += bl.vx * dt
                    bl.y += bl.vy * dt
                    bl.travelled += wc.ex_speed * dt
                if bl.travelled < bl.next_hit_dist:
                    if -lim < bl.x < lim and -100.0 < bl.y < top:
                        append(bl)
                    continue
            else:
                bl.x += bl.vx * dt
                bl.y += bl.vy * dt
            x = bl.x
            if dead:
                hit = False
            elif round_body:  # circle vs bullet box
                ddx = (bx - x if bx > x else x - bx) - bl.half_w
                ddy = (by - bl.y if by > bl.y else bl.y - by) - bl.half_h
                if ddx < 0.0:
                    ddx = 0.0
                if ddy < 0.0:
                    ddy = 0.0
                hit = ddx * ddx + ddy * ddy < r2
            else:
                hit = abs(x - bx) < bl.half_w + bhw and abs(bl.y - by) < bl.half_h + bhh
            if hit:
                if not vulnerable:
                    continue  # absorbed by an invulnerable boss
                if bl.ex:
                    dealt = b.damage(min(bl.damage, bl.damage_left))
                    bl.damage_left -= bl.damage
                    bl.next_hit_dist = bl.travelled + wc.ex_hit_distance
                    bl.freeze = wc.ex_hit_freeze
                    if wc.meter_from_ex:
                        p.add_meter(dealt * wc.meter_per_damage)
                else:
                    dealt = b.damage(bl.damage)
                    p.add_meter(dealt * wc.meter_per_damage)
                self.damage_dealt += dealt
                dead, vulnerable = b.dead, b.vulnerable  # a hit can kill or start a transition
                if not bl.ex or bl.damage_left <= 1e-6:
                    continue
            if -lim < x < lim and -100.0 < bl.y < top:
                append(bl)
        self.bullets = alive
        if b.track is not track0 or b.state != state0:
            self._body = b.body()  # a hit started a death/transform (e.g. cut a punch short)

        # Question marks follow the transformation timeline (each can be parried once).
        if b.transform:
            self.qmarks = [QuestionMark(qx, qy, i) for i, qx, qy in b.qmark_positions()
                           if i not in self._parried_qmarks]

        # Parryables.
        tc = cfg.transition
        pcx, pcy, phw, phh = p.box()
        keep: list[QuestionMark] = []
        for q in self.qmarks:
            if circle_box(q.x, q.y, tc.qmark_radius, pcx, pcy, phw, phh):
                if p.parrying and not p.grounded:
                    p.parry_success()
                    self.parries += 1
                    self._parried_qmarks.add(q.idx)
                    continue
                if tc.qmark_harmful and p.invuln <= 0.0:
                    self._hit(q.x)
            keep.append(q)
        self.qmarks = keep

        # Boss hazards vs player.
        if p.invuln <= 0.0 and not self.lost:
            # Fast path: usually the only hazard is the slime body circle.
            if round_body and b.track is None and b.phase != BossPhase.TOMB and not dead:
                ddx = abs(bx - pcx) - phw
                ddy = abs(by - pcy) - phh
                if ddx < 0.0:
                    ddx = 0.0
                if ddy < 0.0:
                    ddy = 0.0
                if ddx * ddx + ddy * ddy < r2:
                    self._hit(bx)
                hazards = ()
            else:
                hazards = b.hazard_tuples(body)
            for _, hx, hy, hw, hh, _, _, rnd in hazards:
                hit = circle_box(hx, hy, hw, pcx, pcy, phw, phh) if rnd else overlap(pcx, pcy, phw, phh, hx, hy, hw, hh)
                if hit:
                    self._hit(hx)
                    break

        if b.dead:
            self.won = True

    def _hit(self, source_x: float) -> None:
        self.player.take_hit(source_x)
        self.hits_taken += 1
        if self.player.hp <= 0:
            self.player.hp = 0
            self.lost = True

    # ------------------------------------------------------------------ export
    def hazards(self, body=None) -> list[Hazard]:
        out = self.boss.hazards(body)
        tc = self.cfg.transition
        for q in self.qmarks:
            out.append(Hazard(HazardKind.OTHER, q.x, q.y, tc.qmark_radius, tc.qmark_radius,
                              0.0, 0.0, parryable=True))
        return out

    def state(self) -> GameState:
        b = self.boss
        # Geometry of the last frame, recomputed in step() if a hit changed the boss's move.
        x, y, hw, hh, rnd = self._body if self._body is not None else b.body()
        vx, vy = b.velocity()
        return GameState(
            player=self.player.to_spec(),
            boss=spec.BossInfo(
                phase=b.phase, state=b.state, x=x, y=y, half_w=hw, half_h=hh,
                vx=vx, vy=vy, state_time=b.state_time, facing=b.facing, hp=b.hp,
                max_hp=b.max_hp, phase_hp_frac=b.phase_hp_frac, vulnerable=b.vulnerable,
            ),
            hazards=self.hazards((x, y, hw, hh, rnd)),
            time=self.time,
            damage_dealt=self.damage_dealt,
            won=self.won,
            lost=self.lost,
            parries=self.parries,
        )
