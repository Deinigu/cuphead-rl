"""Offscreen pygame renderer with a 1930s cartoon look.

World -> screen is 1:1: the 1280-unit arena fills the 1280 px frame, the ground line sits
at ``GROUND_Y``. All randomness here (grain, scratches) is seeded by the frame number, so
renders are reproducible and never touch the simulation RNG.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from cuphead_rl import spec  # noqa: E402
from cuphead_rl.spec import BossMove, BossPhase, HazardKind  # noqa: E402
from cuphead_rl.sim.config import SimConfig  # noqa: E402
from cuphead_rl.sim.goopy import DYING, FALLING, GROW  # noqa: E402
from cuphead_rl.sim.world import World  # noqa: E402

W, H = 1280, 720
GROUND_Y = 640
GRAIN_HOLD = 5  # sim frames per film-grain frame (60 / 5 = 12 fps)

INK = (38, 26, 20)
CREAM = (244, 231, 196)
PAPER = (236, 220, 180)
SEPIA = (196, 160, 112)
RED = (196, 48, 40)
MUG_BLUE = (48, 104, 196)
GOO = (92, 156, 214)
GOO_DARK = (50, 104, 168)
GOO_LIGHT = (196, 226, 246)
STONE = (150, 146, 136)
STONE_DARK = (104, 100, 92)
PINK = (236, 92, 156)
GRASS = (122, 140, 72)
GRASS_DARK = (84, 100, 50)
SOIL = (150, 110, 70)

PHASE_NAMES = {BossPhase.SMALL: "PHASE I - GOOPY", BossPhase.BIG: "PHASE II - GOOPY LE GRANDE",
               BossPhase.TOMB: "PHASE III - THE TOMBSTONE"}


def sx(x: float) -> int:
    return int(round(x + W / 2))


def sy(y: float) -> int:
    return int(round(GROUND_Y - y))


def ink_ellipse(surf: pygame.Surface, color, rect, width: int = 3) -> None:
    pygame.draw.ellipse(surf, color, rect)
    pygame.draw.ellipse(surf, INK, rect, width)


def ink_rect(surf: pygame.Surface, color, rect, width: int = 3, radius: int = 0) -> None:
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    pygame.draw.rect(surf, INK, rect, width, border_radius=radius)


def ink_poly(surf: pygame.Surface, color, pts, width: int = 3) -> None:
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, INK, pts, width)


# Player looks (purely cosmetic: same hurtbox and motor): accent colour (shorts, nose,
# straw stripes), drink, head half width / height and handle size.
CHARACTERS = {
    "cuphead": {"accent": RED, "drink": (210, 120, 80), "head_hw": 21, "head_h": 40, "handle": 18},
    "mugman": {"accent": MUG_BLUE, "drink": (236, 230, 214), "head_hw": 24, "head_h": 36, "handle": 22},
}


class Renderer:
    def __init__(self, config: SimConfig | None = None, show_hitboxes: bool = False,
                 character: str = "mugman") -> None:
        pygame.init()
        self.cfg = config or SimConfig()
        self.show_hitboxes = show_hitboxes
        if character not in CHARACTERS:
            raise ValueError(f"unknown character {character!r}, expected one of {sorted(CHARACTERS)}")
        self.look = CHARACTERS[character]
        self.screen = pygame.Surface((W, H))
        self._fonts: dict[tuple[int, bool], pygame.font.Font] = {}
        self._text_cache: dict[tuple, pygame.Surface] = {}
        self.background = self._make_background()
        self.vignette = self._make_vignette()
        self.grain = self._make_grain(6)

    # ------------------------------------------------------------------ assets
    def font(self, size: int, bold: bool = True) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("georgia,timesnewroman,dejavuserif,serif", size, bold=bold)
        return self._fonts[key]

    def text(self, s: str, size: int, color=INK, bold: bool = True) -> pygame.Surface:
        key = (s, size, color, bold)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 2000:
                self._text_cache.clear()
            surf = self.font(size, bold).render(s, True, color)
            self._text_cache[key] = surf
        return surf

    def outlined_text(self, s: str, size: int, color, pos, center: bool = False, outline: int = 2) -> None:
        base = self.text(s, size, INK)
        top = self.text(s, size, color)
        x, y = pos
        if center:
            x -= top.get_width() // 2
            y -= top.get_height() // 2
        for dx in (-outline, 0, outline):
            for dy in (-outline, 0, outline):
                if dx or dy:
                    self.screen.blit(base, (x + dx, y + dy))
        self.screen.blit(top, (x, y))

    def _make_background(self) -> pygame.Surface:
        bg = pygame.Surface((W, H))
        # Warm paper sky, lighter towards the horizon.
        for y in range(H):
            k = y / H
            c = tuple(int(a + (b - a) * k) for a, b in zip((222, 196, 150), (248, 238, 208)))
            pygame.draw.line(bg, c, (0, y), (W, y))
        rng = random.Random(7)
        # Soft clouds.
        for _ in range(7):
            cx, cy = rng.randint(0, W), rng.randint(60, 260)
            for _ in range(4):
                r = rng.randint(28, 60)
                pygame.draw.circle(bg, (250, 242, 222), (cx + rng.randint(-60, 60), cy + rng.randint(-14, 14)), r)
        # Rolling hills in two layers.
        for layer, (col, base, amp) in enumerate((((176, 168, 120), 470, 60), ((150, 150, 96), 540, 45))):
            pts = [(0, H)]
            for x in range(0, W + 41, 40):
                y = base - amp * (0.6 * math.sin(x / 170.0 + layer * 2) + 0.4 * math.sin(x / 63.0 + layer))
                pts.append((x, int(y)))
            pts.append((W, H))
            pygame.draw.polygon(bg, col, pts)
            pygame.draw.lines(bg, INK, False, pts[1:-1], 2)
        # Forest mushrooms and trees (Goopy's forest).
        for x, h in ((70, 120), (210, 90), (1050, 140), (1190, 100), (960, 70)):
            trunk = pygame.Rect(x - 9, GROUND_Y - h, 18, h)
            ink_rect(bg, (120, 86, 56), trunk, 2)
            ink_ellipse(bg, (120, 136, 82), (x - 46, GROUND_Y - h - 60, 92, 80), 2)
        for x, s, col in ((330, 34, (210, 90, 70)), (880, 26, (220, 160, 80)), (600, 22, (210, 90, 70))):
            ink_rect(bg, (238, 226, 200), (x - s // 4, GROUND_Y - s, s // 2, s), 2, 3)
            ink_ellipse(bg, col, (x - s, GROUND_Y - s - s // 2 - 4, 2 * s, s + 4), 2)
            for dx in (-s // 2, s // 3):
                pygame.draw.circle(bg, CREAM, (x + dx, GROUND_Y - s - 4), max(2, s // 8))
        # Ground.
        pygame.draw.rect(bg, SOIL, (0, GROUND_Y, W, H - GROUND_Y))
        pygame.draw.rect(bg, GRASS, (0, GROUND_Y - 6, W, 22))
        for x in range(0, W, 14):
            pygame.draw.line(bg, GRASS_DARK, (x, GROUND_Y + 14), (x + 6, GROUND_Y - 10 - (x * 7) % 9), 2)
        pygame.draw.line(bg, INK, (0, GROUND_Y + 16), (W, GROUND_Y + 16), 3)
        for _ in range(60):
            x, y = rng.randint(0, W), rng.randint(GROUND_Y + 24, H)
            pygame.draw.ellipse(bg, (128, 92, 58), (x, y, rng.randint(6, 16), rng.randint(3, 6)))
        return bg

    def _make_vignette(self) -> pygame.Surface:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        alpha = np.clip((d - 0.65) / 0.75, 0.0, 1.0) ** 1.6 * 200
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.fill((40, 22, 8, 0))
        pygame.surfarray.pixels_alpha(surf)[:] = alpha.T.astype(np.uint8)
        return surf

    def _make_grain(self, n: int) -> list[pygame.Surface]:
        rng = np.random.default_rng(3)
        out = []
        for _ in range(n):
            noise = rng.integers(0, 2, size=(W // 2, H // 2)).astype(np.uint8)
            alpha = (rng.random((W // 2, H // 2)) * 26).astype(np.uint8)
            s = pygame.Surface((W // 2, H // 2), pygame.SRCALPHA)
            rgb = pygame.surfarray.pixels3d(s)
            rgb[:] = (noise * 255)[..., None]
            del rgb
            pygame.surfarray.pixels_alpha(s)[:] = alpha
            out.append(pygame.transform.scale(s, (W, H)))
        return out

    # ------------------------------------------------------------------ frame
    def render(self, world: World, episode_reward: float = 0.0, action: spec.Action | None = None,
               overlay: dict[str, Any] | None = None) -> np.ndarray:
        scr = self.screen
        scr.blit(self.background, (0, 0))
        frame = world.frame
        self._draw_shadows(world)
        self._draw_boss(world, frame)
        for q in world.qmarks:
            bob = 6 * math.sin(world.frame * 0.1 + q.idx)
            self.outlined_text("?", 48, PINK, (sx(q.x), sy(q.y + bob)), center=True, outline=3)
        self._draw_player(world, frame)
        self._draw_bullets(world, frame)
        if self.show_hitboxes:
            self._draw_hitboxes(world)
        self._draw_film(frame)
        self._draw_hud(world, episode_reward)
        self._draw_panel(action or spec.Action(), episode_reward, overlay)
        if world.won or world.lost:
            self._draw_banner("A KNOCKOUT!" if world.won else "YOU DIED!", world.won)
        # (W, H, 3) surface array -> (H, W, 3) image
        return pygame.surfarray.array3d(scr).transpose(1, 0, 2)

    # ------------------------------------------------------------------ world
    def _draw_shadows(self, world: World) -> None:
        b = world.boss
        if b.dead:
            return
        w = int(b.half_w * 2 * max(0.35, 1.0 - b.bottom / 900.0))
        shadow = pygame.Surface((w, 18), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (40, 30, 20, 90), (0, 0, w, 18))
        self.screen.blit(shadow, (sx(b.x) - w // 2, GROUND_Y - 6))
        p = world.player
        pw = 44
        shadow = pygame.Surface((pw, 10), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (40, 30, 20, 90), (0, 0, pw, 10))
        self.screen.blit(shadow, (sx(p.x) - pw // 2, GROUND_Y - 4))

    def _draw_player(self, world: World, frame: int) -> None:
        p = world.player
        if p.invuln > 0.0 and p.hp > 0 and (frame // 4) % 2 == 0 and not p.parrying:
            return  # i-frame flicker
        s = self.screen
        f = p.facing
        x, y = sx(p.x), sy(p.y)
        if p.dashing:
            for i in range(1, 4):
                pygame.draw.line(s, INK, (x - f * (20 + 16 * i), y - 30 - 8 * i), (x - f * (44 + 16 * i), y - 30 - 8 * i), 2)
        if p.ducking and p.grounded:
            self._draw_player_duck(p, x, y, f)
            return
        run = abs(p.vx) > 1 and p.grounded and not p.dashing
        swing = int(8 * math.sin(frame * 0.5)) if run else 0
        if p.grounded:
            # legs + shoes
            for d in (-1, 1):
                fx = x + d * 7 + (swing * d)
                pygame.draw.line(s, INK, (x + d * 5, y - 22), (fx, y - 6), 4)
                ink_ellipse(s, (120, 70, 40), (fx - 9 + f * 3, y - 10, 18, 10), 2)
        else:
            pygame.draw.line(s, INK, (x - 5, y - 22), (x - 10, y - 10), 4)
            pygame.draw.line(s, INK, (x + 5, y - 22), (x + 9, y - 12), 4)
            ink_ellipse(s, (120, 70, 40), (x - 18, y - 16, 16, 9), 2)
            ink_ellipse(s, (120, 70, 40), (x + 2, y - 18, 16, 9), 2)
        look = self.look
        accent, hw, hh, hd = look["accent"], look["head_hw"], look["head_h"], look["handle"]
        ink_rect(s, accent, (x - 13, y - 34, 26, 14), 2, 5)  # shorts
        ink_rect(s, (30, 26, 30), (x - 11, y - 46, 22, 14), 2, 4)  # torso
        # arm: finger-gun when shooting
        if p.shooting or p.ex_frames > 0:
            _, _, dx, dy = p._shot_pose()  # point the finger-gun along the shot
            hx, hy = x + int(26 * dx), y - 44 - int(26 * dy)
            pygame.draw.line(s, INK, (x, y - 42), (hx, hy), 4)
            ink_ellipse(s, CREAM, (hx - 6, hy - 6, 12, 12), 2)
        else:
            pygame.draw.line(s, INK, (x + f * 6, y - 42), (x + f * 14, y - 30), 4)
            ink_ellipse(s, CREAM, (x + f * 14 - 6, y - 34, 12, 12), 2)
        # cup / mug head
        top = y - 44 - hh
        head = pygame.Rect(x - hw, top, 2 * hw, hh)
        pygame.draw.ellipse(s, INK, (x - f * (hw + 9) - hd // 2, top + 10, hd, hd + 2), 4)  # handle
        ink_ellipse(s, (252, 250, 244), head, 3)
        ink_ellipse(s, look["drink"], (x - hw + 2, top - 4, 2 * hw - 4, 12), 2)  # rim + drink
        pygame.draw.line(s, (240, 240, 240), (x + f * 2, top), (x + f * 16, top - 24), 6)  # straw
        pygame.draw.line(s, accent, (x + f * 5, top - 5), (x + f * 9, top - 12), 6)
        pygame.draw.line(s, accent, (x + f * 12, top - 17), (x + f * 15, top - 22), 6)
        for ex in (4, 15):
            ink_ellipse(s, INK, (x + f * ex - 4, top + 10, 7, 13), 1)
            pygame.draw.polygon(s, (252, 250, 244), [(x + f * ex, top + 14), (x + f * ex + 4, top + 10), (x + f * ex + 4, top + 18)])
        pygame.draw.circle(s, accent, (x + f * (hw - 2), top + 23), 4)  # nose
        if p.parrying and not p.grounded:
            pygame.draw.circle(s, PINK, (x, y - 50), 46, 3)

    def _draw_player_duck(self, p, x: int, y: int, f: int) -> None:
        """Crouched pose: short body, head low, gun forward at shin height."""
        s = self.screen
        for d in (-1, 1):
            ink_ellipse(s, (120, 70, 40), (x + d * 10 - 9, y - 10, 18, 10), 2)
        look = self.look
        hw = look["head_hw"]
        ink_rect(s, look["accent"], (x - 15, y - 22, 30, 14), 2, 5)
        if p.shooting:
            pygame.draw.line(s, INK, (x, y - 20), (x + f * 30, y - 22), 4)
            ink_ellipse(s, CREAM, (x + f * 30 - 6, y - 28, 12, 12), 2)
        head = pygame.Rect(x - hw + f * 6, y - 22 - look["head_h"], 2 * hw, look["head_h"])
        pygame.draw.ellipse(s, INK, (x - f * (hw + 3) - 9, head.y + 8, 18, 20), 4)
        ink_ellipse(s, (252, 250, 244), head, 3)
        ink_ellipse(s, look["drink"], (head.x + 2, head.y - 4, 2 * hw - 4, 12), 2)
        pygame.draw.line(s, (240, 240, 240), (head.centerx, head.y), (head.centerx + f * 14, head.y - 20), 6)
        for ex in (4, 15):
            ink_ellipse(s, INK, (head.centerx + f * ex - 4, head.y + 12, 7, 12), 1)

    def _draw_bullets(self, world: World, frame: int) -> None:
        s = self.screen
        for b in world.bullets:
            x, y = sx(b.x), sy(b.y)
            d = 1 if b.vx > 0 else -1
            sp = math.hypot(b.vx, b.vy) or 1.0
            ux, uy = b.vx / sp, b.vy / sp
            if b.ex:
                r = int(b.half_w)
                glow = pygame.Surface((r * 3, r * 3), pygame.SRCALPHA)
                pygame.draw.circle(glow, (110, 180, 255, 70), (r * 3 // 2, r * 3 // 2), int(r * 1.5))
                s.blit(glow, (x - r * 3 // 2, y - r * 3 // 2))
                ink_ellipse(s, (120, 190, 255), (x - r, y - r, 2 * r, 2 * r), 3)
                pygame.draw.circle(s, (230, 245, 255), (x - d * 8, y - 8), r // 3)
                ang = frame * 0.4
                pygame.draw.arc(s, INK, (x - r + 6, y - r + 6, 2 * r - 12, 2 * r - 12), ang, ang + 2.0, 2)
            else:
                pygame.draw.line(s, (170, 210, 240), (x - int(ux * 30), y + int(uy * 30)),
                                 (x - int(ux * 10), y + int(uy * 10)), 4)
                ink_ellipse(s, (200, 235, 255), (x - 9, y - 7, 18, 14), 2)

    def _draw_boss(self, world: World, frame: int) -> None:
        b = world.boss
        if b.dead:
            self._draw_tomb(world, frame, broken=True)
            return
        if b.phase == BossPhase.TOMB and not (b.state == BossMove.TRANSFORM and b.transform == DYING):
            self._draw_tomb(world, frame)
        else:
            self._draw_slime(world, frame)

    def _draw_slime(self, world: World, frame: int) -> None:
        b = world.boss
        s = self.screen
        big = b.phase != BossPhase.SMALL
        dying = b.state == BossMove.TRANSFORM and b.transform == DYING
        bx_, by_, hw, hh, _ = b.body()
        # squash & stretch
        k = 1.0
        if b.state == BossMove.JUMP:
            k = 1.0 + max(-0.12, min(0.15, b.vy / 9000.0))
        elif b.state == BossMove.IDLE and b.state_time < 0.12:
            k = 0.85 + b.state_time
        elif b.state == BossMove.PUNCH_WINDUP:
            k = 0.92
        elif b.state == BossMove.TRANSFORM and b.transform == GROW:
            k = 1.0 + 0.08 * math.sin(b.state_time * 18.0)
        elif dying:
            k = max(0.45, 1.0 - b.state_time * 0.5)
        w, h = hw * 2 / math.sqrt(k), hh * 2 * k
        cx = sx(bx_)
        bottom = sy(max(0.0, by_ - hh))  # circles dip below the ground line
        body = pygame.Rect(int(cx - w / 2), int(bottom - h), int(w), int(h))
        color, dark = (GOO, GOO_DARK) if not dying else ((150, 170, 190), (110, 120, 140))
        if b.state == BossMove.PUNCH_WINDUP and (frame // 3) % 2 == 0:
            color = (150, 200, 240)  # telegraph flash
        # punch hitboxes from the live track: arm boxes, then the glove/thrown circle
        fists = b.fist_tuples()
        for _, fx, fy, fhw, fhh, _, _, rnd in fists:
            if not rnd and fhw > 2:
                ink_rect(s, color, (sx(fx - fhw), sy(fy + fhh), int(2 * fhw), int(2 * fhh)), 3, int(min(fhw, fhh)))
        for _, fx, fy, fhw, _, _, _, rnd in fists:
            if rnd:
                glove = RED if big else GOO_DARK
                ink_ellipse(s, glove, (sx(fx) - int(fhw), sy(fy) - int(fhw), int(2 * fhw), int(2 * fhw)), 3)
                pygame.draw.arc(s, INK, (sx(fx) - int(fhw * 0.5), sy(fy) - int(fhw * 0.5), int(fhw), int(fhw)), 0.5, 2.6, 3)
        ink_ellipse(s, color, body, 4)
        pygame.draw.ellipse(s, dark, (body.x + 6, body.centery, body.w - 12, body.h // 2 - 4), 0)
        pygame.draw.ellipse(s, INK, body, 4)
        hl = pygame.Rect(body.x + body.w * 0.18, body.y + body.h * 0.12, body.w * 0.22, body.h * 0.16)
        pygame.draw.ellipse(s, GOO_LIGHT, hl)
        # face, looking at the player
        look = 1 if world.player.x > b.x else -1
        er = max(7, int(hw * 0.2))
        ey = body.y + int(body.h * 0.36)
        for ex in (-0.28, 0.12):
            ex_px = cx + int(ex * w) + look * int(w * 0.06)
            if dying:
                pygame.draw.line(s, INK, (ex_px - er // 2, ey - er // 2), (ex_px + er // 2, ey + er // 2), 4)
                pygame.draw.line(s, INK, (ex_px - er // 2, ey + er // 2), (ex_px + er // 2, ey - er // 2), 4)
            else:
                ink_ellipse(s, (255, 255, 255), (ex_px - er // 2, ey - er, er, int(er * 1.6)), 2)
                pygame.draw.ellipse(s, INK, (ex_px - er // 4 + look * 2, ey - er // 2, er // 2, er // 1.4))
        mouth_y = body.y + int(body.h * 0.62)
        mw = int(w * 0.36)
        mx = cx + look * int(w * 0.05) - mw // 2
        ink_ellipse(s, (120, 30, 40), (mx, mouth_y - mw // 5, mw, mw // 2), 3)
        pygame.draw.rect(s, (255, 255, 255), (mx + mw // 5, mouth_y - mw // 5 + 2, mw * 3 // 5, mw // 8))
        if big:
            # the famous moustache
            my = mouth_y - mw // 4
            for d in (-1, 1):
                mcx = cx + look * int(w * 0.05) + d * int(mw * 0.42)
                ink_ellipse(s, INK, (mcx - mw // 3, my - mw // 9, mw * 2 // 3, mw // 4), 1)
                pygame.draw.circle(s, INK, (mcx + d * mw // 3, my - mw // 9), mw // 9, 3)
            # eyebrows
            for ex in (-0.28, 0.12):
                ex_px = cx + int(ex * w) + look * int(w * 0.06)
                pygame.draw.line(s, INK, (ex_px - er, ey - er - 8), (ex_px + er, ey - er - 2 * (1 if ex < 0 else -1) - 8), 5)
        if b.state == BossMove.TRANSFORM and b.transform == GROW and (frame // 5) % 2 == 0:
            self.outlined_text("!", 40, (250, 220, 90), (cx, body.y - 30), center=True)

    def _draw_tomb(self, world: World, frame: int, broken: bool = False) -> None:
        """Tombstone sprite (200 x 440, standing on its transform) driven by the body circle:
        during the smash it rises, then crashes down flat while the circle swells."""
        b = world.boss
        s = self.screen
        tc = self.cfg.tomb
        bx_, by_, r, _, _ = b.body()
        cx = sx(bx_)
        hw, height = 100, 440
        standing = r < 150.0
        if b.state == BossMove.TOMB_SLAM_WINDUP or (b.state == BossMove.TRANSFORM and b.transform == FALLING):
            # Telegraph: where the smash lands (circle r ~250 reaching the floor).
            mark = pygame.Surface((460, 24), pygame.SRCALPHA)
            a = 110 + int(90 * (0.5 + 0.5 * math.sin(frame * 0.6)))
            pygame.draw.ellipse(mark, (190, 30, 20, a), mark.get_rect())
            pygame.draw.ellipse(mark, (*INK, 200), mark.get_rect(), 2)
            s.blit(mark, (cx - 230, GROUND_Y - 10))
        if b.state == BossMove.TRANSFORM and b.transform == FALLING and b.crushed_slime_alive:
            # Big Goopy, dead but not yet flattened.
            gw = int(self.cfg.big.body_radius * 1.6)
            ink_ellipse(s, (150, 170, 190), (cx - gw // 2, GROUND_Y - 60, gw, 66), 3)
            for d in (-1, 1):
                ex = cx + d * gw // 6
                pygame.draw.line(s, INK, (ex - 6, GROUND_Y - 44), (ex + 6, GROUND_Y - 32), 3)
                pygame.draw.line(s, INK, (ex - 6, GROUND_Y - 32), (ex + 6, GROUND_Y - 44), 3)
        surf = pygame.Surface((hw * 2 + 16, height + 16), pygame.SRCALPHA)
        col = STONE if not broken else (130, 126, 118)
        top = pygame.Rect(8, 8, hw * 2, hw * 2)
        body_r = pygame.Rect(8, 8 + hw, hw * 2, height - hw)
        pygame.draw.ellipse(surf, col, top)
        pygame.draw.rect(surf, col, body_r)
        pygame.draw.arc(surf, INK, top, 0, math.pi, 4)
        pygame.draw.line(surf, INK, (8, 8 + hw), (8, body_r.bottom), 4)
        pygame.draw.line(surf, INK, (8 + hw * 2 - 1, 8 + hw), (8 + hw * 2 - 1, body_r.bottom), 4)
        pygame.draw.line(surf, INK, (8, body_r.bottom - 2), (8 + hw * 2, body_r.bottom - 2), 4)
        pygame.draw.rect(surf, STONE_DARK, (8, body_r.bottom - 24, hw * 2, 22))
        rip = self.text("R.I.P.", 30, INK)
        surf.blit(rip, (8 + hw - rip.get_width() // 2, 60))
        fy = 130
        if broken:
            pygame.draw.line(surf, INK, (hw - 20, 40), (hw + 10, 220), 3)
            pygame.draw.line(surf, INK, (hw + 10, 220), (hw - 6, 300), 3)
        else:
            for d in (-1, 1):
                ex = 8 + hw + d * hw // 2.6
                pygame.draw.ellipse(surf, (255, 255, 255), (ex - 12, fy, 24, 26))
                pygame.draw.ellipse(surf, INK, (ex - 12, fy, 24, 26), 2)
                pygame.draw.circle(surf, INK, (ex + (4 if world.player.x > b.x else -4), fy + 14), 6)
                pygame.draw.line(surf, INK, (ex - 16, fy - 6 - 6 * d), (ex + 16, fy - 6 + 6 * d), 5)
            pygame.draw.arc(surf, INK, (8 + hw - 30, fy + 44, 60, 30), 0.3, math.pi - 0.3, 4)
        pygame.draw.line(surf, STONE_DARK, (24, 250), (48, 290), 2)
        foot = sy(tc.rest_y + b.lift - tc.rest_y)  # sprite stands on the transform's ground point
        if standing:
            rise = max(0.0, by_ - (tc.rest_y + tc.body_offset_y)) if b.state == BossMove.TOMB_SLAM else 0.0
            tilt = -max(-8.0, min(8.0, b.vx / 80.0)) if b.state == BossMove.TOMB_MOVE else 0.0
            if b.state == BossMove.TOMB_SLAM_WINDUP:
                tilt = 3.0 * math.sin(frame * 1.3)
            if tilt:
                surf = pygame.transform.rotate(surf, tilt)
            s.blit(surf, (cx - surf.get_width() // 2, foot - int(rise) - surf.get_height() + 8))
        else:
            # Crashing down: lying flat, sinking with the circle.
            flat = pygame.transform.rotate(surf, 90 if world.player.x < b.x else -90)
            sink = max(0.0, min(80.0, (100.0 - by_) * 0.25))
            s.blit(flat, (cx - flat.get_width() // 2, GROUND_Y - flat.get_height() + 10 + int(sink)))
            for i, d in enumerate((-1, 1, -1, 1)):
                rr = 20 + 7 * i
                pygame.draw.circle(s, (220, 206, 176), (cx + d * (230 + 24 * i), GROUND_Y - rr // 2), rr)
                pygame.draw.circle(s, INK, (cx + d * (230 + 24 * i), GROUND_Y - rr // 2), rr, 2)

    def _draw_hitboxes(self, world: World) -> None:
        colors = {HazardKind.BODY: (220, 40, 40), HazardKind.FIST: (240, 120, 0), HazardKind.TOMB: (160, 0, 160),
                  HazardKind.OTHER: PINK}
        for kind, x, y, hw, hh, _, _, rnd in world.boss.hazard_tuples():
            if rnd:
                pygame.draw.circle(self.screen, colors[kind], (sx(x), sy(y)), int(hw), 2)
            else:
                pygame.draw.rect(self.screen, colors[kind], (sx(x - hw), sy(y + hh), int(2 * hw), int(2 * hh)), 2)
        b = world.boss
        if not b.dead and b.phase == BossPhase.TOMB and not b.smash_damage:  # bullet target only
            bx_, by_, br, _, _ = b.body()
            pygame.draw.circle(self.screen, (120, 120, 120), (sx(bx_), sy(by_)), int(br), 1)
        tr = self.cfg.transition.qmark_radius
        for q in world.qmarks:
            pygame.draw.circle(self.screen, PINK, (sx(q.x), sy(q.y)), int(tr), 2)
        cx, cy, hw, hh = world.player.box()
        pygame.draw.rect(self.screen, (0, 160, 60), (sx(cx - hw), sy(cy + hh), int(2 * hw), int(2 * hh)), 2)
        for bl in world.bullets:
            pygame.draw.rect(self.screen, (0, 90, 220),
                             (sx(bl.x - bl.half_w), sy(bl.y + bl.half_h), int(2 * bl.half_w), int(2 * bl.half_h)), 1)

    # ------------------------------------------------------------------ film look
    def _draw_film(self, frame: int) -> None:
        s = self.screen
        # Slight sepia wash.
        wash = pygame.Surface((W, H), pygame.SRCALPHA)
        wash.fill((150, 110, 60, 22))
        s.blit(wash, (0, 0))
        # Grain/scratches change at ~12 fps like real film (and keep the H.264 bitrate sane).
        film_frame = frame // GRAIN_HOLD
        s.blit(self.grain[film_frame % len(self.grain)], (0, 0))
        rng = random.Random(film_frame)
        for _ in range(rng.randint(0, 2)):  # scratches
            x = rng.randint(0, W)
            pygame.draw.line(s, (90, 70, 50), (x, 0), (x + rng.randint(-20, 20), H), 1)
        for _ in range(rng.randint(2, 6)):  # dust
            pygame.draw.circle(s, (60, 45, 30), (rng.randint(0, W), rng.randint(0, H)), rng.randint(1, 3))
        s.blit(self.vignette, (0, 0))

    # ------------------------------------------------------------------ HUD
    def _draw_hud(self, world: World, episode_reward: float) -> None:
        s = self.screen
        b, p = world.boss, world.player
        # Boss name + HP bar, segmented by phase.
        bar = pygame.Rect(300, 26, 600, 26)
        self.outlined_text(PHASE_NAMES[b.phase], 22, CREAM, (bar.centerx, 12), center=True)
        ink_rect(s, PAPER, bar.inflate(8, 8), 3, 6)
        frac = max(0.0, b.hp / b.max_hp)
        if frac > 0:
            col = (150, 150, 150) if not b.vulnerable else RED
            pygame.draw.rect(s, col, (bar.x, bar.y, int(bar.w * frac), bar.h), border_radius=4)
        hps = self.cfg.phase_hps  # bar drains right to left: tomb HP is the leftmost segment
        for cut in (hps[2] / b.max_hp, (hps[2] + hps[1]) / b.max_hp):
            x = bar.x + int(bar.w * cut)
            pygame.draw.line(s, INK, (x, bar.y - 4), (x, bar.bottom + 4), 3)
        self.outlined_text(f"{b.hp:.0f} / {b.max_hp:.0f}", 16, CREAM, (bar.centerx, bar.bottom + 14), center=True)
        # Clock.
        t = world.time
        self.outlined_text(f"{int(t // 60):02d}:{t % 60:04.1f}", 26, CREAM, (24, 18))
        # HP card (bottom-left, like the game).
        card = pygame.Rect(24, H - 62, 92, 42)
        flash = p.hp == 1 and (world.frame // 15) % 2 == 0
        ink_rect(s, RED if flash else (250, 246, 236), card, 3, 6)
        hp_txt = "DEAD" if p.hp <= 0 else f"HP. {p.hp}"
        self.outlined_text(hp_txt, 22, (250, 246, 236) if flash else RED, card.center, center=True, outline=1)
        # Super cards.
        for i in range(int(spec.SUPER_METER_MAX)):
            r = pygame.Rect(132 + i * 26, H - 60, 20, 32)
            fill = max(0.0, min(1.0, p.meter - i))
            ink_rect(s, (240, 230, 210), r, 2, 3)
            if fill > 0:
                fh = int((r.h - 4) * fill)
                color = PINK if fill >= 1.0 else (210, 150, 170)
                pygame.draw.rect(s, color, (r.x + 2, r.bottom - 2 - fh, r.w - 4, fh), border_radius=2)
            pygame.draw.rect(s, INK, r, 2, border_radius=3)
        self.outlined_text(f"PARRIES {world.parries}", 16, CREAM, (272, H - 54))

    def _fit_text(self, txt: str, max_w: int, sizes=(18, 16, 14, 12, 11), color=INK) -> pygame.Surface:
        """Largest font size that fits ``max_w``; truncate with an ellipsis as a last resort."""
        for size in sizes:
            surf = self.text(txt, size, color)
            if surf.get_width() <= max_w:
                return surf
        while len(txt) > 4 and self.text(txt + "...", sizes[-1], color).get_width() > max_w:
            txt = txt[:-1]
        return self.text(txt + "...", sizes[-1], color)

    def _draw_panel(self, a: spec.Action, episode_reward: float, overlay: dict[str, Any] | None) -> None:
        """Side panel: reward, input, and the overlay.

        Structured overlay keys (from the training pipeline): ``title`` (caption bar),
        ``value`` (gauge), ``probs`` ({head: probabilities} bar rows); ``action`` is
        skipped (the INPUT row shows it). Anything else is drawn as key/value text.
        """
        s = self.screen
        ov = dict(overlay or {})
        title = ov.pop("title", None)
        value = ov.pop("value", None)
        probs = ov.pop("probs", None)
        ov.pop("action", None)
        probs = probs if isinstance(probs, dict) else ({} if probs is None else {"probs": probs})
        lines = list(ov.items())
        pw = 300
        panel_h = (150 + (34 if value is not None else 0) + 24 * len(probs) + (8 if probs else 0) + 24 * len(lines)
                   + sum(26 for _, v in lines if isinstance(v, (list, tuple, np.ndarray))))
        panel = pygame.Surface((pw, panel_h), pygame.SRCALPHA)
        panel.fill((250, 240, 214, 205))
        pygame.draw.rect(panel, (*INK, 255), panel.get_rect(), 3, border_radius=8)
        x0, y0 = W - pw - 24, 70
        s.blit(panel, (x0, y0))
        s.blit(self.text("REWARD", 16, INK), (x0 + 14, y0 + 10))
        s.blit(self.text(f"{episode_reward:+.2f}", 22, RED), (x0 + 120, y0 + 6))
        s.blit(self.text("INPUT", 16, INK), (x0 + 14, y0 + 44))
        # Action glyphs, lit when active: the stick on one row, the buttons on the next.
        rows = [[("<", a.move < 0), (">", a.move > 0), ("UP", a.aim > 0), ("DOWN", a.aim < 0)],
                [("JUMP", a.jump), ("DASH", a.dash), ("SHOOT", a.shoot), ("EX", a.ex)]]
        for row_i, keys in enumerate(rows):
            bx, by = x0 + 14, y0 + 68 + row_i * 34
            for label, on in keys:
                surf = self.text(label, 14, CREAM if on else INK)
                w = max(26, surf.get_width() + 12)
                r = pygame.Rect(bx, by, w, 28)
                ink_rect(s, RED if on else (226, 214, 186), r, 2, 6)
                s.blit(surf, (r.centerx - surf.get_width() // 2, r.centery - surf.get_height() // 2))
                bx += w + 5
        y = y0 + 144
        if value is not None:
            v = float(value)
            s.blit(self.text("VALUE", 15, INK), (x0 + 14, y + 4))
            s.blit(self.text(f"{v:+.2f}", 15, RED), (x0 + 228, y + 4))
            gauge = pygame.Rect(x0 + 84, y + 6, 136, 16)
            lo, hi = -10.0, 60.0  # display range; values outside are clipped
            zero = gauge.x + int(gauge.w * (0.0 - lo) / (hi - lo))
            xv = gauge.x + int(gauge.w * (max(lo, min(hi, v)) - lo) / (hi - lo))
            pygame.draw.rect(s, (226, 214, 186), gauge)
            pygame.draw.rect(s, RED if v >= 0 else GOO_DARK, (min(zero, xv), gauge.y, abs(xv - zero), gauge.h))
            pygame.draw.line(s, INK, (zero, gauge.y - 2), (zero, gauge.bottom + 1), 2)
            pygame.draw.rect(s, INK, gauge, 2)
            y += 34
        for head, pr in probs.items():
            vals = [float(v) for v in np.asarray(pr, dtype=float).ravel()]
            s.blit(self.text(str(head)[:6].upper(), 13, INK), (x0 + 14, y + 3))
            if len(vals) == 2:  # binary head: show P(press)
                labels, vals = [""], vals[1:]
            elif len(vals) == 3 and head == "move":
                labels = ["<", "-", ">"]
            elif len(vals) == 3 and head == "aim":
                labels = ["v", "-", "^"]
            else:
                labels = [""] * len(vals)
            n = max(1, len(vals))
            cw = 196 // n
            for i, (lab, v) in enumerate(zip(labels, vals)):
                cx0 = x0 + 84 + i * cw
                pad = 12 if lab else 0
                if lab:
                    s.blit(self.text(lab, 12, INK), (cx0, y + 3))
                bar = pygame.Rect(cx0 + pad, y + 5, cw - pad - 4, 14)
                pygame.draw.rect(s, (226, 214, 186), bar)
                pygame.draw.rect(s, RED, (bar.x, bar.y, int(bar.w * max(0.0, min(1.0, v))), bar.h))
                pygame.draw.rect(s, INK, bar, 1)
                if n == 1:
                    s.blit(self.text(f"{v:.0%}", 12, INK), (bar.right + 6 - 40, y + 4))
            y += 24
        if probs:
            y += 8
        for key, val in lines:
            s.blit(self.text(str(key)[:18], 15, INK, bold=True), (x0 + 14, y))
            if isinstance(val, (list, tuple, np.ndarray)):
                vals = [float(v) for v in np.asarray(val, dtype=float).ravel()][:24]
                bw = max(4, min(22, 268 // max(1, len(vals)) - 3))
                for i, v in enumerate(vals):
                    hgt = int(24 * max(0.0, min(1.0, v)))
                    pygame.draw.rect(s, RED, (x0 + 14 + i * (bw + 3), y + 46 - hgt, bw, hgt))
                    pygame.draw.rect(s, INK, (x0 + 14 + i * (bw + 3), y + 22, bw, 24), 1)
                y += 50
            else:
                txt = f"{val:.3f}" if isinstance(val, float) else str(val)
                surf = self._fit_text(txt, 150, sizes=(15, 13, 11), color=RED)
                s.blit(surf, (x0 + pw - 14 - surf.get_width(), y))
                y += 24
        if title:
            # Caption bar, bottom-right, shrinking to fit.
            surf = self._fit_text(str(title), 560)
            bar = pygame.Rect(0, 0, surf.get_width() + 28, 34)
            bar.bottomright = (W - 24, H - 22)
            ink_rect(s, PAPER, bar, 3, 6)
            s.blit(surf, (bar.centerx - surf.get_width() // 2, bar.centery - surf.get_height() // 2))

    def _draw_banner(self, msg: str, won: bool) -> None:
        self.outlined_text(msg, 84, (250, 220, 90) if won else RED, (W // 2, H // 2 - 40), center=True, outline=4)

    def close(self) -> None:
        self._text_cache.clear()
