# Goopy Le Grande — game constants (Regular difficulty)

Machine-readable copy: [`data/goopy_constants.json`](../data/goopy_constants.json).

**Source.** Cuphead Steam build 20241530 (DLC-era), Unity **2017.4.9f1**, **x64**, **Mono**
backend. `Cuphead_Data/Managed/Assembly-CSharp.dll` decompiled with `ilspycmd 9.1` into
`tools/decompiled/` (gitignored — never committed). Scene data read from
`Cuphead_Data/level20` (`scene_level_slime`, build index 20) with
[`tools/unity_probe.py`](../tools/unity_probe.py) (UnityPy + TypeTreeGeneratorAPI).
Regular = `Level.Mode.Normal`.

**Units.** Unity world units (the 1280×720 camera shows 1280 units across, so 1 unit ≈ 1 px),
seconds, units/s. The arena centre is x = 0 and `Level.Ground` is y = −250, so
`x_spec = x_unity` and `y_spec = y_unity + 250` (the bridge calibrates the exact ground at runtime).

**Time step.** `TimeManager.Fixed Timestep` = 0.0166667 s. All gameplay (player motor,
boss jumps, weapon cooldowns) runs in `FixedUpdate`, via `CupheadTime.FixedDelta`
= `Time.fixedDeltaTime * CupheadTime.GlobalSpeed`.

## Boss HP and phases

| Value | Regular | Source |
|---|---|---|
| Total HP | **1200** | `LevelProperties.Slime.GetMode(Normal)`: `hp = 1200` |
| Phase 2 (Big) starts when HP fraction < | **0.76** (≤ 912 HP left) | `State(0.76f, …, States.BigSlime, …)` |
| Phase 3 (Tombstone) starts when fraction < | **0.31** (≤ 372 HP left) | `State(0.31f, …, States.Tombstone, …)` |
| HP per phase | 288 / 540 / 372 | derived |
| First attack delay after "Knockout!" intro | 1.0 s | `SlimeLevel.slimePattern_cr` |

State selection (`AbstractLevelProperties.DealDamage`): after every hit the state index becomes the
last state whose `healthTrigger` > `CurrentHealth / TotalHealth`. Win when `CurrentHealth <= 0`.
No damage cap per phase in code: excess damage carries into the next phase.

**Phase 1 → 2.** `OnStateChanged` sets `wantsToTransform` on the small slime; the transform only
starts *on a landing with |x| < 350* (`SlimeLevelSlime.jump_cr`, `TRANSFORM_MAX_X`). Until
then small Goopy keeps hopping. The big slime then spawns at the small slime's position and its
first punch comes after exactly `bigSlimeInitialJumpPunchCount` = **2** jumps.

**Phase 2 → 3.** `bigSlime.DeathTransform()`; if airborne it dies on landing. After the
`Death_Loop` animation starts, wait **3.5 s**, then the tombstone drops from y = 550 to −80 in
**0.2 s** (linear) at the big slime's x, crushing it.

## Phase 1 — small Goopy (`SlimeLevelSlime`, `isBig = 0`)

| Value | Regular | Source |
|---|---|---|
| Spawn x / ground y | 369 / −247 | scene `Slime` transform; `onGroundY = transform.y` in `Awake` |
| Horizontal turn-around bound | ±560 | serialized `maxX` |
| Punch allowed when x in | (−1000, 200) toward player (mirrored) | serialized `punchMinX/punchMaxX`, `inPunchPosition()` |
| Ground delay between jumps | 0.2 s | `Jump.groundDelay` |
| High jump vy / vx / gravity | 2350 / 840 / 6500 | `Jump.highJump*` |
| Low jump vy / vx / gravity | 2000 / 780 / 6500 | `Jump.lowJump*` |
| Jumps before a punch | uniform int in [5, 8] (`MinMax.RandomInt`), then the punch waits until `inPunchPosition()` | `Jump.numJumps` |
| Jump pattern | `HJ,LJ,HJ,LJ,D.5, LJ,RJ,HJ,LJ,HJ,LJ,LJ,HJ,RJ,LJ,LJ` | `Jump.patternString` |
| Punch windup (`Punch_Pre_Hold`) | 0.65 s + animation | `Punch.preHold` |
| Punch extended (`Punch_Hold`) | 0.4 s + animation | `Punch.mainHold` |
| Body hitbox | circle r = 80, offset (0, 75) | scene `CircleCollider2D` |
| Question marks (parryable) | 3 circles r = 28, enabled by animation events | `Slime/SpikerParent*/Spiker`, `SlimeLevelQuestionMark` |

Jump patterns: start at a random index, then cycle. `H` = high, `L` = low, `R` = random
high/low, `Dx` = set the next ground delay to *x* s. Quirk: entries with a leading space (`" LJ"`)
and the typo `HL` are read by first character, so `" LJ"` is a low jump and `HL` a high jump.

Jump integration (`jump_cr`, per fixed tick): `vy -= g·dt; pos += (±vx, vy)·dt` (semi-implicit
Euler) until the slime is back at `onGroundY`. Hitting `±maxX` mirrors direction; if this happens
on the way down, vx is zeroed for the rest of the jump. Approximate arcs: high jump apex ≈ 425,
airtime ≈ 0.72 s, range ≈ 610; low jump apex ≈ 308, airtime ≈ 0.62 s, range ≈ 480.

The punch faces the targeted player (`punchDirection` from player x at punch time). Its hitbox is
`ArmCollider` (`BoxCollider2D`, size animated by the punch clips — read live by the bridge).

## Phase 2 — big Goopy (`SlimeLevelSlime`, `isBig = 1`)

| Value | Regular | Source |
|---|---|---|
| Turn-around bound | ±500 | serialized `maxX` |
| Punch allowed when x in | (−425, 200) toward player | serialized `punchMinX/punchMaxX` |
| Ground delay | 0.3 s | `Jump.groundDelay` |
| High jump vy / vx / gravity | 2850 / 820 / 7500 | `Jump.highJump*` |
| Low jump vy / vx / gravity | 2250 / 760 / 7500 | `Jump.lowJump*` |
| Jumps before a punch | [5, 8]; first punch after exactly 2 | `numJumps`, `bigSlimeInitialJumpPunchCount` |
| Jump pattern | `HJ,HJ,LJ,HJ,RJ,LJ,D.5, LJ,HL,LJ,RJ,HJ,LJ,LJ,HJ,LJ,LJ` | `Jump.patternString` |
| Punch pre-hold / main hold | 0.65 s / 0.4 s | `Punch` |
| Body hitboxes | circles r = 80 and r = 160, offset (0, 130) | scene `BigSlime` colliders |

Arcs: high apex ≈ 540, airtime ≈ 0.76 s, range ≈ 620; low apex ≈ 338, airtime ≈ 0.6 s, range ≈ 455.

## Phase 3 — tombstone (`SlimeLevelTombstone`)

| Value | Regular | Source |
|---|---|---|
| Rest y | −80 | `const onGroundY` |
| Patrol bound | ±500 | `const maxX` |
| Move speed | 950 u/s (average per leg) | `Tombstone.moveSpeed` |
| Movement | `easeInOutSine` from current x to ±500; leg time = \|Δx\| / 950 | `move_cr` |
| Time until it wants to smash | uniform [2, 4] s | `Tombstone.attackDelay` |
| Smash target | `player.center.x + offset`, offsets cycle `-130,70,0,130,-70,0` | `attackOffsetString` |
| Smash anticipation (`Smash_Pre_Hold`) | 0.35 s + animation | `anticipationHold` |
| Body hitbox | circle r = 100, offset (0, 170) → spans y_spec ≈ 210–440 while moving; standing peashots (y 64–104) **miss** (measured 0 damage); during the `smash` clip r grows to ≈ 260 and reaches the ground | scene `Tombstone` collider; live: `data/live_hitboxes.json` |

Smash damage is gated by `dealDamage`, which animation events (`EnableDamageDealer` /
`DisableDamageDealer`) toggle during the slam, so outside slams the tombstone body does **not**
hurt. Once it wants to smash, it slides toward the next wall and stops as soon as it passes
`targetX`, then smashes. Tiny slimes (`SlimeLevelTinySlime`) only spawn if `SlimeLevelSlime.TINIES`,
which is always false in this build.

## Player (`LevelPlayerMotor.Properties` — code defaults; `OnAwake` does `properties = new Properties()`, so prefab values are ignored)

| Value | Value | Source |
|---|---|---|
| Max HP | 3 | `PlayerStatsManager.HEALTH_MAX` |
| Run speed | 490 u/s | `moveSpeed` |
| Jump initial vy | 1223.1 u/s | `jumpPower = −0.755` × `maxSpeedY = 1620` (see below) |
| Jump hold | 0.01–0.16 s (vy held constant while held) | `jumpHoldMin/Max` |
| Gravity | 5991.8 u/s² | `maxSpeedY / (timeToMaxY·60)` per tick on the normalised velocity |
| Max fall speed | 1620 u/s | `velocityManager.y` clamped to 1 |
| Full-hold jump | apex ≈ 283, airtime ≈ 0.667 s | tick-exact replay of `HandleJumping/HandleFalling/Move` |
| Tap jump | apex ≈ 134, airtime ≈ 0.433 s | same |
| Coyote time / input buffer | 0.0834 s / 0.134 s | `JUMP_BUFFER_TIME`, `INPUT_BUFFER_TIME` |
| Dash | 1100 u/s for 0.3 s = 330 u; vy = 0 during dash | `dashSpeed`, `dashTime` |
| Dash limits | 1 air dash until grounded; 0.1 s cooldown after a ground dash | `HandleDash`, `DASH_COOLDOWN_DURATION` |
| Dash i-frames | **none** (only with Smoke Bomb charm) | `LevelPlayerController.CanTakeDamage` |
| I-frames after hit | 2.0 s | `PlayerDamageReceiver.TakeDamage` → `Invulnerable(2f)` |
| Hit stun / knockback | 0.3 s / 300 u/s, knocked up with vy 972 | `hitStunTime`, `hitKnockbackPower`, `hitJumpPower` |
| Parry bounce | vy 1620 | `parryPower = −1` |
| Hurtbox (standing) | 50 × 105, centre (0, 62) above feet | `LevelPlayerColliderManager` |
| Hurtbox (air / duck) | 50 × 50 @ (0, 50) / 50 × 35 @ (0, 27) | same |
| Spawn | (−410, −245) single player | scene `Level.spawns.playerOneSingle` |

Normalised vertical velocity: the motor keeps `velocityManager.y ∈ [−10, 1]` and the world
vy is `−1620 · y` (`VelocityManager.Total`, linear ease). Gravity adds `1620/438 · dt` to y per
tick. Order per tick: `HandleInput` (jump/dash/walk) → `HandleFalling` → `Move` → raycasts.

## Weapons and super meter

| Value | Value | Source |
|---|---|---|
| Peashooter damage | 4 per bullet | `WeaponProperties.LevelWeaponPeashot.Basic.damage` |
| Peashooter nominal rate | 0.11 s | `Basic.rapidFireRate` |
| **Peashooter real interval** | **8 ticks = 0.1333 s → 30 DPS** | `AbstractLevelWeapon.fireWeapon_cr` alternates "increment t" and "fire" per tick, so t reaches ≥ 0.11 after 7 ticks and fires on the 8th |
| Bullet speed | 2250 u/s | `Basic.speed` |
| Bullet y offsets (cycle) | 0, 20, 40, 20 | `WeaponPeashot.yPositions` |
| No shooting while dashing | cooldown timer also paused | `fireWeapon_cr` |
| EX | 8.334 per hit, max 25 total, speed 1500, 0.05 s freeze per hit | `LevelWeaponPeashot.Ex` |
| EX cost | 10 meter = 1 card | `PlayerStatsManager.EX_COST` |
| Meter max | 50 = 5 cards | `SUPER_MAX` |
| Meter per damage dealt | 0.0625 per damage point (= 0.25 per bullet; 160 dmg per card) | `OnDealDamage`, `SUPER_ON_DEAL_DAMAGE` |
| Meter per parry | +10 (one card) | `SuperChangedFromParry` |

`spec.PlayerState.super_meter` is in cards: `cards = SuperMeter / 10`.

## Hidden mechanics

* **Co-op damage halving**: `PlayerManager.DamageMultiplier = 0.5` when two players are present;
  meter gain divides it back out. Single-player RL is unaffected (1.0).
* Health-up charms reduce weapon damage (`CharmHealthUpOne/Two.weaponDebuff`) — not in our loadout.
* Super damage hit-pauses the boss for 0.15 s (`DamageReceiver.ENEMY_HIT_PAUSE_TIME`); jumps freeze
  with it (`hitPauseCoefficient`).
* Player cooldown quantisation (8-tick peashooter) — see above; sims using 0.11 s overestimate DPS by 21 %.
* Goopy will not transform until it lands within |x| < 350, so phase 1 can run past 76 %.

## Not statically extractable (measured live by the bridge)

Mecanim clips drive the punch `ArmCollider` box size, the tombstone smash collider and the
question-mark windows; their exact sizes/timings come from runtime sampling of the live colliders
via the bridge (`hazards` in each frame). Animation lengths around `preHold`/`mainHold` likewise.
