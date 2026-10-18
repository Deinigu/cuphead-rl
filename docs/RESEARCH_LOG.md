# Research log

Dated lab notebook. Newest entries at the bottom.

## 2026-10-15 — Kickoff

* Machine: i5-13400F, 32 GB, AMD RX 7800 XT. No CUDA → CPU training.
* Cuphead (Steam app 268910) downloading to `E:\SteamLibrary\steamapps\common\Cuphead`.
* Decision: simulator first, then the real game, connected by a shared symbolic interface
  (`spec.py`), not pixels. Rationale in `docs/PLAN.md`.
* Action space: MultiDiscrete `[move(3), jump(2), dash(2), shoot(2), ex(2)]` → 48 combos.
* Observation: 142-dim vector — player (12), boss (25), 8 nearest hazards (13 each), time (1).
* Reward v1: +1 per 1 % boss HP, −3 per hit taken, +5 per phase cleared, +50 win, −10 loss,
  −0.001 per step, +0.5 per parry.

## 2026-10-15 — Game bridge, phase 1: decompile + constants

* Install: Steam build 20241530, **Unity 2017.4.9f1, x64, Mono** (`Cuphead_Data/Managed/Assembly-CSharp.dll`),
  Rewired input (`Rewired_Core.dll`). Fixed timestep 1/60 s.
* Tooling: no .NET SDK on the machine and `winget` hung on UAC → installed .NET 8 SDK user-locally with
  `dotnet-install.ps1` (`%LOCALAPPDATA%\dotnet-sdk`). Latest `ilspycmd` needs .NET 10 → pinned
  `ilspycmd 9.1.0.7988`. Decompiled to `tools/decompiled/` (gitignored).
* Goopy's code name is **Slime**: `SlimeLevel`, `SlimeLevelSlime` (small + big), `SlimeLevelTombstone`,
  `LevelProperties.Slime`. Scene `scene_level_slime` = build index 20 (`level20`).
* UnityPy reads scene transforms/colliders fine, but MonoBehaviours have no typetrees; TypeTreeGeneratorAPI
  works only with a DLL subset (loading the whole Managed/ folder throws NRE) → `tools/unity_probe.py`.
* Findings worth feeding the sim: Regular HP 1200, phases at 76 % / 31 %; small Goopy only transforms on a
  landing with |x| < 350; big Goopy's first punch after exactly 2 jumps; tombstone smashes at
  player x + cycling offset (−130, 70, 0, 130, −70, 0), and only hurts during the slam.
* **Peashooter fires every 8 ticks (0.133 s, 30 DPS), not every 0.11 s** — the cooldown coroutine spends
  the firing tick instead of incrementing the timer. A sim using 0.11 s would be 21 % too strong.
* Player motor uses a normalised vertical velocity in [−10, 1] × 1620: jump vy 1223, gravity 5992 u/s²,
  full-hold apex 283 u in 0.667 s. Dash 330 u in 0.3 s, no i-frames. Hit i-frames 2 s.
* Not statically extractable: animated collider sizes (punch arm, slam) — to be sampled live via the bridge.

## 2026-10-15 — Game bridge, phase 2: BepInEx plugin + Python env

* BepInEx 5.4.23.2 x64 loads fine on Unity 2017.4 Mono (CLR 2.0). Plugin targets net35.
* Input: patched `Rewired.Player.GetButton*/GetAxis(int)` for player one. All gameplay input is read
  in FixedUpdate, so edges latch per fixed tick.
* Navigation without key spoofing: forcing `Controller.GetAnyButtonDown` on the title screen, then
  calling `SceneLoader.LoadLevel(Levels.Slime)` from the slot screen. Title → fight takes 2.4 s and a
  restart takes 0.6 s (intro at 8× timeScale).
* Lockstep with `Time.captureFramerate = 60` works: 175 ticks/s (2.9× real time) at frame_skip 4,
  limited by render rate. Free-run gives 59.7 ticks/s, with states streamed at about 173 frames/s.
* Live data: the punch stretches the body collider (half-width 80 → 170) and adds the ArmCollider FIST
  hazard. Grounded player y_spec = 0 exactly (ground −250).
* Open: `tools/verify_bridge.py` hung on first run, so jump and dash injection are not verified yet.

## 2026-10-15 — Game bridge: verification, live hitboxes, traces

* Root cause of the verify hang: when the fight ended mid-lockstep, the plugin answered no further
  `act`s. It now replies with an `in_fight:false` state, and every client read has a timeout.
* Live hitboxes (`data/live_hitboxes.json`) settled the tombstone question. The only collider is the
  r=100 circle, spanning y 210–440 while moving, so standing peashots (y 64–104) do 0 damage. The slam
  grows it to r≈260 down to the ground. This led to spec v2's `aim` head. In-game check: aim-up
  shots damage the tombstone and standing ones don't.
* Big Goopy punches with a second root circle thrown about 550 u forward plus a root box. The bridge
  first reported these as BODY; they are now reported as FIST.
* The tombstone state machine flips back to `Move` while the smash clip is still playing, so moves
  are now taken from the clip name.
* All action heads match the decompiled physics (table in GAME_BRIDGE.md), including a successful
  parry on the phase-1 question marks.
* Determinism: shipped fixedDeltaTime 0.016666669 ≠ captured frame 1/60f → lockstep sets it to 1/60f.
  Velocities now use the tick counter, not differences of absolute `Time.fixedTime`.
  `check_env` passes. Long rollouts still diverge after 0.7–11 s (cause unknown), so treat
  episodes as stochastic.
* Throughput: lockstep at time_scale 1 runs at 175 ticks/s (2.9×), 2 at 5.8× and 4 at 11.6×.
  time_scale > 1 coarsens Update and animation steps, so 1 is recommended.

## 2026-10-15 — Simulator v1

**What exists.** `cuphead_rl.sim`: a deterministic, headless 60 Hz re-creation of the Regular
Goopy fight, `GoopyEnv` (gymnasium; obs/reward only via `spec`), a 1930s-style pygame
renderer + MP4 recorder, and two baseline policies.

**Design decisions**
* Fixed 60 Hz step; the env repeats an action for `frame_skip` frames. Jump/dash/EX
  edges fire on the first sub-frame only (spec: "press edge = held now, not held before").
* Player `y` = feet (0 on the ground); boss `y` = hitbox centre (as in `spec.BossInfo`).
* Pure-Python scalar maths (no numpy in the inner loop) → fast and bit-for-bit deterministic.
  Boss randomness (hop-pattern start, tomb pass count) comes from a `random.Random` seeded
  from the gym `np_random`, so `seed + actions` → identical trajectories (tested).
* Every tunable is in `SimConfig` (nested dataclasses). `SimConfig.from_json` takes a
  *partial* nested JSON and rejects unknown keys. That's the drop-in point for the decompiled
  `LevelProperties.Slime` / player motor numbers from the decompiled game.
* `SimConfig.randomized(rng, scale)` jitters every boss/transition float by U(1±scale), with
  hitboxes at ⅔ of the scale. HP, counts and player/weapon stay fixed, since those are
  identical in sim and game.
* Phase change happens at the *start* of the transformation (so `phase_clear` reward fires
  immediately). The boss is invulnerable while transforming; overkill damage is discarded.
  Phase 1→2: grows over 2.5 s and spawns 3 parryable pink "?" (harmless, `qmark_harmful` flag).
  Phase 2→3: big Goopy collapses (1.2 s), then the tombstone falls on him from the sky.
* Tomb AI: accelerates toward the player (1500 u/s², max 620 u/s), counts 2–3 passes, then
  lines up, rises 230 u (0.55 s) + 0.2 s hang, slams, and leaves a 0.18 s crush hitbox
  (±165 u) and 0.7 s of recovery.
* `record_episode` hooks `env.frame_callback`, so **every** simulation frame is rendered
  (smooth 60 fps even at frame_skip 4). It calls `reset(seed, options)` itself, calls
  `overlay_fn()` per frame and returns stats (won, phase, boss_hp_frac, player_hp, time,
  total_reward).
* pygame is only imported on the first `render()` → env is picklable for SubprocVecEnv.

**Numbers and sources** (all placeholders until B's decompile lands)

| what | value | source |
|---|---|---|
| Boss HP (Regular) | 288 / 540 / 372 = 1200 | cuphead.wiki.gg/wiki/Goopy_Le_Grande lists 440/288/336, 560/540/560, 372/504 and totals 1000/1200/1400. The only consistent reading is Simple/Regular/Expert per phase (Simple has no phase 3): 288+540+372 = 1200. |
| Peashooter | 4 dmg, 30 DPS → 7.5 shots/s (every 8 frames) | cuphead.wiki.gg/wiki/Peashooter |
| EX (Mega Blast) | 3 hits × 8.33 = 25 | same page |
| Pea speed 1800 u/s, EX speed 700 u/s | guess ("high velocity" / "slow") | footage |
| Run 490 u/s, jump 1000 u/s, g 4000, hold ≤0.16 s → apex 125 (tap) … 285 (hold) | guess | footage proportions (720-unit screen) |
| Dash 1100 u/s × 0.27 s = 297 u | guess | footage |
| i-frames 1.5 s, parry window 0.2 s, super meter 1 card / 120 dmg | guess | — |
| Small Goopy: 110×104 box, low/high hop apex 139/350, 330 u/s, punch windup 0.55 s, reach 360 | guess | wiki describes behaviour only |
| Big Goopy: 250×236 box, hop apex 174/377, 400 u/s, punch reach 560, first punch +0.6 s (wiki: "blinks a few times before the first punch") | guess | wiki + footage |
| Tomb: 176×160 box, 2–3 passes before a slam (wiki: "usually slides past 2–3 times") | wiki + guess | |

**Benchmark** (i5-13400F, single process, frame_skip=4, no render, random actions incl.
resets): **27.8k env steps/s** (≈111k sim frames/s), well above the 3k target. Rendering +
H.264 encoding runs at ≈75 frames/s.

**Baselines** (`scripts/sim_demo.py`, 50 episodes each, seeds 0–49, full fight):

| policy | win rate | mean phase reached | mean boss HP left | mean episode |
|---|---|---|---|---|
| random | 0 % | 0.00 | 98 % | 9.8 s |
| heuristic | 72 % | 1.86 | 8 % | 120 s |

Per phase (heuristic, 50 eps): starting in phase 2 it wins 34 %, starting in phase 3 it
wins 100 %. Big Goopy's hops and punch are the hard part. The heuristic is a 1 s look-ahead
planner over the spec observation only. Each decision it simulates every plausible
continuation of Goopy's move (both hop heights, the delayed first punch, the tomb chasing
the player) and takes the first manoeuvre that never touches him. It waits until a threat
is within ~0.3 s, because dodging too early (e.g. jumping before the tomb arrives) lands
it in harm's way.

**Caveats / next.** Phase 2 is long for the heuristic (Goopy spends much of his time
airborne above the pea line). Real fights are faster, so once B's numbers arrive, re-check
episode length against the 240 s cap. The heuristic never parries (it only tries when Goopy
is >250 u away). Videos: `videos/sim_random.mp4`, `videos/sim_heuristic.mp4` (gitignored).

## 2026-10-15 — Simulator calibrated to the decompiled game

**Source.** `data/goopy_constants.json` + `docs/GAME_CONSTANTS.md`
(Steam build 20241530, `Assembly-CSharp.dll` + `scene_level_slime`). Mapping y_spec = y_unity + 250.
`SimConfig` defaults are now the game values. `sim/calibration.py` maps every game-backed field to
its JSON key, and `scripts/calibrate_sim.py` prints sim vs game per field and exits 1 on any mismatch
or unexplained unmapped key. It also has `--write-overrides` to use a fresh bridge dump before the
defaults are edited. A test runs it. Current result: **0 mismatches, 39 unmapped keys, all with
a stated reason** (derived values, tiny slimes that never spawn, duck/co-op/revive, etc.).

**Behaviour changes (game facts now modelled)**
* HP is one 1200 pool. Phase 2 starts below 76 %, phase 3 below 31 %, and excess damage carries over.
  Small Goopy keeps hopping until it lands with |x| < 350, then transforms. Big Goopy dies on landing
  (or after its punch). Then comes a 3.5 s death loop and the tombstone drops 800 → 170 in 0.2 s.
* Slimes: game jump patterns (start at a random index; R = random, D.5 = longer ground delay; the
  `HL` typo reads as high), per-jump vy/vx/gravity, ±max_x bounce (vx zeroed if it bounces on the way
  down), 5–8 jumps between punches (big: first after 2), punch only from the punch-position band,
  punch pre-hold 0.65 s and main hold 0.4 s. Circular body hitboxes: r 80 at +75, and big r 160 at +130.
* Tombstone: easeInOutSine legs wall to wall (±500) at 950 u/s on average. After U[2,4] s it picks
  target = player.x + cycling offset (-130, 70, 0, 130, -70, 0), stops once it passes the target and
  smashes. **It only deals damage during the smash.**
* Player: run 490, jump vy 1223.1 held constant for 1–9 ticks, gravity 5991.8, max fall 1620.
  Hurtbox 50×105 standing, 50×50 in the air. Dash 1100 u/s × 0.3 s with no i-frames. 2 s hit i-frames
  plus 0.3 s stun with 300 u/s knockback and vy 972. Parry bounce 1620. 0.134 s input buffer for
  jump/dash/EX (also fixes presses lost on blocked frames).
* Weapons: peashooter on an integer 8-tick cadence (30 DPS; the timer pauses while dashing), bullets
  2250 u/s cycling lanes 0/20/40/20. EX 8.334 per hit up to 25, re-hits every 80 u travelled with a
  0.05 s freeze, speed 1500. Meter 1 card per 160 damage (EX damage counts) and 1 card per parry.

**Calibrated vs estimated**

| calibrated (from the JSON) | estimated (not statically extractable) |
|---|---|
| total HP, phase trigger fractions, intro delay | punch ArmCollider reach/height and extend/retract speeds (animated) |
| player motor: run, jump vy/hold min/max, gravity, max fall, dash, cooldown | animation time around punch pre/main hold (+0.25 / +0.1 s) |
| i-frames, hit stun, knockback, parry bounce, hurtboxes, input buffer | grow (phase 1→2) animation 2.5 s, invulnerability during transforms |
| peashooter dmg/cadence/speed/lanes, EX dmg/speed/re-hit distance/freeze | death animation before the 3.5 s loop (0.6 s) |
| meter per damage/parry, EX meter | **tombstone body hitbox** (see below), smash rise/speed/crush box, recovery |
| slime spawn, bounds, punch bands, ground delays, jump vy/vx/g, patterns, punch counts | question-mark positions, lifetime, harmlessness |
| slime body circles | parry window 0.2 s, EX freeze of the player 0.3 s, muzzle position, bullet boxes |
| tomb rest/drop heights, drop time, patrol bound, speed, attack delay, offsets, anticipation | tomb drop being harmless |

*Tomb hitbox.* The scene circle (r 100 at +170 above the transform at y_unity −80) sits at
y_spec 240–440, out of reach of standing shots, which contradicts the real fight. The sim uses an estimated 200×440 box from the ground until the bridge samples
`Collider2D.bounds` live.
*Jump apex.* Airtimes match the game's tick replay exactly (0.433 s tap, 0.667 s full hold). The
tap apex is 135 (game 133.5) and the full-hold apex is 298 (game 283.3). The tick count is right, so
the remaining 5 % presumably comes from the game's normalised-velocity easing.

**Code-review fixes**: integer peashooter cadence (was 9 frames); one parry per airtime;
no mid-air freeze on phase change; excess damage carried over; game numbers adopted; hazard collision
on raw tuples (Hazard objects only for the observation); renderer config refreshed on reset (randomize);
jump/dash/EX input buffer; seeded `random_policy` default RNG; `GoopyEnv` asserts
`max_seconds <= spec.MAX_EPISODE_SECONDS`.

**Renderer.** Structured agent overlay: `title` becomes a shrink-to-fit caption bar (bottom right),
`value` a gauge clipped to [−10, 60], `probs` per-head bar rows (move as < - >, binary heads as
P(press)). `action` is skipped. The generic key/value fallback stays. i-frame blinking checked on
frames (sprite alternates every 4 frames after a hit).

**Video size.** Film grain and scratches now change every 5 sim frames (12 fps, like film), and the
encoding is libx264 CRF 20, preset medium. Before: `sim_heuristic.mp4` 714 MB for 151 s (37.8 Mbps).
After: 92.7 MB for 97 s (**7.6 Mbps**), random clip 6.7 Mbps. No visible quality loss on extracted frames.

**Baselines after calibration** (`scripts/sim_demo.py`, 50 episodes, seeds 0–49):

| policy | win | mean phase | boss HP left | mean episode (max) | mean time per completed phase |
|---|---|---|---|---|---|
| random | 0 % | 0.00 | 99 % | 9.8 s (19.3 s) | — |
| heuristic | 16 % | 1.28 | 35 % | 63.1 s (98.0 s) | P1 26.6 s, P2 46.8 s, P3 17.8 s |

No episode comes near the 240 s cap: the longest heuristic episode is 98 s, with 0 truncations.
Throughput is 16.6k env steps/s (frame_skip 4, random actions), down from 27.8k because of circle
collisions and the richer state machine, and still 5× the 3k target. The heuristic is weaker than
before (72 % → 16 %): big Goopy's punch (estimated reach 560) and the faster game hops (vy 2350–2850)
cause most hits. That fits the fight being harder and shorter with real numbers.

## 2026-10-15 — Review follow-ups on the calibrated sim

**Performance.** Merged `origin/main` (faster `spec.encode_observation`; `GoopyEnv` now passes
`max_seconds`). In `World.step` the boss body geometry (x, y, half extents, round) is read once per
frame via `Goopy.body()`, reused by the bullet loop, the hazard check and `state()`. Plain peas skip
the EX bookkeeping, the common case (slime body is the only hazard) is an inline circle test, the
player timers are integer frame counters / plain comparisons instead of `max()`, and there is a
fast path for int-array actions (tested identical to `spec.decode_action`). Interleaved A/B on the
same machine, 20k steps, frame_skip 4:

| policy | before (7a9777a) | after |
|---|---|---|
| shoot every step (random move/jump) | 12.5–14.8k steps/s | **20.6–21.7k steps/s** |
| uniform random actions | 14.6–18.9k | 21.6–24.2k |

**Exact durations.** Hit stun and EX freeze are now integer frame counters (18 frames each; float
accumulation made them 19). The input buffer is frames too (8). Tests assert the exact counts.

**Domain randomization.** `randomized()` requires 0 ≤ scale < 0.5. `*_min`/`*_max` pairs share one
factor, and both slime forms share the body-radius factor, so pairs can't invert and big Goopy always
stays bigger (no division by zero in `Goopy.body()`).

**Dying big Goopy deals contact damage — verified in the decompiled code.**
`SlimeLevelSlime.OnCollisionPlayer` calls `damageDealer.DealDamage(hit)` with no state check, and
neither `Die()` nor `DeathTransform()` disables the colliders. They go away only in `Explode()`,
which `SlimeLevelTombstone.crush_slime_cr` calls once the dropping tombstone passes y_unity 70
(y_spec 320). The drop itself is harmless (`SlimeLevelTombstone.OnCollisionPlayer` is gated by
`dealDamage`, set only by the smash animation events). The sim now keeps the BODY hazard through the
whole death and the tomb drop until that point (`TombConfig.crush_slime_y = 320`). Caveat: a Mecanim
curve could still toggle `Collider2D.enabled` during the death clips; that isn't visible statically.
The bridge can confirm it live. Heuristic after this change: 12 % wins over 50 episodes (random 0 %),
longest episode 98 s.

### Add-on — spec v2 aim/duck and live hitboxes (same day)

**Spec v2 (merged `origin/main`).** Action head 6 is the vertical stick (down/none/up). From the
decompiled `LevelPlayerMotor` / `LevelPlayerWeaponManager`:
* Aiming up does **not** stop walking (`HandleWalking` only blocks when look.y < 0 on the ground).
  Up while standing fires straight up, up while running fires diagonally.
* Down on the ground is a duck (`Ducking`): no walking, a 50×35 hurtbox at +25 (live), and
  horizontal shots from (102, 24). In the air the aim is 8-way, including straight down.
* Spawn offsets per pose come from `ProjectilePosition.Get`, e.g. forward (78, 64), running
  (70, 46), up (27, 158), running up (73, 107), jumping up (0, 158). Lanes 0/20/40/20 are spread
  perpendicular to the shot. The EX uses the same aim.

**Live hitboxes (`data/live_hitboxes.json`, bridge 57a7b93) replace the estimates.**
`scripts/extract_live_tracks.py` turns the per-tick timelines into
`sim/live_tracks.py`, and Goopy replays them:
* Small punch: a 64-tick windup (the body leans back to r 120), then the whole body lunges
  390 units forward and grows to r 170 at y 213, with the ArmCollider box (up to 135×130 half)
  in between. A 12-tick blend back to rest is appended (est).
* Big punch: a 118-tick windup, then a second circle (r 80→110) is thrown up to 550 units
  forward at y 183, plus an arm box (up to 165×21 half). Both are hazards; the r 160 body stays put.
* Tombstone: a single r 100 circle at +170 (+145 while sliding), i.e. y_spec 310–340, so
  **standing shots miss**. The smash is a 30-tick anticipation, then the circle swells to r ≈ 250
  and drops through the floor before recovering. It hurts while r ≥ 101 (est. `dealDamage`
  window, ≈ 20 ticks).
* Player hurtboxes: standing 50×105 at +62, airborne 50×50 at **+57** (supersedes the static
  constant of 50), ducking 50×35 at +25. Peashot box 5×5.

The calibration covers both sources. `calibrate_sim.py` checks the static constants, the live
measurements (live wins on overlap) and that `live_tracks.py` is regenerated from the current
JSON. Result: 0 mismatches.

**Heuristic.** It plans against the same tracks with a 1.5 s horizon:
* It never waits out a punch windup.
* It can air-dash later in a jump and can duck. Ducking under big Goopy's thrown circle and arm
  is what made phase 2 winnable.
* In phase 3 it stays under the tomb and aims up.

**Baselines (50 episodes, seeds 0–49):**

| policy | win | mean episode (max) | P1 / P2 / P3 mean time | truncated |
|---|---|---|---|---|
| random | 0 % | 10.3 s (18.0 s) | — | 0 |
| heuristic | **72 %** | 86.9 s (111.6 s) | 27.1 / 40.9 / 31.1 s | 0 |

**Throughput.** 22.3k steps/s in `sim_demo` (random actions). The A/B with the shoot policy
under the same (busy) machine load: 16.2–16.6k before → 20.7–21.5k after. On another run, with
the machine at 91 % CPU from training jobs: 12.1–12.5k → 15.6–16.2k.
Videos: `sim_heuristic.mp4` 93.5 MB (7.6 Mbps), `sim_random.mp4` 9.6 MB (7.2 Mbps).

## 2026-10-15 — Training pipeline

* `cuphead_rl.train`: YAML experiment configs (`configs/*.yaml`, dotted `-o key=value`
  overrides), PPO and RecurrentPPO (sb3-contrib) on `SubprocVecEnv`, optional frame stacking.
* `EvalVideoCallback`: every `eval_every` steps it runs deterministic evaluation episodes,
  saves a checkpoint, keeps `best.zip` (score = 10·win_rate + boss_damage), and **records a
  video when something interesting happens for the first time**: untrained policy,
  reaching phase 2 / phase 3, 25/50/75 % boss damage, first win, 25/50/90 % win rate (plus
  a periodic one). Every video gets a line in `videos/index.jsonl` saying why it exists,
  which is what the YouTube edit is built from.
* Reverse phase curriculum (`AdaptiveCurriculum`): start training on the tombstone only,
  promote to "phase 2 onwards" and then the full fight once the eval win rate from the
  current start phase reaches `promote_at`. Rationale: the +50 win reward is reachable
  from day one instead of being hidden behind 3 minutes of fighting.
* The video overlay shows the agent's value estimate and its action probabilities per head.
* Integration smoke test against the real simulator: 300k steps in 2.3 min including
  evals and a video (~2.2k steps/s end-to-end with 12 envs). Found that the renderer's
  generic overlay can't show the probability dict; added a structured panel to the renderer.

## 2026-10-15 — The tombstone you can't shoot (spec v2)

* Review round 1 of the simulator found 4 major issues: the peashooter fired every
  9 frames instead of 8 (float timer drift → 11 % less DPS), parry re-armable all airtime,
  Goopy frozen mid-air when a phase ended during a hop (a safe pocket underneath him), and
  several constants that disagreed with the decompiled game. All fixed in the calibration
  pass; review round 2 verified each fix with scripts and found only a perf regression and
  one-frame timer overruns.
* Calibration made the fight much harder: the hand-written heuristic dropped from 72 % to
  16 % wins (big Goopy's punch and faster hops).
* **Key finding from live hitbox sampling in the real game:** the phase-3 tombstone is a
  single r=100 circle spanning y≈210–440 above the ground while it moves; standing
  peashots fly at y≈64–104 and deal *zero* damage. The simulator had papered over this with
  an estimated box from the ground up. Human players aim up — our agent had no "up".
* Fix: **spec v2** adds a 6th action head, the vertical stick {down/duck, none, up}
  (`ACTION_NVEC = (3,2,2,2,2,3)`, 144 action combinations). Without it an agent would have
  been structurally unable to beat phase 3 in the real game, no matter how well it trained
  in the sim. Lesson: measure the real thing before trusting your re-creation.
* Pilot PPO run on the uncalibrated v1 sim (5M steps, 10 envs, ~2.1k steps/s with the game
  and two agents sharing the CPU) is used only to sanity-check the pipeline.

## 2026-10-15 — Reviews of the bridge and the training code; experiment protocol

* **Bridge review** (static + recorded traces + fake sockets) found the sim and the game
  disagreeing on the *meaning* of the contract — the most dangerous kind of sim-to-real bug,
  because nothing crashes: the tomb was a hazard all the time in the game but only while
  slamming in the sim; question marks had different hazard kinds; the phase one-hot flipped
  ~2.4 s earlier in the game (HP threshold vs visible transformation); and in the real game
  Goopy **is** vulnerable while transforming (82 hits landed during transforms in a trace).
  Also: a debug invincibility flag could leak across sessions, unbounded lag in free-run mode,
  loadout (charms!) not controlled. Fix: semantics now written into `spec.py` docstrings,
  and both backends were aligned to them.
* **Training review** found nothing that invalidates results but several things that would
  have quietly skewed them: unscaled rewards (+150 returns) let the value loss dominate the
  shared grad-norm clip (policy steps ~4× smaller) → train with `reward_scale=0.1`;
  fine-tuning ignored the new config; `best.zip` chosen on the same 20 eval seeds (selection
  bias) → every run now ends with a 100-episode held-out eval (`final_eval.json`); domain-rand
  evals ran on randomized physics → all evals on nominal physics.
* **Throughput** (PPO, MLP 256×256, sim v3): 8 envs 4.3k, 12 envs 5.6k, 16 envs 6.2k steps/s
  (4 torch threads; 8 threads is slower). Two runs in parallel with 8 envs/2 threads each.
* **Protocol.** Comparative experiments run on sim v3 (calibrated, spec v2, before the
  transform-vulnerability change); all compared runs use the same sim version. The final
  transfer model is trained afterwards on the final sim with the winning recipe.
  Headline metric: full-fight win rate on nominal physics, 20 fixed eval seeds during
  training, 100 held-out seeds at the end. Pilot on sim v1 (5M steps): 0 % wins but reached
  phase 3 at 4M steps with up to 80 % boss damage → the pipeline learns.

## 2026-10-17 — First comparison runs, a misleading video, and why plain PPO plateaus

* Session restart killed both runs at 6.5M steps (one `final.zip` half-written). Added
  `train --resume RUN_DIR`: newest *readable* checkpoint, eval timeline/milestones/curriculum
  stage restored from `metrics.jsonl`. Both runs continued on the same timeline.
* **The "first_win" video showed a loss.** Videos replayed eval seed 10000, not the episode
  that earned the milestone. Fix: evaluation reports `win_seed`/`best_seed` and the video
  replays that episode; verified the replay reproduces the win exactly (deterministic policy,
  seeded sim). `scripts/rerender_milestones.py` re-records the older videos.
* Results on sim v3 (8 envs each, two runs in parallel):
  | run | 3.5M | 6.5M | 10M |
  |---|---|---|---|
  | plain PPO (full fight) | 0 % win, 25 % dmg | 0 %, 41 % | 0 %, 34 % — plateau, stopped at 10M |
  | reverse curriculum | **first win** (5 %) | 5 %, 29 % | 5 %, 35 % (59 % train-time wins, mostly phase-2 starts) |
* **Diagnosis of the plateau** (30 episodes of the 10M baseline policy, `hits.py`): the agent
  stands still 87 % of the time and shoots; 90 % of the HP it loses is Goopy's *body* landing
  on it during hops (small 56, big 23), punches and the tomb barely matter. Optimisation
  diagnostics are healthy (explained variance 0.84–0.98, KL ≈ 0.006), so this is a reward
  problem: a hit (−3) costs the same as 3 % boss damage, so trading hits for damage is
  "rational" — but with 3 HP survival is what wins fights.
* Next: reward ablation with hit penalty −8 (`configs/ppo_curriculum_hit8.yaml`), and the
  final experiments on the corrected simulator (a review found Goopy's punch-position
  rule mirrored vs the game since calibration — fixed next).

## 2026-10-17 — Contract alignment and sim-vs-game gap report

`scripts/compare_traces.py` replays the bridge's recorded game actions
(`data/traces/*.jsonl`) in the sim over 30 seeds and
compares hop/punch/transform/tomb timings, phase durations and the HP curve
(`docs/SIM_VS_GAME.md`, `docs/sim_vs_game_hp.png`).

**Findings and fixes** (rules from the decompiled code, anim lengths from the trace):
* Goopy takes damage while transforming (196 HP lost small→big, 120 HP while dying on
  the trace). No cap: every hit goes through `DealDamage` on one HP pool.
* The sim missed the landing squish anims (ground stay 0.22 → 0.57 s small, 0.3 → 0.6 s big),
  the Punch_End anim (now IDLE, as `StartJump` runs at its start), the first-punch-only blink
  windup of big Goopy, the 23-tick vanished turn of small Goopy's turn-punch, and the
  1 s intro delay that never gates jumps.
* Big Goopy dies immediately if hit below 31 % on the ground, even mid-punch.
* The tombstone only decides to smash between legs and pauses before every leg (wall-to-wall
  1.07 → 1.38 s; game 1.4 s); it is invulnerable 22 ticks underground in its smash.
* Spec semantics (d435b30) each have a test: hazards damaging-or-parryable, TOMB only while
  the slam hurts, ? marks OTHER + parryable, phase = visible entity, time from episode start,
  BODY while dying.

Before → after, stand_shoot: hop period 0.86 → 1.21 s (game 1.19), punch→jump 1.23 → 1.77 s
(1.77, big), small→big 2.5 → 6.5 s (6.5), final HP 361 → 240 (240). All stats with more than one
game sample are now within 9 %. The largest remaining gaps are tomb smash timings with
n = 1 (game values at the sim's 5th/99th percentile). Heuristic: 72 % wins before and after,
mean episode 86.9 → 57.6 s.

## 2026-10-17 — Review fixes, re-recorded traces

* The punch-position rule was **mirrored** (wrong since 0dcdb46). The game punches right
  when punch_min_x < x < punch_max_x and left inside (−max, −min). Both small punches in the
  re-recorded traces were impossible in the sim before. The punch is now decided once, on
  the landing frame, as `jump_cr` does.
* Big turn-punches add 11 windup ticks (trace: blink start 75 vs 64 ticks). Later big punches
  skip 57 blink ticks, so the non-turn Punch_Start is an estimated 7 ticks. The random
  pattern start is drawn over raw tokens. Trace rows are compared after the step (fixes a
  1-tick offset). Also: state export after a hit cut a punch short, the grow timeline under
  randomization, and the heuristic's windup variants.
* The gap report on the new traces (e74b501): every multi-sample stat is within 3 % on
  stand_shoot. The largest gaps are single samples: phase 1 length 18.7 vs 21.9 ± 2.0 s and
  tomb smash period 4.12 vs 4.78 ± 0.87 s.
* Heuristic: **92 %** wins over 50 episodes (72 % before). With the mirrored rule, Goopy
  punched from positions the game never punches from.

## 2026-10-17 — Round 2 on the final simulator: 100 % wins

Final sim (game punch rule, landing-time punch decision, transform vulnerability, phase
timing; every multi-sample stat within 3 % of the game traces). Reverse curriculum, 8 envs,
15M steps, hit penalty −3 vs −8 (`configs/r2_*.yaml`).

| run | first win | eval win ≥ 90 % | held-out (100 fights, unseen seeds) | HP lost / fight | mean win time |
|---|---|---|---|---|---|
| curriculum, hit −3 | 4.0M | 12.0M | **100 %** (final.zip) | 0.17 | 45.5 s |
| curriculum, hit −8 | 6.5M | 9.5M | **100 %** (final.zip) | 0.31 | 47.6 s |

* Same recipe that plateaued at ~5–10 % on sim v3 now reaches 100 %: with the mirrored punch
  rule fixed, Goopy's punches follow the real (learnable) rule. Round 3 runs plain PPO on the
  final sim to separate "curriculum" from "simulator fix".
* The hit −8 penalty made learning *slower* to the first win (6.5M vs 4M) and gave no final
  benefit — the plateau in round 1 was not (only) a reward problem.
* The agent learned the spec-v2 trick: in phase 3 it stands next to the tombstone, aims **up**
  (P(up) ≈ 1) and shoots, dashing out of the slam.

## 2026-10-17 — First contact with the real game

* `scripts/play_game.py` (real-game eval in the training-eval format, optional recording
  + per-decision log) and `scripts/overlay_game_video.py` (the sim renderer's "agent thoughts"
  panel composited on the captured game video). Capture switched to ddagrab (gdigrab froze on
  fullscreen).
* **Zero-shot teaser with the round-1 model** (sim v3, 5 % sim win rate), as Mugman: lockstep
  **won 1/5 real fights** (63 s, 1 hit taken), mean boss damage 54 %, phase III reached; free-run
  (real-time, needed for recording) 0/15, 54 % damage — decision latency (~13 vs 15 decisions
  per game second) seems to hurt. Next: the 100 % model, evaluated in lockstep and recorded in a
  real-time-paced lockstep mode.
* Saves: before/after hashes of slot 1/2 identical in every session; only the bridge slot changes.

## 2026-10-18 — Round 2 in the real game, and three sim-to-real bugs

* **r2_curriculum_hit3 zero-shot, Mugman.** Lockstep: 33/40 wins (83 %), 1.70 hits per
  fight. Paced (lockstep slowed to real time, for recording): 9/10, 0.80 hits. Free-run: 0/10
  (uncapped render rate, variable frame time, late actions; not a valid evaluation). In the sim:
  0.14 hits per fight. Most of the 10x gap was in phase I, mainly small Goopy's punch lunge.
* **Plugin fix: velocities in lockstep.** The plugin computed velocities between reads, i.e. a 4-tick
  average in lockstep; the sim reports per-tick values. With per-tick sampling, the *same* model
  won **19/20** real fights at 0.80 hits per fight (hits during phase-I hops: 8 → 0).
* **Sim fix: the punch had no velocity.** While a recorded hitbox track played (punch windup and
  lunge, tomb smash), the sim reported the body as standing still. The policy ducks under the
  lunge using exactly that velocity (perturbing it drops P(duck) from 1.00 to 0.28). Fixed.
* **Sim fix: three more mechanics.** The held jump applies one tick of gravity (full-hold apex was
  15 u too high); the tomb's damage window is ticks 40–57; EX locks the player for 16 + 18 ticks
  with a slide back (was a 0.3 s freeze). An EX fired during a slam was a hit in 4/4 game cases
  and 0/4 sim cases. After these fixes the r2 model scores 95 % / 0.81 hits in the sim against
  95 % / 0.80 in the game: **the simulator now predicts real-game performance.**
* Lesson: compare the two worlds per *cause*, not just per outcome. Each fix came from a
  hit-cause table (`scripts/game_hits.py`) that was identical in both backends.

## 2026-10-18 — Round 3: plain PPO control, domain randomization

| run (round-2 sim) | first win | eval ≥ 90 % | held-out wins | hits / fight | win time |
|---|---|---|---|---|---|
| plain PPO, 15M | 4.5M | 9.0M | 94 % (best.zip 96 %) | 1.0 | 60.1 s |
| curriculum hit −3 (round 2) | 4.0M | 12.0M | 100 % | 0.17 | 45.5 s |
| r2 + 10M on ±15 % physics | – | – | 100 % | 0.23 | 43.5 s |

* Plain PPO learns *faster* early (80 % at 6M vs 10 %) but plateaus around 90 % with 6x the hits.
  The round-1 plateau (0 % at 10M) was the simulator's mirrored punch rule, not a missing
  curriculum. The curriculum buys final quality, not speed.
* `train --resume` of an `init_from` run counted the budget from step 0 and stopped at once;
  `evaluate` always forced nominal physics, so `-o randomize=true` was silently ignored. Both
  fixed with tests. Robustness of r2: 100 % at ±15 %, 95 % at ±25 % physics noise.
* r3_domain_rand in the real game (after the plugin fix): 19/20, 0.85 hits, 4 s faster than r2. The
  tombstone slam became the main real-game cause (0.50 / fight), which the mechanics fixes then explained.

## 2026-10-18 — Round 4 on the corrected sim: the final agent

* On the corrected sim, the r2 / r3 models score 97 % / 98 % wins, 0.56 / 0.64 hits.
  `r4_finetune` = r3_domain_rand + 10M steps on the corrected sim with ±15 % randomization:
  **100 % held-out, 0.39 hits, 45.2 s**; 99 % at ±15 % and 94 % at ±25 % physics noise.
* **Real game, Mugman, no real-game training at all: 20/20 lockstep wins at 0.40 hits per
  fight (sim: 0.39), 12 no-hit wins; 10/10 recorded paced wins, 5 without a hit.** Remaining hits:
  tomb slam 6, phase-I punch 1, big fist 1 (20 lockstep fights).
* Video: a no-hit win in
  45.5 s with the agent's thoughts overlaid.
* `r4_scratch` (curriculum + randomization from zero on the final sim, for the learning footage)
  reached 90 % evaluation wins at 6.5M steps, faster than any earlier run (r2: 12M).

| model | sim held-out | real lockstep | real hits / fight |
|---|---|---|---|
| round 1 (ppo_curriculum) | ~5 % | 1/5 | 2.6 |
| round 2 (after the plugin fix) | 100 % | 19/20 | 0.80 |
| round 3 (domain randomization) | 100 % | 19/20 | 0.85 |
| **round 4 (corrected sim)** | **100 %** | **20/20** | **0.40** |

## 2026-10-18 — Wrap-up: r4_scratch, videos

* `r4_scratch` finished 20M steps: first win at 3.0M, 90 % evaluation wins at 6.5M, held-out
  99 % (final.zip), 0.54 hits, 47.6 s. The final model stays `r4_finetune` (100 %, 0.39 hits).
* Curated clips and what each shows: `docs/VIDEOS.md`. Figure: `docs/figures/round4.png`.
