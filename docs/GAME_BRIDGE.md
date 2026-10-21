# Real-game bridge (CupheadRLBridge)

A BepInEx 5 plugin (`mod/CupheadRLBridge`) that reads the Goopy Le Grande fight into
`spec.GameState`, injects agent actions into Cuphead's input layer, and controls time.
Python side: `cuphead_rl.game.client` (protocol), `cuphead_rl.game.env.CupheadGameEnv` (Gymnasium),
`cuphead_rl.game.capture` (ffmpeg window recorder: ddagrab, or gdigrab), `cuphead_rl.game.smoke` (end-to-end check).

## Install
1. `powershell -File tools/install_bepinex.ps1` adds BepInEx 5.4.23.2 x64 (SHA-256 checked) to
   `E:\SteamLibrary\steamapps\common\Cuphead`. It **adds files only**, so no original game file
   is overwritten: `winhttp.dll`, `doorstop_config.ini`, `.doorstop_version`, `changelog.txt` and
   `BepInEx\`. The list is in `BepInEx\.installed_by_cuphead_rl.txt`.
2. Build the plugin with the .NET 8 SDK (installed user-locally at `%LOCALAPPDATA%\dotnet-sdk`) with
   the game **closed**, because the DLL is locked while it runs:
   `dotnet build mod/CupheadRLBridge -c Release`. This copies `CupheadRLBridge.dll` to `BepInEx\plugins\CupheadRLBridge\`.

## Uninstall
`powershell -File tools/uninstall_bepinex.ps1` removes everything listed above. Optionally verify
the game files in Steam afterwards.

## Launch into the fight
Run `start steam://rungameid/268910`, then connect a client and send `reset`. No OS key
spoofing is used:
1. On the title screen the plugin forces `Rewired.Controller.GetAnyButtonDown`, so player one joins.
2. It waits for the slot screen to load save data (`PlayerData.Initialized`, state `MainMenu`).
3. It calls `SceneLoader.LoadLevel(Levels.Slime)` on Regular in save slot 2 (fixed, see
   [Save safety](#save-safety)), with player one as Mugman unless a client asked for Cuphead.

The level intro runs at `IntroTimeScale` = 8×. Measured: title → fight in 2.4 s; restart inside
the fight in 0.6 s. Winning marks Goopy as beaten in that save slot. Every fight start forces the
contract loadout of a fresh save (peashooter, no secondary, no charm, no super); `hello` and
`reset_done` report it as `loadout` and `CupheadGameEnv` refuses any other.

## Protocol (127.0.0.1:47800, newline-delimited JSON)
Client → plugin (`{"cmd": ...}`):

| cmd | fields | reply |
|---|---|---|
| `hello` | | `hello` (version, spec_version, scene, no_hit, loadout, character, save_slot, save_guard, saves_blocked); the client raises unless spec_version == `spec.SPEC_VERSION` |
| `config` | `stream_every` (frames, 0 = off), `lockstep`, `frames` (ticks/action), `time_scale` (0–20), `control`, `debug` | `config` |
| `act` | `move` −1/0/1, `jump` (held), `dash` (one-tick press), `shoot` (held), `ex` (one-tick press), `move_y` −1/0/1 (= spec v2 `aim`: −1 duck, +1 aim up / diagonal while moving), `frames`, `seq` (echoed in later states) | lockstep: a `state` after `frames` ticks |
| `reset` / `goto` | `seed` (optional; seeds `UnityEngine.Random` at `SlimeLevel.Start`), `character` (optional, `mugman`/`cuphead`, kept for later loads; default Mugman) | `reset_done` (incl. `character`), then a `state` |
| `state` | | `state` |
| `damage` | `amount` (debug/curriculum: `LevelProperties.DealDamage`) | `state` |
| `set_meter` | `cards` (debug) | `state` |
| `invincible` | `on` (debug: player ignores every hit, including knockback; cleared on every connect, disconnect and reset, reported as `no_hit`) | `state` |
| `release` | | none |

Plugin → client: `state` messages are already in spec coordinates (x unchanged, y − Level.Ground (−250)).
They carry the `spec.GameState` fields, with enum ints matching `spec.BossPhase/BossMove/HazardKind`.
Extras: `tick`, `frame`, `odd_frames`, `no_hit`, `character` (`PlayerManager.player1IsMugman`), `seq`/`act_tick` (last applied action and its tick), `boss.entity` (small/big/tomb), `boss.anim` (current clip),
hazard `name`, player `hb_*` (hurtbox), `ducking`, `raw_y`. With `debug`, the state also carries
`bullets` and every boss collider. If the fight ends while a lockstep step is in flight, the reply
is a `state` with `in_fight: false`. One client at a time: a second connection gets an `error`
line and is closed (the client raises `BridgeError`).

Hazards follow the `spec.HazardKind` contract (only what can hurt this tick, plus parryables).
BODY = the entity's body circle, identified once per entity (the root circle with the largest
radius), in every state. FIST = small Goopy's ArmCollider; for big Goopy, the root box plus the
r=80 fist circle once it leaves the body circle (resting inside the body it adds no damage area and
is omitted, as in the sim). TOMB = the tombstone circle, only while `dealDamage` is set (the slam);
the patrolling tombstone is visible through `boss`. OTHER (parryable) = phase-1 question marks.
Velocities are finite differences per collider; a gap of more than 8 ticks or a speed above
6000 u/s (teleport, entity swap) reports 0.

Boss phase follows `spec.BossPhase`: the visible entity, not `CurrentState` (which flips at the HP
threshold). BIG from the moment small Goopy starts its transform animation, TOMB once the tombstone
leaves `Init`. `vulnerable` is reported from the DamageReceiver, so it stays true while transforming.

Boss moves: the tombstone's state machine says `Move` while the smash clip is still playing, so the
bridge reports TOMB_SLAM_WINDUP/TOMB_SLAM from the clip name (`smash_start`/`smash_pre_hold` →
WINDUP, `smash` → SLAM).

## Modes and throughput
* **Free-run**: real time × `time_scale`. States are streamed every `stream_every` rendered frames.
  Measured: 174 states/s (rendered frames) at 60.0 game ticks/s. `step_free` discards the backlog,
  sends the action with a `seq`, and returns the first state with that `seq` at least `frames` ticks
  after `act_tick`. Streamed states are dropped (never blocking the game) when the socket's send
  buffer is full; other writes time out after 2 s and drop the client.
* **Lockstep**: sets `Time.captureFramerate = 60` and `Time.fixedDeltaTime = 1/60f` (the shipped
  0.016666669 caused 0/2-tick frames), so one rendered frame = one fixed tick. After `frames` ticks
  the plugin sends the state and blocks the main thread until the next `act`.

Lockstep throughput (`tools/verify_bridge.py`; RX 7800 XT, 1280×720, render-bound at about 175 frames/s):

| time_scale | frame_skip | steps/s | ticks/s | × real time | ticks/step |
|---|---|---|---|---|---|
| 1 | 2 | 87.2 | 174 | 2.91 | 2 |
| 1 | 4 | 43.7 | 175 | 2.91 | 4 |
| 2 | 2 | 174.5 | 349 | 5.82 | 2 |
| 2 | 4 | 87.2 | 349 | 5.81 | 4 |
| 4 | 2 | 174.2 | 697 | 11.61 | **4** (one frame = 4 ticks, so frame_skip is rounded up) |
| 4 | 4 | 174.2 | 697 | 11.61 | 4 |

`time_scale > 1` gives Update/animation steps of `time_scale/60` s (several fixed ticks per rendered
frame). Coroutines and animator waits then resolve on coarser frames. Same seed and same actions at
time_scale 4 versus 1 diverged within a few seconds (39 % agreement on boss moves). That is partly
the same chaos as below, but timing at time_scale > 1 is not tick-identical to normal play.
**Best safe setting: time_scale 1 (2.9× real time).** Use time_scale 2 (5.8×) only if the
sim-to-real gap you measure at 2 is acceptable; frame_skip should be a multiple of time_scale.

## Verification (`tools/verify_bridge.py`, results in `data/verify_bridge.json`)

| check | measured | expected (decompiled) | result |
|---|---|---|---|
| jump full-hold apex | 283.3 | 283.3 | PASS |
| jump full-hold airtime | 0.633 s | 0.667 s (the replay counts the landing tick) | PASS |
| jump tap apex | 133.5 | 133.5 | PASS |
| run speed | 490 | 490 | PASS |
| ground dash | 312 u, 18 ticks | 330 u, 0.3 s (one tick lost to the dash start) | PASS |
| air dash | 1 per airtime, vy = 0 while dashing | same | PASS |
| peashooter | every 8 ticks, 4 damage/hit | 8 ticks, 4 | PASS |
| EX | fires with ≥ 1 card, costs 1 card, blocked at 0 | same | PASS |
| duck (aim = −1) | hurtbox half_h 17.5 (standing 52.5) | 17.5 | PASS |
| parry | 1 parry on a phase-1 question mark | ≥ 1 | PASS |
| phase 3: standing shots | 0 damage | 0 (tomb circle spans y 210–440) | PASS |
| phase 3: aim-up shots | 40–60 damage per 5 s | > 0 | PASS |
| determinism (seed, fs=4) | identical for the full 151-step episode in one run, 1.95 s in another | identical | flaky |
| determinism (seed, fs=1) | identical for 0.7–11 s, then diverges | identical | FAIL |

**Determinism.** A seeded lockstep reset is frame-identical: `gymnasium.utils.env_checker.check_env`
passes, including its reset/step determinism checks. Long rollouts, however, eventually show a
1-unit difference (once at 0.7 s, once at 5.5 s, once at 11 s, once never in 10 s). After that the
fight diverges chaotically (different hop choices). Gameplay code reads no wall-clock time, and
`odd_frames` stays 0. The cause is probably Unity-internal ordering across scene reloads (e.g.
Physics2D callbacks); it has not been pinned down. Treat episodes as stochastic.

## Live hitboxes and traces
* `tools/sample_hitboxes.py` → `data/live_hitboxes.json`: collider bounds per animation clip for
  each boss form, the punch and slam timelines, the player hurtbox (standing / duck / air) and
  peashooter bullet heights (standing y 64–104).
* `tools/record_trace.py` → `data/traces/stand_shoot.jsonl` (60 s, no-hit, phases 1 → 3; BIG from tick 1124, TOMB from 2731) and
  `dodge_shoot.jsonl` (natural damage, dies at 10.9 s). Each line has the tick, the action and the
  compact state; line 1 is metadata.

## Save safety
`%APPDATA%\Cuphead\cuphead_player_data_v1_slot_{0,1}.sav` are your own saves and must never
be written; the bridge uses slot 2 only. `mod/CupheadRLBridge/SaveGuard.cs` enforces this while
the plugin is loaded, whatever the game or a client does:
* `OnlineInterfaceSteam.SaveCloudData` (the only code that writes `.sav` files; `PlayerData.Save`,
  `SaveAll` and the migration/no-data paths all end there) drops every
  `cuphead_player_data_v1_slot_N` key except slot 2 (settings/achievement keys pass). If nothing is
  left it reports success to the callback, so the game does not retry forever.
* `PlayerData.Save(i)` and `PlayerData.ClearSlot(i)` are no-ops for i ≠ 2.
* The `CurrentSaveFileIndex` setter is pinned to 2, and the plugin sets it at load (before the
  title or slot screen could call `SaveCurrentFile`), again when entering the fight and on every reset.
* `SaveSlot` in the config is fixed to 2; any other value logs an error and is ignored.
* **Fail closed.** The 4 guard classes are patched first, one by one (not through `PatchAll`, whose
  order is undefined), and each one is verified by name through Harmony. If any is missing, the plugin
  unpatches itself and re-applies only the SaveCloudData guard. That guard then drops *every*
  player-data key, slot 2 included, so while the plugin is loaded nothing is saved at all. The
  plugin logs `NOT started: save guards incomplete` and creates no bridge, no server and no auto-goto.
  This matters because the riskiest vanilla write happens at the slot screen, before any client
  connects: `PlayerData.OnLoaded` → NoData/migration → `SaveAll`.
* `hello` reports `save_slot`, the patched methods (`save_guard`, read live from Harmony at
  hello time) and `saves_blocked`. `BridgeClient.connect` raises unless all 4 guards are patched and
  the slot is 2. Every tool goes through it (`CupheadGameEnv`, `verify_bridge.py`,
  `sample_hitboxes.py`, `record_trace.py`, `save_safety_session.py`).
* Blocked writes are logged as `SaveGuard: blocked ...` in `BepInEx\LogOutput.log`.
* Testing the fail-closed path: set `[Debug] DebugSkipGuard = ClearSlotGuard` in
  `cuphead_rl.bridge.cfg`. The plugin leaves that guard out and must refuse to start. Clear the
  setting afterwards.

Rules for people:
* **Never delete `cuphead_player_data_v1_slot_2.sav`** to reset the bridge slot. If a slot file is
  missing, the game takes its NoData path and calls `SaveAll()`, which in vanilla rewrites *all three*
  slots with defaults. The guard would drop slots 0/1, but do not rely on it, and without the
  plugin it would wipe your saves.
* With the plugin loaded, every slot is slot 2: picking the first or second slot on the slot screen (files slot_0/slot_1) by
  hand silently plays slot 2's data, and nothing is saved to the slot you picked. Remove or
  disable the plugin (`tools/uninstall_bepinex.ps1`) to play your own saves.
* Steam Auto-Cloud syncs `%APPDATA%\Cuphead` (`steam_autocloud.vdf`). If a slot file is ever
  restored from a backup by hand, Steam may show a sync-conflict prompt at the next launch:
  pick the local files, and check the hashes with `tools/check_saves.py`.

`tools/check_saves.py` prints sha256/size/mtime of the three slots (read-only). `--out` writes a
snapshot and `--assert-unchanged` compares slots 0 and 1 against one.

Evidence (2026-10-17). The snapshot was taken with the game closed. Then two
full sessions, each in a fresh game process: run 1 was `verify_bridge.py` (37/40, only the known
determinism checks fail) followed by `save_safety_session.py`. Run 2 repeated `save_safety_session.py`
on the final plugin. Each `save_safety_session.py` did 3 resets, a natural death, and a no-hit win
that reached `scene_win`, which saves the level result. After each run the game was closed
normally (WM_CLOSE):

| slot | sha256 before | after runs 1, 2 and a final hello check | mtime |
|---|---|---|---|
| 0 (main save) | recorded | identical | unchanged |
| 1 | recorded | identical | unchanged |
| 2 (bridge) | recorded | changed (the wins were saved) | changed |

Fail-closed evidence (2026-10-17). Test 1 was a dry run with `DebugSkipGuard = ClearSlotGuard`
and `AutoGotoSlime = true`. The log showed `SaveGuard: MISSING PlayerData.ClearSlot ...`,
`fail-closed state, patched: [OnlineInterfaceSteam.SaveCloudData]` and `NOT started: save guards
incomplete`. Nothing listened on port 47800, the fight never loaded, and after a normal close all
three slots were byte-identical. Test 2 was a normal run with the final plugin, done after the
config was restored: `verify_bridge.py` (38/40, only the known chaos checks) and
`save_safety_session.py`. Slots 0 and 1 were unchanged (sha256 and
mtime), while slot 2 changed (wins).

`saves_blocked` stayed 0: with the index pinned, the game never even tried to write another slot.
The guards are a second line of defence, confirmed installed via `save_guard`. They were not
exercised by forcing a slot-0 write, because that would risk the real save.

## Character (Mugman)
The agent plays Mugman by default (`CupheadGameEnv(character="mugman")`, or `"cuphead"`). Before the
level loads, the plugin sets `PlayerManager.player1IsMugman` and slot 2's `isPlayer1Mugman`, as
`SlotSelectScreen.EnterGame` does. In the decompiled code, every Mugman branch is presentation
only: sprites, animations, intro and death effects, HUD cards, the game-over portrait and the
save-slot UI. `LevelPlayerMotor`, the collider manager, `PlayerStatsManager`,
`LevelPlayerWeaponManager` and the weapons never test the character, and both characters use the
same player prefab. Hurtbox, movement and weapons are therefore identical, and the sim and the spec
contract do not depend on the character. Confirmed live: `verify_bridge.py` ran as Mugman with the
same results as with Cuphead (jump, dash, run, duck hurtbox, peashooter, EX, parry), and
`videos/game_mugman.mp4` shows Mugman in the fight.

## Gymnasium env
`CupheadGameEnv(frame_skip=4, reward_config=..., max_seconds=spec.MAX_EPISODE_SECONDS, lockstep=True,
time_scale=1.0, no_hit=False, character="mugman", host, port, connect_wait)`; unknown kwargs raise `TypeError`. There is
one game, so only one env may exist per process (use `n_envs=1` for the game backend):
* `reset(seed=None, options={"start_phase": k})`: k = 1/2 deals 290 / 290 + 545 damage through the
  bridge, then steps (player shielded) until the boss has transformed. This takes 3.0 s / 4.5 s.
  Scripted damage is excluded from `info["damage_dealt"]`; `time` counts from the end of `reset`
  (spec: seconds since the episode started), also with `start_phase`. Episodes truncate at
  `max_seconds`, which is also the observation's time scale.
* Debug no-hit is sent off on connect unless `no_hit=True`. If in-game time does not advance for
  50 steps (game paused, controller prompt), `step` raises.
* `info` keys: `won, phase, boss_hp_frac, player_hp, damage_dealt, time, parries`. The raw message is `env.last_msg`.
* The reset seed comes from the env's `np_random`, so `reset(seed=s)` reproduces the start state.

## Playing trained agents (real-game evaluation)
Before every session, with the game **closed**, take a save snapshot:
`python tools/check_saves.py --out data/save_safety/before.json`. Then launch the game and run:

    python -m uv run python scripts/play_game.py runs/<name>/best.zip --episodes 20            # evaluation
    python -m uv run python scripts/play_game.py runs/<name>/best.zip --episodes 5 --record videos/<dir>
    python -m uv run python scripts/overlay_game_video.py videos/<dir>/episode_02.mp4 -o videos/<dir>/teaser.mp4

Close the game and run `python tools/check_saves.py --assert-unchanged data/save_safety/before.json`.

* `play_game.py` loads the checkpoint with `load_model` + `AgentPolicy` (deterministic unless
  `--stochastic`). `frame_skip`, `frame_stack`, the reward and `env_kwargs.max_seconds` come from the
  run's `config.yaml`. The env is `CupheadGameEnv` with time_scale 1, no-hit off and Mugman
  (`--character`), and `--start-phase k` is optional. Stats use the same keys as
  `cuphead_rl.train.evaluate.summarize`, plus `per_episode` (with `wall_seconds`),
  `mean_survival_time` and `settings`.
* **Modes** (`--mode`):
  * `lockstep` (default): the evaluation setting, about 2.9× real time. Measured: 13 s wall per
    episode on an idle PC, and 17.5 s with two training runs sharing the CPU, for a ~50 s fight, plus
    about 0.6 s per reset. 20 episodes take 5–6 minutes.
  * `paced` (default with `--record`): `PacedGameEnv`, lockstep paced to real time. Each decision's
    frame_skip ticks go out as single-tick `act`s, at most one per 1/60 s of wall time, so the capture
    plays at normal speed with every game frame on screen (a 47 s fight takes ~50 s wall). Dash and
    EX, which are one-tick presses, go on the first tick only; held buttons are held for all ticks.
    The game is still in lockstep, so CPU load from other processes (training, the encoder) only
    slows the wall clock and does not change what the agent sees. Before the per-tick velocity fix, paced and lockstep
    observations differed in the velocities (see "Velocities" below). Since then the two modes see
    the same observations.
* **Velocities.** All reported velocities (player, boss, hazards) are 1-tick finite
  differences, as in the sim. `StateReader.Vel` differences positions between samples, and the
  bridge now samples (`StateReader.Track`) on every fight frame that sends no state. In lockstep a
  frame is one tick, so a state sent after `frames` = 4 ticks carries the motion of the last tick
  rather than the 4-tick average. In free-run, frames without a tick keep the last velocity. The
  protocol is unchanged. Boss `state_time` is also tracked every tick now, not only at reads.
  Before this fix, lockstep with frame_skip 4 reported 4-tick averages, which lag accelerating motion by
  about 1.5 ticks. Paced mode reads every tick, so it already had 1-tick velocities. With the same
  seed, the policy acted differently at about 0.7 s (Goopy's first hop) and the episodes diverged.
  `tools/check_velocities.py` checks this. It plays the same tick-aligned script (idle, hold
  jump 20 ticks, Goopy's first hop) with frames=1 and with frames=4 from one seeded reset, and
  compares velocities at the matching ticks. Results are in `data/check_velocities_{before,after}.json`:

  | | player vy, max \|f4 − f1\| | boss vy, max \|f4 − f1\| | example: tick 44 player vy f1 / f4 / sim | tick 44 boss vy f1 / f4 |
  |---|---|---|---|---|
  | before (4-tick average) | 775 u/s (landing) | 1456 u/s | 923.5 / 1048.3 / 923.5 | 1891.7 / 472.9 |
  | after (per tick) | **0.0** | **0.0** | 923.5 / 923.5 / 923.5 | 2241.7 / 2241.7 |

  On the falling part of the jump, the game's 1-tick vy matches the sim's exactly (923.5, 524.1,
  124.6, −274.8, …). One spec-level difference remains, which this fix does not change: while jump is
  held, the game moves 18.72 u per tick (1123.2 u/s), but the sim reports `vy` = 1223.1 (its
  `jump_speed`). The game's motor applies one tick of gravity (6000/60 = 100 u/s) before it moves
  the player, so the finite difference is 100 u/s lower for the ~9 hold ticks. The positions agree
  (apex 283.3 in both).
  * `free`: the free-running game in real time (about 35 s wall for a 32 s fight). This is **not**
    the setting the sim was calibrated on (see "Why free-run differs").
* `--record DIR` writes `episode_XX.mp4` (60 fps, native resolution, h264_amf by default; `--encoder`
  selects another one; game audio from WASAPI loopback in a separate process, `--no-audio` to skip it;
  the intro plays at real speed, `--fast-intro` to skip it, and the knockout or death card and results
  screen are kept) and `episode_XX.jsonl` (meta + one line per decision: game time, wall time,
  action, value, probs, reward/return, phase, boss HP, player HP, and the compact post-step state:
  player, boss move/box, hazard boxes). `--log DIR` writes the same logs without a video, e.g. for a
  lockstep evaluation. `scripts/game_hits.py DIR...` uses them to find the likely cause of every hit
  (the hazard closest to the player's hurtbox around the step where HP dropped). With
  `--sim CHECKPOINT`, it plays the same policy in the nominal sim instead, for comparison. `overlay_game_video.py` resamples the capture to
  1280×720 @ 60 fps. It draws the sim renderer's own side panel (`Renderer._draw_panel`), synced
  through `video_start_epoch` and with phase and boss HP added. It encodes H.264 CRF 20.
* **Not reproducible.** Real-game episodes are not exactly reproducible, even with the same
  seed and a deterministic policy, because boss patterns diverge chaotically after a few seconds
  (see Determinism). Report means over ≥ 20 episodes, and do not compare single episodes. In
  particular, a "best episode" seed cannot be replayed to reproduce a video, unlike in the sim.

**First sim-to-real data point (zero-shot, 2026-10-17).** The checkpoint was
`runs/ppo_curriculum/best.zip`, a round-1 model trained on an older sim version. Mugman,
deterministic, no fine-tuning:

| setting | episodes | win rate | boss damage (mean / max) | max phase | mean survival | hits taken |
|---|---|---|---|---|---|---|
| lockstep | 5 | **1/5** (won in 63.4 s game time, 1 hit taken) | 54 % / 100 % | III | 37.8 s | 2.6 |
| free-run, recorded | 15 | 0/15 | 54 % / 86 % | III (2×) | 31.4 s | 3.0 |

No parries. Of the 20 episodes, 14 ended in phase II (big Goopy), 3 in phase I and 3 in phase III (one of them the win). Stats are in `data/game_eval/zero_shot_*.json`.
The teaser video `videos/game_zero_shot/teaser_zero_shot.mp4` shows a free-run episode that reaches
the tombstone at 86 % damage.

**Round 2: `r2_curriculum_hit3` (zero-shot, 2026-10-17/18).** `runs/r2_curriculum_hit3/final.zip`
(100 % sim wins, 0.17 HP lost per fight on 100 held-out sim fights). Mugman, deterministic, no
fine-tuning, seeds 10000+. Round 1 (`runs/ppo_curriculum/best.zip`) is included for comparison:

| model | mode | episodes | win rate | boss damage (mean) | hits taken (mean, all / wins) | no-hit wins | win time (mean / fastest) | end phase I / II / III |
|---|---|---|---|---|---|---|---|---|
| round 1 | lockstep | 5 | 1/5 (20 %) | 54 % | 2.6 / 1.0 | 0 | 63.4 s | 1 / 3 / 1 |
| round 1 | free-run | 15 | 0/15 | 54 % | 3.0 / – | 0 | – | 2 / 11 / 2 |
| r2_curriculum_hit3 | lockstep | 40 (2 × 20) | **33/40 (83 %)** (16/20, 17/20) | 94 % | 1.70 / 1.42 | 2 | 52.8 s / 49.1 s | 1 / 5 / 34 |
| r2_curriculum_hit3 | paced (recorded) | 10 | **9/10 (90 %)** | 96 % | 0.80 / 0.56 | 4 | 49.8 s / 45.5 s | 0 / 1 / 9 |
| r2_curriculum_hit3 | **lockstep, 1-tick velocities (per-tick plugin)** | 20 | **19/20 (95 %)** | 99 % | **0.80** / 0.68 | 9 | 50.5 s / 45.2 s | 0 / 0 / 20 |
| r3_domain_rand | lockstep, per-tick plugin | 20 | **19/20 (95 %)** | 99.5 % | **0.85** / 0.74 | 7 | 46.6 s / 44.5 s | 0 / 0 / 20 |
| **r4_finetune (final)** | lockstep, per-tick plugin | 20 | **20/20 (100 %)** | 100 % | **0.40** / 0.40 | **12** | 48.2 s / 44.3 s | 0 / 0 / 20 |
| **r4_finetune (final)** | paced (recorded) | 10 | **10/10 (100 %)** | 100 % | 0.70 / 0.70 | 5 | 47.1 s / 45.5 s | 0 / 0 / 10 |
| r2_curriculum_hit3 | free-run (recorded) | 10 | 0/10 | 23 % | 3.0 / – | 0 | – | 8 / 2 / 0 |
| r2_curriculum_hit3 | sim (nominal, `game_hits.py --sim`) | 200 | 200/200 | 100 % | 0.14 / 0.14 | – | – | – |

Stats: `data/game_eval/r2_zero_shot_lockstep.json`, `r2_lockstep_logged.json` (the second lockstep
run, with logs), `r2_paced_record.json` and `r2_zero_shot_free_record.json`. Before the session was
interrupted, three earlier paced runs (`videos/game_r2_zero_shot`, `game_r2_retry`, `game_r2_retry2`,
10 episodes) had won 10/10 with 0.9 hits each. Pooled, paced mode is 19/20 with 0.85 hits per fight,
against 1.70 in lockstep. With 20 vs 40 episodes that is unlikely to be noise. The cause was the
1-tick vs 4-tick velocity in the observation. **The plugin now samples every tick** (see Velocities). With
the same model, lockstep went from 33/40 and 1.70 hits per fight to **19/20 and 0.80**, the paced
numbers. Mean boss damage rose from 94 % to 99 %, and all 20 fights reached phase III. Stats:
`data/game_eval/r2_lockstep_vel1.json`. Hit causes: `r2_hits_lockstep_vel1.json`. The rows above
the per-tick row were measured with the old 4-tick velocities.

Videos (overlay panel, 1280×720 @ 60 fps): `videos/game_r2_final/r2_win_mugman.mp4` (paced episode 1:
won in 46.8 s game time with 0 hits, the fastest no-hit win of the run) and
`videos/game_r2_final/r2_loss_mugman.mp4` (paced episode 2: hit by the first punch at 8.0 s, then twice
in phase II, dies at 60 % boss damage). Raw captures and logs: `videos/game_r2_paced/`.

**What hits Mugman** (`scripts/game_hits.py`, outputs in `data/game_eval/r2_hits_*.json`).
Note the labels: the sim calls small Goopy's lunging body during a punch FIST, while the bridge
reports it as BODY (its FIST is the separate ArmCollider). Below, both are counted as "punch lunge".

| cause | sim (200 fights) | lockstep (20) | lockstep, 1-tick velocities (20) | r3_domain_rand, lockstep (20) | r4_finetune, lockstep (20) | r4_finetune, paced (10) | paced (10) | free-run (10) |
|---|---|---|---|---|---|---|---|---|
| phase I punch (lunge body; free-run also 2 arm hits) | 7 (0.04/fight) | 15 (0.75) | 5 (0.25) | 4 (0.20) | 1 (0.05) | 2 (0.20) | 3 (0.30) | 9 (0.90) |
| phase I hop / idle body contact | 0 | 8 (0.40) | 0 | 1 (0.05) | 0 | 1 (0.10) | 0 | 17 (1.70) |
| phase II big Goopy body (hops, idle) | 11 (0.06) | 6 (0.30) | 3 (0.15) | 1 (0.05) | 0 | 0 | 3 (0.30) | 1 (0.10) |
| phase II transform (body while growing) | 7 (0.04) | 1 (0.05) | 0 | 0 | 0 | 0 | 1 (0.10) | 2 (0.20) |
| phase II big Goopy fist | 0 | 1 (0.05) | 2 (0.10) | 1 (0.05) | 1 (0.05) | 0 | 0 | 1 (0.10) |
| phase III tombstone slam | 4 (0.02) | 2 (0.10) | 6 (0.30) | 10 (0.50) | 6 (0.30) | 4 (0.40)¹ | 1 (0.10) | 0 |
| **total hits / fight** | **0.145** | **1.65** | **0.80** | **0.85** | **0.40** | **0.70** | **0.80** | **3.00** |

* The biggest game-only cause is **small Goopy's punch**, mostly the first one (12 of 15 lockstep
  punch hits at t ≈ 7.5–10.4 s, 3 at 16–17 s). Mugman stands next to
  or under Goopy when the lunge starts, and the body circle (r 170, centre y 213, so its bottom is
  43 u above the ground) catches him. The lunge frames have the same sizes as in the sim: the
  dominant frame is (y 213, r 170) in both. What differs is how often the agent is inside it, which
  happens 9× (paced) to 20× (lockstep) more often in the game than in the sim. Positions before the punch already differ, because boss
  patterns diverge after the first second. This is the place to look for the remaining sim-to-real gap.
* Half of the phase I hop/idle hits in lockstep are one repeatable hit at t = 4.20 s (4 of 20 episodes,
  identical states, because the start is deterministic for a few seconds). Paced mode and the per-tick
  plugin avoid it: it came from the 4-tick velocities.
* **After the velocity fix (lockstep, 20 fights):** phase I hop/idle hits went from 8 to 0, and
  punch hits from 15 to 5. All 5 are the first punch at t = 8.0–10.1 s, 3 of them with Mugman
  standing near the right wall (x 441–577). The punch is still the main phase I cause. Tombstone
  slam hits rose from 2 to 6 (all at t = 41–45 s, around the first slams), partly because all 20
  fights now reached phase III (17 before). It is now the largest single cause (0.30 per fight vs
  0.02 in the sim), the next one to look at.
* **Round 3, domain-randomized (2026-10-18):** `runs/r3_domain_rand/final.zip` is
  r2_curriculum_hit3 fine-tuned for 10M steps with ±15 % physics randomization, on the sim from
  before the round-4 simulator fixes. Lockstep, per-tick plugin, Mugman, seeds 10000+: **19/20 wins, 0.85 hits per fight**,
  99.5 % boss damage, all 20 fights reached phase III, and 7 no-hit wins. That is the same as r2
  after the velocity fix (19/20, 0.80); the hit difference is within noise at 20 fights. It wins about 4 s faster
  (46.6 s vs 50.5 s mean win time). The phase I and II causes went down (punch 5 → 4, phase II
  5 → 2 in total), but tombstone slam hits rose from 6 to 10 (0.50 per fight, all at t = 39–50 s,
  9 of 10 with Mugman in the air). The tombstone is now the main real-game cause for both models.
  Two later sim fixes (the tombstone damage window, and the EX lockout: an EX during a slam was
  4/4 hits in the game) came after r3's training, so r3 has not trained against them. The logs
  cannot show whether an EX actually fired, because the policy holds EX on 51 % of phase III
  steps. Wall clock: 16.3 s per episode in lockstep (two training runs on the CPU), 5.5 minutes for
  20. Stats: `data/game_eval/r3dr_lockstep.json`, hit causes: `r3dr_hits_lockstep.json`.
* **Final model: round 4 (2026-10-18).** `runs/r4_finetune/final.zip` (sha256 `8830ea6e…`) is
  the round-3 model fine-tuned for 10M steps on the corrected sim (live-track velocities, jump, tomb and
  EX). In the sim it wins 100/100 held-out fights at 0.39 hits per fight. In the real game (Mugman,
  per-tick plugin, seeds 10000+), lockstep is **20/20 wins at 0.40 hits per fight**, with 12 no-hit wins;
  paced, recorded, is 10/10 at 0.70 with 5 no-hit wins. For the first time the real-game hit rate
  matches the sim's (0.40 vs 0.39). The phase I punch is down to 1 hit in 20 lockstep fights. The
  tombstone slam (6 of 8 lockstep hits, all at t = 41–50 s) is the only cause left. Comparison
  across rounds, all lockstep, Mugman:

  | round | model | real-game wins | hits / fight | no-hit wins | mean win time |
  |---|---|---|---|---|---|
  | 1 | ppo_curriculum (sim v3) | 1/5 | 2.6 | 0 | 63.4 s |
  | 2 | r2_curriculum_hit3, after the plugin velocity fix | 19/20 | 0.80 | 9 | 50.5 s |
  | 3 | r3_domain_rand (±15 % physics) | 19/20 | 0.85 | 7 | 46.6 s |
  | 4 | **r4_finetune (corrected sim)** | **20/20** | **0.40** | **12** | 48.2 s |

  Video: `videos/game_r4_final/r4_nohit_win_mugman.mp4` (paced episode 7: no-hit win in 45.5 s, the
  fastest of the run; titled "Final agent | trained only in a simulator | real Cuphead | Mugman"). There
  was no loss, so there is no loss video. Raw captures and logs: `videos/game_r4_paced/`. Stats:
  `data/game_eval/r4_lockstep.json`, `r4_paced_record.json`; hit causes: `r4_hits_lockstep.json`,
  `r4_hits_paced.json`. Wall clock: 16.9 s per lockstep episode and ~50 s per paced episode, with one
  training run on the CPU.

  ¹ One of the 4 paced tombstone hits has a 126 u gap to the nearest logged hazard, so its
  attribution is uncertain.
* Phases II and III are close to the sim. The tombstone strategy (stand next to it, aim up, dash out
  of the slam) transfers: 34/40 lockstep and 9/10 paced episodes reached phase III, and 33 of 34 and 9 of 9
  of them won.

**Why free-run differs.** The simulator was calibrated tick by tick against lockstep traces. Free-run
breaks three assumptions the policy relies on:
1. **Variable timing.** The game renders uncapped (~175 frames/s), so Update runs with a variable
   deltaTime, while fixed physics ticks at 60 Hz. Timings that are exact multiples of 1/60 s in
   lockstep jitter here.
2. **Decision latency.** An action is applied at whatever tick it arrives, 0–1 ticks late, and the
   observation it was based on is older still. In lockstep the agent acts on the exact state.
3. **Noisy velocities.** States are streamed per rendered frame, so the finite-difference velocities
   use variable, sub-tick intervals.

The free-run logs show the result. In 6 of 10 episodes the first hit is Goopy's **first hop** landing
on Mugman at t ≈ 1.0–1.2 s, a hop that lockstep and paced play dodge every time. After that, Mugman
is hit about every 6–8 s by phase I body contacts (26 of 30 hits are in phase I). Free-run is
therefore a different, harder setting, not a noisier version of the evaluation. Closing that gap
would mean training with latency and timing randomisation. It is not needed for recording, because
paced mode records the calibrated setting at real-time speed.

**Session notes (2026-10-18).** In the first launch, the client connected while the game was
already on the slot-select screen, and `reset` (goto) hung: the game kept responding, but it never
loaded the level, and two resets timed out after 60 s. Nothing was pressed on the slot screen.
Cuphead was closed with WM_CLOSE (no kill needed). The protected slots were checked unchanged, then
the game was relaunched, and the client connected on the title screen. From then on the reset took
2.4 s, and the 10 paced and 20 lockstep episodes ran without problems. If it happens again, restart
the game and connect before it reaches the slot screen.

## Known limitations
* Long-episode determinism, as above.
* With `invincible` off, contact knockback interrupts dashes, as in the real game.
* At time_scale ≥ 4, frame_skip below time_scale is rounded up.
* `capture.py` uses ddagrab (Desktop Duplication) cropped to the game window, so the game must be
  visible and on top of monitor 0. gdigrab (`method="gdigrab"`) returns a frozen frame when the
  game runs fullscreen in the foreground.
