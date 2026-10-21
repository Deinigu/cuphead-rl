using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace CupheadRLBridge
{
    /// <summary>
    /// Main-thread side of the bridge: executes client commands, streams state, injects actions,
    /// drives the time mode (real-time / accelerated / lockstep) and navigates into the fight.
    ///
    /// Lockstep: <c>Time.captureFramerate = 60</c> makes every rendered frame advance game time by
    /// exactly one fixed tick (Update and FixedUpdate stay in sync, unlike blocking inside
    /// FixedUpdate). After <c>frames</c> ticks the bridge sends the state at end of frame and blocks
    /// the main thread until the next action arrives, so the game runs as fast as it can render.
    /// </summary>
    [DefaultExecutionOrder(-32000)]
    public sealed class Bridge : MonoBehaviour
    {
        public static Bridge Instance;

        /// <summary>cuphead_rl.spec.SPEC_VERSION this plugin implements.</summary>
        public const int SpecVersion = 2;

        /// <summary>Sequence id of the last applied action and the tick it was applied at.</summary>
        public static long LastSeq = -1, ActTick;

        /// <summary>Seed for UnityEngine.Random, applied when the next fight's SlimeLevel starts.</summary>
        public static int? PendingSeed;
        // Set per reset: play the level load and intro at real-time speed (for recordings) instead of
        // IntroTimeScale, and in lockstep cap rendering at 60 fps until the fight starts.
        bool realtimeIntro;

        /// <summary>Debug: player one ignores all hits (no damage, no knockback).</summary>
        public static bool NoHit;

        Server server;
        readonly StateReader reader = new StateReader();

        // config
        int streamEvery = 1;      // send a state every N rendered frames in free-run mode (0 = only on request)
        bool lockstep;
        int framesPerAction = 1;  // fixed ticks per action in lockstep
        float timeScale = 1f;
        bool control = true;

        // runtime
        long tick;                // fixed ticks since the current fight's scene loaded
        long ticksSinceAction;
        bool stepPending;         // lockstep: an act is waiting for its state reply
        int frame;
        bool gotoRequested, resetting;
        Level levelAtReset;
        float resetStartRealtime;
        int originalVSync = -1;
        float originalFixedDt = -1f;
        long ticksAtFrameStart;
        public static int OddFrames;  // lockstep frames that did not run exactly one tick
        Level lastLevel;
        int lastGeneration;

        static readonly FieldInfo fSlotState = AccessTools.Field(typeof(SlotSelectScreen), "state");

        void Awake()
        {
            Instance = this;
            ApplyCharacter();  // the game's flag is the one source of truth for hello and state
            Application.runInBackground = true;
            server = new Server(Plugin.Port.Value);
            server.Log += m => Plugin.Log.LogInfo(m);
            server.Start();
            if (Plugin.AutoGotoSlime.Value) gotoRequested = true;
            StartCoroutine(EndOfFrameLoop());
        }

        void OnDestroy()
        {
            if (server != null) server.Stop();
        }

        public void OnParry() { reader.Parries++; }

        SlimeLevel Fight { get { return Level.Current as SlimeLevel; } }

        bool FightRunning
        {
            get
            {
                var l = Fight;
                return l != null && l.Started && !resetting;
            }
        }

        // ---------------------------------------------------------------- frame hooks
        void FixedUpdate()
        {
            tick++;
            InputOverride.BeginTick();
            if (FightRunning) ticksSinceAction++;
        }

        /// <summary>Debug no-hit never outlives the client or episode that asked for it.</summary>
        static void ClearNoHit()
        {
            NoHit = false;
            AccessTools.Property(typeof(PlayerStatsManager), "DebugInvincible").SetValue(null, false, null);
        }

        void CheckConnection()
        {
            int g = server.Generation;
            if (g == lastGeneration) return;
            lastGeneration = g;
            ClearNoHit();
            LastSeq = -1;
        }

        /// <summary>Player one's character for the next level load (spec: the agent plays Mugman by default).</summary>
        public static bool Mugman = true;

        public static string Character { get { return PlayerManager.player1IsMugman ? "mugman" : "cuphead"; } }

        /// <summary>As SlotSelectScreen.EnterGame does it; only the bridge slot's data is touched.</summary>
        static void ApplyCharacter()
        {
            PlayerManager.player1IsMugman = Mugman;
            if (PlayerData.Initialized) PlayerData.GetDataForSlot(SaveGuard.Slot).isPlayer1Mugman = Mugman;
        }

        /// <summary>The contract loadout: a fresh save's (peashooter, no secondary, no charm, no super).</summary>
        static void ForceLoadout()
        {
            if (!PlayerData.Initialized) return;
            var lo = PlayerData.Data.Loadouts.GetPlayerLoadout(PlayerId.PlayerOne);
            lo.primaryWeapon = Weapon.level_weapon_peashot;
            lo.secondaryWeapon = Weapon.None;
            lo.charm = Charm.None;
            lo.super = Super.None;
        }

        static void WriteLoadout(JsonWriter w)
        {
            w.BeginObject("loadout");
            if (PlayerData.Initialized)
            {
                var lo = PlayerData.Data.Loadouts.GetPlayerLoadout(PlayerId.PlayerOne);
                w.S("primary", lo.primaryWeapon.ToString()).S("secondary", lo.secondaryWeapon.ToString())
                    .S("charm", lo.charm.ToString()).S("super", lo.super.ToString());
            }
            w.End();
        }

        void Update()
        {
            CheckConnection();
            string line;
            int guard = 0;
            while (guard++ < 256 && server.TryDequeue(out line)) Handle(line);

            var l = Fight;
            if (l != null && !ReferenceEquals(lastLevel, l))
            {
                lastLevel = l;
                reader.OnNewLevel(l);
                InputOverride.ResetPlayerCache();
                tick = 0;
                ticksSinceAction = 0;
            }
            if (l == null) lastLevel = null;

            bool wantActive = control && server.Connected && l != null && !resetting;
            if (InputOverride.Active && !wantActive) InputOverride.Release();
            InputOverride.Active = wantActive;

            Navigate();
            ApplyTimeMode();
        }

        IEnumerator EndOfFrameLoop()
        {
            var eof = new WaitForEndOfFrame();
            while (true)
            {
                yield return eof;
                frame++;
                long ran = tick - ticksAtFrameStart;
                ticksAtFrameStart = tick;
                if (lockstep && FightRunning && Time.timeScale == 1f && ran != 1) OddFrames++;
                sentThisFrame = false;
                try
                {
                    AfterFrame();
                    // Per-tick velocities: sample on every fight frame that sent no state.
                    if (!sentThisFrame && Fight != null && !resetting) reader.Track(tick, frame);
                }
                catch (Exception e) { Plugin.Log.LogError(e); }
            }
        }

        void AfterFrame()
        {
            if (resetting && Fight != null && !ReferenceEquals(Fight, levelAtReset) && Fight.Started)
            {
                resetting = false;
                ticksSinceAction = 0;
                ApplyTimeMode();
                var rw = new JsonWriter().Begin().S("type", "reset_done")
                    .F("seconds", Time.realtimeSinceStartup - resetStartRealtime).B("no_hit", NoHit).S("character", Character);
                WriteLoadout(rw);
                server.Send(rw.End().ToString());
                SendState();
                if (lockstep) BlockForAction();
                return;
            }
            if (!server.Connected) return;
            if (lockstep && FightRunning)
            {
                if (ticksSinceAction >= framesPerAction)
                {
                    SendState();
                    BlockForAction();
                }
            }
            else if (lockstep && stepPending && !resetting)
            {
                // The fight ended (win/lose screen loaded) while a step was in flight: answer it
                // with an in_fight=false state instead of leaving the client waiting.
                SendState();
            }
            else if (streamEvery > 0 && frame % streamEvery == 0)
            {
                SendState(droppable: true);
            }
        }

        bool sentThisFrame;

        void SendState(bool droppable = false)
        {
            stepPending = false;
            sentThisFrame = true;
            server.Send(reader.Read(tick, frame, lockstep ? "lockstep" : "free"), droppable);
        }

        void BlockForAction()
        {
            while (true)
            {
                string line;
                CheckConnection();
                if (!server.WaitForLine(500, out line))
                {
                    if (!server.Connected)
                    {
                        lockstep = false;
                        InputOverride.Release();
                        ApplyTimeMode();
                        return;
                    }
                    continue;
                }
                bool acted = Handle(line);
                if (acted || resetting || !lockstep) return;
            }
        }

        // ---------------------------------------------------------------- commands
        /// <returns>true when the line was an action (lockstep resumes).</returns>
        bool Handle(string line)
        {
            Dictionary<string, object> m;
            try { m = Json.ParseObject(line); }
            catch (Exception e)
            {
                Reply("error", "bad json: " + e.Message);
                return false;
            }
            string cmd = Json.Str(m, "cmd", "");
            switch (cmd)
            {
                case "act":
                    InputOverride.SetAction((int)Json.Num(m, "move", 0), Json.Bool(m, "jump", false),
                        Json.Bool(m, "dash", false), Json.Bool(m, "shoot", false), Json.Bool(m, "ex", false),
                        (int)Json.Num(m, "move_y", 0));
                    framesPerAction = Math.Max(1, (int)Json.Num(m, "frames", framesPerAction));
                    ticksSinceAction = 0;
                    stepPending = lockstep;
                    LastSeq = (long)Json.Num(m, "seq", LastSeq + 1);
                    ActTick = tick;
                    return true;
                case "hello":
                    SaveGuard.CollectPatched();  // live Harmony state, not a cached list
                    var hw = new JsonWriter().Begin().S("type", "hello").S("version", Plugin.Version).I("spec_version", SpecVersion)
                        .F("fixed_dt", Time.fixedDeltaTime).S("scene", SceneManager.GetActiveScene().name)
                        .B("in_fight", Fight != null).B("no_hit", NoHit).S("character", Character).I("save_slot", PlayerData.CurrentSaveFileIndex)
                    .I("saves_blocked", SaveGuard.Blocked).S("save_guard", string.Join(",", SaveGuard.Patched.ToArray()));
                    WriteLoadout(hw);
                    server.Send(hw.End().ToString());
                    return false;
                case "config":
                    streamEvery = (int)Json.Num(m, "stream_every", streamEvery);
                    lockstep = Json.Bool(m, "lockstep", lockstep);
                    framesPerAction = Math.Max(1, (int)Json.Num(m, "frames", framesPerAction));
                    timeScale = Mathf.Clamp((float)Json.Num(m, "time_scale", timeScale), 0f, 20f);
                    control = Json.Bool(m, "control", control);
                    reader.Debug = Json.Bool(m, "debug", reader.Debug);
                    ApplyTimeMode();
                    server.Send(new JsonWriter().Begin().S("type", "config").I("stream_every", streamEvery)
                        .B("lockstep", lockstep).I("frames", framesPerAction).F("time_scale", timeScale)
                        .B("control", control).B("debug", reader.Debug).End().ToString());
                    return false;
                case "state":
                    SendState();
                    return false;
                case "reset":
                case "goto":
                    PendingSeed = m.ContainsKey("seed") && m["seed"] != null ? (int?)(int)Json.Num(m, "seed", 0) : null;
                    realtimeIntro = Json.Bool(m, "realtime_intro", false);
                    string ch = Json.Str(m, "character", null);
                    if (ch == "mugman" || ch == "cuphead") { Mugman = ch == "mugman"; ApplyCharacter(); }
                    else if (ch != null) { Reply("error", "unknown character '" + ch + "' (mugman|cuphead)"); return false; }
                    StartReset();
                    return false;
                case "damage":
                    // Debug/curriculum: deal damage to the boss as if it had been shot.
                    if (Fight != null) StateReader.Props(Fight).DealDamage((float)Json.Num(m, "amount", 0));
                    SendState();
                    return false;
                case "invincible":
                    // Debug: PlayerStatsManager.DebugInvincible (player takes no damage).
                    NoHit = Json.Bool(m, "on", true);
                    AccessTools.Property(typeof(PlayerStatsManager), "DebugInvincible").SetValue(null, NoHit, null);
                    SendState();
                    return false;
                case "set_meter":
                    // Debug: set player one's super meter, in cards (0..5).
                    var pl = PlayerManager.GetPlayer(PlayerId.PlayerOne);
                    if (pl != null) AccessTools.Property(typeof(PlayerStatsManager), "SuperMeter")
                        .SetValue(pl.stats, Mathf.Clamp((float)Json.Num(m, "cards", 0) * 10f, 0f, 50f), null);
                    SendState();
                    return false;
                case "release":
                    InputOverride.Release();
                    return false;
                default:
                    Reply("error", "unknown cmd '" + cmd + "'");
                    return false;
            }
        }

        void Reply(string type, string msg)
        {
            server.Send(new JsonWriter().Begin().S("type", type).S("message", msg).End().ToString());
        }

        // ---------------------------------------------------------------- reset / navigation
        void StartReset()
        {
            InputOverride.Release();
            ClearNoHit();
            resetting = true;
            resetStartRealtime = Time.realtimeSinceStartup;
            levelAtReset = Level.Current;
            if (Fight != null && !SceneLoader.CurrentlyLoading)
            {
                SaveGuard.PinSlot();
                ApplyCharacter();
                ForceLoadout();
                PauseManager.Unpause();
                Level.SetCurrentMode(Level.Mode.Normal);
                SceneLoader.ReloadLevel();
            }
            else
            {
                gotoRequested = true;
            }
        }

        void Navigate()
        {
            if (!gotoRequested)
            {
                InputOverride.ForceAnyButton = false;
                return;
            }
            if (!resetting)
            {
                resetting = true;
                resetStartRealtime = Time.realtimeSinceStartup;
                levelAtReset = Level.Current;
            }
            if (SceneLoader.CurrentlyLoading) return;
            string scene = SceneManager.GetActiveScene().name;
            if (scene == "scene_title")
            {
                InputOverride.ForceAnyButton = true;  // "press any button" → player one joins
                return;
            }
            InputOverride.ForceAnyButton = false;
            if (scene == "scene_slot_select")
            {
                var screen = FindObjectOfType<SlotSelectScreen>();
                if (screen == null || !PlayerData.Initialized) return;
                if (fSlotState.GetValue(screen).ToString() != "MainMenu") return;
                EnterSlime();
                return;
            }
            if (scene.StartsWith("scene_level") || scene.StartsWith("scene_map") || scene.StartsWith("scene_win") || scene.StartsWith("scene_cutscene"))
            {
                if (PlayerData.Initialized) EnterSlime();
            }
        }

        void EnterSlime()
        {
            gotoRequested = false;
            SaveGuard.PinSlot();
            PlayerData.inGame = true;
            ApplyCharacter();
            ForceLoadout();
            Level.ResetPreviousLevelInfo();
            Level.SetCurrentMode(Level.Mode.Normal);
            if (StartScreenAudio.Instance != null) Destroy(StartScreenAudio.Instance.gameObject);
            PauseManager.Unpause();
            SceneLoader.LoadLevel(Levels.Slime, SceneLoader.Transition.Fade, SceneLoader.Icon.None);
            Plugin.Log.LogInfo("Loading Goopy Le Grande (slot " + PlayerData.CurrentSaveFileIndex + ")");
        }

        void ApplyTimeMode()
        {
            Time.timeScale = resetting ? (realtimeIntro ? 1f : Plugin.IntroTimeScale.Value) : timeScale;
            if (lockstep)
            {
                if (originalVSync < 0) originalVSync = QualitySettings.vSyncCount;
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = resetting && realtimeIntro ? 60 : -1;
                Time.captureFramerate = 60;
                // fixedDeltaTime ships as 0.016666669 but a captured frame is 1/60f = 0.016666668:
                // with them unequal, absolute-time float drift makes some frames run 0 or 2 ticks
                // and runs stop being reproducible. Make them identical while in lockstep.
                if (originalFixedDt < 0) originalFixedDt = Time.fixedDeltaTime;
                Time.fixedDeltaTime = 1f / 60f;
            }
            else
            {
                Time.captureFramerate = 0;
                if (originalFixedDt > 0) { Time.fixedDeltaTime = originalFixedDt; originalFixedDt = -1f; }
                if (originalVSync >= 0) { QualitySettings.vSyncCount = originalVSync; originalVSync = -1; }
                Application.targetFrameRate = 60;
            }
        }
    }
}
