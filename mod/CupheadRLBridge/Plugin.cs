using BepInEx;
using BepInEx.Configuration;
using HarmonyLib;
using UnityEngine;

namespace CupheadRLBridge
{
    [BepInPlugin(Guid, "Cuphead RL Bridge", Version)]
    public class Plugin : BaseUnityPlugin
    {
        public const string Guid = "cuphead_rl.bridge";
        public const string Version = "0.1.0";

        internal static BepInEx.Logging.ManualLogSource Log;
        internal static ConfigEntry<int> Port;
        internal static ConfigEntry<int> SaveSlot;
        internal static ConfigEntry<bool> AutoGotoSlime;
        internal static ConfigEntry<float> IntroTimeScale;
        internal static ConfigEntry<string> DebugSkipGuard;

        void Awake()
        {
            Log = Logger;
            Port = Config.Bind("Bridge", "Port", 47800, "Localhost TCP port for the Python client.");
            SaveSlot = Config.Bind("Bridge", "SaveSlot", SaveGuard.Slot,
                "Save slot used by the bridge. Fixed to 2: slots 0 and 1 are never written while the plugin is loaded. Winning marks Goopy beaten in slot 2.");
            if (SaveSlot.Value != SaveGuard.Slot)
                Log.LogError("SaveSlot=" + SaveSlot.Value + " is not allowed; using slot " + SaveGuard.Slot);
            AutoGotoSlime = Config.Bind("Bridge", "AutoGotoSlime", false, "Skip the title/slot menus and load the Goopy fight at startup, without waiting for a client.");
            IntroTimeScale = Config.Bind("Bridge", "IntroTimeScale", 8f, "Time.timeScale used while (re)loading the fight and playing its intro.");

            DebugSkipGuard = Config.Bind("Debug", "DebugSkipGuard", "",
                "TESTING ONLY: name of a save-guard class to leave unpatched, to check that the plugin then refuses to start. Leave empty.");

            // Save guards first, explicitly, and verified; fail closed if any is missing.
            var harmony = new Harmony(Guid);
            if (!SaveGuard.Install(harmony, DebugSkipGuard.Value))
            {
                Log.LogError("Cuphead RL Bridge " + Version + " NOT started: save guards incomplete (see above). No server, no auto-goto.");
                return;
            }
            foreach (var t in AccessTools.GetTypesFromAssembly(typeof(Plugin).Assembly))
                if (!SaveGuard.IsGuardType(t)) harmony.CreateClassProcessor(t).Patch();
            SaveGuard.PinSlot();
            var go = new GameObject("CupheadRLBridge");
            Object.DontDestroyOnLoad(go);
            go.hideFlags = HideFlags.HideAndDontSave;
            go.AddComponent<Bridge>();
            Log.LogInfo("Cuphead RL Bridge " + Version + " listening on 127.0.0.1:" + Port.Value);
        }
    }

    [HarmonyPatch(typeof(PlayerStatsManager), "OnParry")]
    static class ParryCounter
    {
        static void Postfix(PlayerStatsManager __instance)
        {
            if (Bridge.Instance != null && __instance.basePlayer != null && __instance.basePlayer.id == PlayerId.PlayerOne)
                Bridge.Instance.OnParry();
        }
    }
}

namespace CupheadRLBridge
{
    [HarmonyPatch(typeof(SlimeLevel), "Start")]
    static class SeedOnLevelStart
    {
        static void Prefix()
        {
            if (Bridge.PendingSeed.HasValue) Random.InitState(Bridge.PendingSeed.Value);
        }
    }
}

namespace CupheadRLBridge
{
    [HarmonyPatch(typeof(PlayerDamageReceiver), "TakeDamage")]
    static class NoHitPatch
    {
        static bool Prefix() { return !Bridge.NoHit; }
    }
}
