using System;
using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;

namespace CupheadRLBridge
{
    /// <summary>
    /// While the plugin is loaded, the game may persist only the bridge's save slot. Slots 0 and 1
    /// hold the user's real progress: their keys never reach the disk, ClearSlot/Save on them are
    /// no-ops, and CurrentSaveFileIndex is pinned to the bridge slot from plugin load on (before
    /// the title or slot screen could call SaveCurrentFile).
    /// </summary>
    public static class SaveGuard
    {
        /// <summary>The only save slot the bridge may use. Hard-coded on purpose.</summary>
        public const int Slot = 2;
        const string KeyPrefix = "cuphead_player_data_v1_slot_";
        public static readonly string AllowedKey = KeyPrefix + Slot;

        /// <summary>Player-data keys that were dropped (for the log / tests).</summary>
        public static int Blocked;

        /// <summary>
        /// True only after <see cref="Install"/> verified every guard. Until then (and forever if a
        /// guard failed) SaveCloudData drops every player-data key, slot 2 included.
        /// </summary>
        public static bool Complete;

        /// <summary>The guards, patched first and one by one (PatchAll order is undefined).</summary>
        static readonly Type[] GuardTypes =
        {
            typeof(SaveCloudDataGuard), typeof(SaveSlotGuard), typeof(ClearSlotGuard), typeof(SaveFileIndexGuard),
        };

        /// <summary>Methods that must carry our patch, as "DeclaringType.Method".</summary>
        static readonly string[] Required =
        {
            "OnlineInterfaceSteam.SaveCloudData", "PlayerData.ClearSlot", "PlayerData.Save", "PlayerData.set_CurrentSaveFileIndex",
        };

        public static bool IsGuardType(Type t) { return Array.IndexOf(GuardTypes, t) >= 0; }

        public static bool IsForbiddenKey(string key)
        {
            if (key == null || !key.StartsWith(KeyPrefix)) return false;
            return !Complete || key != AllowedKey;
        }

        /// <summary>
        /// Patch and verify the guards. On any failure: unpatch everything, re-apply only the
        /// SaveCloudData guard (now dropping every player-data key) if possible, and return false;
        /// the caller must then not start the bridge. ``skip`` (config DebugSkipGuard) leaves one
        /// guard out on purpose, to test this path without provoking any write.
        /// </summary>
        internal static bool Install(Harmony harmony, string skip)
        {
            Complete = false;
            foreach (var t in GuardTypes)
            {
                if (!string.IsNullOrEmpty(skip) && t.Name == skip)
                {
                    Plugin.Log.LogWarning("SaveGuard: DebugSkipGuard=" + skip + ", not patching it (fail-closed test)");
                    continue;
                }
                try { harmony.CreateClassProcessor(t).Patch(); }
                catch (Exception e) { Plugin.Log.LogError("SaveGuard: patching " + t.Name + " failed: " + e); }
            }
            CollectPatched();
            var missing = new List<string>();
            foreach (var r in Required) if (!Patched.Contains(r)) missing.Add(r);
            if (missing.Count == 0)
            {
                Complete = true;
                Plugin.Log.LogInfo("SaveGuard: " + string.Join(", ", Patched.ToArray()));
                return true;
            }
            Plugin.Log.LogError("SaveGuard: MISSING " + string.Join(", ", missing.ToArray())
                + ". The bridge will NOT start, and all player-data saves are dropped while this plugin is loaded.");
            harmony.UnpatchSelf();
            try { harmony.CreateClassProcessor(typeof(SaveCloudDataGuard)).Patch(); }
            catch (Exception e) { Plugin.Log.LogError("SaveGuard: re-patching SaveCloudData failed: " + e); }
            CollectPatched();
            Plugin.Log.LogError("SaveGuard: fail-closed state, patched: [" + string.Join(", ", Patched.ToArray()) + "]");
            return false;
        }

        internal static void Warn(string what)
        {
            Blocked++;
            Plugin.Log.LogWarning("SaveGuard: blocked " + what + " (only slot " + Slot + " may be written)");
        }

        /// <summary>Guarded methods Harmony actually patched, e.g. "OnlineInterfaceSteam.SaveCloudData".</summary>
        public static readonly List<string> Patched = new List<string>();

        internal static void CollectPatched()
        {
            Patched.Clear();
            foreach (var m in Harmony.GetAllPatchedMethods())
            {
                var info = Harmony.GetPatchInfo(m);
                if (info == null || !info.Owners.Contains(Plugin.Guid)) continue;
                bool ours = m.Name == "SaveCloudData" || (m.DeclaringType == typeof(PlayerData)
                    && (m.Name == "Save" || m.Name == "ClearSlot" || m.Name == "set_CurrentSaveFileIndex"));
                if (ours) Patched.Add(m.DeclaringType.Name + "." + m.Name);
            }
            Patched.Sort();
        }

        internal static void PinSlot()
        {
            PlayerData.CurrentSaveFileIndex = Slot;  // goes through the setter patch below
        }
    }

    /// <summary>The one path to disk: OnlineInterface(Steam).SaveCloudData(dict, handler).</summary>
    [HarmonyPatch]
    static class SaveCloudDataGuard
    {
        static IEnumerable<MethodBase> TargetMethods()
        {
            foreach (var t in AccessTools.AllTypes())
            {
                if (t.IsInterface || t.IsAbstract || !typeof(OnlineInterface).IsAssignableFrom(t)) continue;
                var m = AccessTools.Method(t, "SaveCloudData");
                if (m != null && m.DeclaringType == t) yield return m;
            }
        }

        static bool Prefix(ref IDictionary<string, string> data, SaveCloudDataHandler handler)
        {
            if (data == null) return true;
            Dictionary<string, string> kept = null;
            foreach (var kv in data)
            {
                if (!SaveGuard.IsForbiddenKey(kv.Key)) continue;
                if (kept == null) kept = new Dictionary<string, string>(data);
                kept.Remove(kv.Key);
                SaveGuard.Warn("write of " + kv.Key);
            }
            if (kept == null) return true;
            if (kept.Count == 0)
            {
                // Report success so PlayerData does not retry forever.
                if (handler != null) handler(true);
                return false;
            }
            data = kept;
            return true;
        }
    }

    [HarmonyPatch(typeof(PlayerData), "Save", new[] { typeof(int) })]
    static class SaveSlotGuard
    {
        static bool Prefix(int fileIndex)
        {
            if (fileIndex == SaveGuard.Slot) return true;
            SaveGuard.Warn("PlayerData.Save(" + fileIndex + ")");
            return false;
        }
    }

    [HarmonyPatch(typeof(PlayerData), "ClearSlot")]
    static class ClearSlotGuard
    {
        static bool Prefix(int slot)
        {
            if (slot == SaveGuard.Slot) return true;
            SaveGuard.Warn("PlayerData.ClearSlot(" + slot + ")");
            return false;
        }
    }

    [HarmonyPatch(typeof(PlayerData), "CurrentSaveFileIndex", MethodType.Setter)]
    static class SaveFileIndexGuard
    {
        static void Prefix(ref int value)
        {
            if (value == SaveGuard.Slot) return;
            Plugin.Log.LogWarning("SaveGuard: CurrentSaveFileIndex " + value + " -> " + SaveGuard.Slot);
            value = SaveGuard.Slot;
        }
    }
}
