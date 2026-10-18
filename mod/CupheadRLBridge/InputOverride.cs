using System;
using HarmonyLib;
using UnityEngine;

namespace CupheadRLBridge
{
    /// <summary>
    /// Replaces player one's Rewired input with the agent's action. Cuphead reads input through
    /// <c>PlayerInput.actions</c> (a <c>Rewired.Player</c>) with <c>GetButton/GetButtonDown/
    /// GetButtonUp/GetAxis(int actionId)</c>, so prefixing those methods for player one's
    /// instance is enough; no OS-level key events are involved and the window may be unfocused.
    ///
    /// Edges are latched per fixed tick (all gameplay input is read in FixedUpdate), using
    /// Time.fixedTime as the tick id, so every script sees the same press on the same tick.
    /// </summary>
    public static class InputOverride
    {
        // CupheadButton ids
        public const int MoveHorizontal = 0, MoveVertical = 1, Jump = 2, Shoot = 3, Super = 4, Lock = 6, Dash = 7;

        sealed class Btn
        {
            public readonly bool pulse;  // pressed for exactly one tick per request
            public bool held, pendingDown, pendingUp;
            public double downId = double.NaN, upId = double.NaN;

            public Btn(bool pulse) { this.pulse = pulse; }

            public void Set(bool want)
            {
                if (pulse)
                {
                    if (want) pendingDown = true;
                    return;
                }
                if (want && !held) { held = true; pendingDown = true; pendingUp = false; }
                else if (!want && held) { held = false; pendingUp = true; pendingDown = false; }
            }

            public void Latch(double id)
            {
                if (pulse)
                {
                    if (held && downId != id) { held = false; upId = id; }
                    if (pendingDown) { pendingDown = false; held = true; downId = id; }
                    return;
                }
                if (pendingDown) { pendingDown = false; downId = id; }
                if (pendingUp) { pendingUp = false; upId = id; }
            }

            public void Clear()
            {
                held = pendingDown = pendingUp = false;
                downId = upId = double.NaN;
            }
        }

        static readonly Btn jump = new Btn(false), shoot = new Btn(false), dash = new Btn(true), ex = new Btn(true);
        static int move, moveY;
        static double lastLatchId = double.NaN;

        /// <summary>True while the bridge owns player one's input.</summary>
        public static bool Active;

        /// <summary>When set, any "press any button" query returns true (title-screen auto-join).</summary>
        public static bool ForceAnyButton;

        static Rewired.Player playerOne;

        public static void SetAction(int moveX, bool jumpHeld, bool dashPress, bool shootHeld, bool exPress, int moveYAxis = 0)
        {
            move = Math.Sign(moveX);
            moveY = Math.Sign(moveYAxis);
            jump.Set(jumpHeld);
            shoot.Set(shootHeld);
            dash.Set(dashPress);
            ex.Set(exPress);
        }

        public static void Release()
        {
            move = moveY = 0;
            jump.Clear(); shoot.Clear(); dash.Clear(); ex.Clear();
        }

        static double CurrentId()
        {
            return Time.inFixedTimeStep ? Time.fixedTime : -Time.frameCount;
        }

        /// <summary>Called at the start of each fixed tick (and lazily on first query).</summary>
        public static void BeginTick()
        {
            double id = CurrentId();
            if (id == lastLatchId) return;
            lastLatchId = id;
            jump.Latch(id); shoot.Latch(id); dash.Latch(id); ex.Latch(id);
        }

        static Btn Get(int actionId)
        {
            switch (actionId)
            {
                case Jump: return jump;
                case Shoot: return shoot;
                case Dash: return dash;
                case Super: return ex;
                default: return null;
            }
        }

        static bool Mine(Rewired.Player p)
        {
            if (!Active || p == null) return false;
            if (playerOne == null)
            {
                try { playerOne = PlayerManager.GetPlayerInput(PlayerId.PlayerOne); } catch { return false; }
            }
            return ReferenceEquals(p, playerOne);
        }

        public static void ResetPlayerCache() { playerOne = null; }

        [HarmonyPatch(typeof(Rewired.Player), "GetButton", new[] { typeof(int) })]
        static class PGetButton
        {
            static bool Prefix(Rewired.Player __instance, int __0, ref bool __result)
            {
                if (!Mine(__instance)) return true;
                BeginTick();
                Btn b = Get(__0);
                __result = b != null && b.held;
                return false;
            }
        }

        [HarmonyPatch(typeof(Rewired.Player), "GetButtonDown", new[] { typeof(int) })]
        static class PGetButtonDown
        {
            static bool Prefix(Rewired.Player __instance, int __0, ref bool __result)
            {
                if (!Mine(__instance)) return true;
                BeginTick();
                Btn b = Get(__0);
                __result = b != null && b.held && b.downId == CurrentId();
                return false;
            }
        }

        [HarmonyPatch(typeof(Rewired.Player), "GetButtonUp", new[] { typeof(int) })]
        static class PGetButtonUp
        {
            static bool Prefix(Rewired.Player __instance, int __0, ref bool __result)
            {
                if (!Mine(__instance)) return true;
                BeginTick();
                Btn b = Get(__0);
                __result = b != null && !b.held && b.upId == CurrentId();
                return false;
            }
        }

        [HarmonyPatch(typeof(Rewired.Player), "GetAxis", new[] { typeof(int) })]
        static class PGetAxis
        {
            static bool Prefix(Rewired.Player __instance, int __0, ref float __result)
            {
                if (!Mine(__instance)) return true;
                __result = __0 == MoveHorizontal ? move : __0 == MoveVertical ? moveY : 0f;
                return false;
            }
        }

        [HarmonyPatch(typeof(Rewired.Player), "GetAxisRaw", new[] { typeof(int) })]
        static class PGetAxisRaw
        {
            static bool Prefix(Rewired.Player __instance, int __0, ref float __result)
            {
                if (!Mine(__instance)) return true;
                __result = __0 == MoveHorizontal ? move : __0 == MoveVertical ? moveY : 0f;
                return false;
            }
        }

        [HarmonyPatch(typeof(Rewired.Player), "GetAnyButtonDown")]
        static class PAnyButtonDown
        {
            static bool Prefix(ref bool __result)
            {
                if (!ForceAnyButton) return true;
                __result = true;
                return false;
            }
        }

        [HarmonyPatch(typeof(Rewired.Controller), "GetAnyButtonDown")]
        static class PControllerAnyButtonDown
        {
            static bool Prefix(Rewired.Controller __instance, ref bool __result)
            {
                if (!ForceAnyButton || !(__instance is Rewired.Keyboard)) return true;
                __result = true;
                return false;
            }
        }
    }
}
