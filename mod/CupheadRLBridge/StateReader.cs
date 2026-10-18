using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;
using UnityEngine;

namespace CupheadRLBridge
{
    /// <summary>
    /// Reads the Goopy Le Grande fight into the shape of <c>cuphead_rl.spec.GameState</c>.
    /// Coordinates are converted to spec space here: x unchanged (arena centre is 0),
    /// y measured from <c>Level.Ground</c>. Velocities are per-tick finite differences: the bridge
    /// calls <see cref="Track"/> on every frame that sends no state, so whatever the number of ticks
    /// between states (lockstep <c>frames</c>, stream rate), each velocity is the motion of the last
    /// tick, i.e. instantaneous, as in the sim.
    /// Enum ints match spec.BossPhase / spec.BossMove / spec.HazardKind.
    /// </summary>
    public sealed class StateReader
    {
        // spec.BossMove
        const int INTRO = 0, IDLE = 1, JUMP = 2, PUNCH_WINDUP = 3, PUNCH = 4, TRANSFORM = 5,
            TOMB_MOVE = 6, TOMB_SLAM_WINDUP = 7, TOMB_SLAM = 8, DEAD = 9;
        // spec.HazardKind
        const int BODY = 0, FIST = 1, TOMB = 2, PROJECTILE = 3, OTHER = 4;

        static readonly FieldInfo fProps = AccessTools.Field(typeof(SlimeLevel), "properties");
        static readonly FieldInfo fSmall = AccessTools.Field(typeof(SlimeLevel), "smallSlime");
        static readonly FieldInfo fBig = AccessTools.Field(typeof(SlimeLevel), "bigSlime");
        static readonly FieldInfo fTomb = AccessTools.Field(typeof(SlimeLevel), "tombStone");
        static readonly FieldInfo fInAir = AccessTools.Field(typeof(SlimeLevelSlime), "inAir");
        static readonly FieldInfo fWantsTransform = AccessTools.Field(typeof(SlimeLevelSlime), "wantsToTransform");
        static readonly FieldInfo fTombDealDamage = AccessTools.Field(typeof(SlimeLevelTombstone), "dealDamage");
        static readonly FieldInfo fDmgTimer = AccessTools.Field(typeof(PlayerDamageReceiver), "timer");
        static readonly FieldInfo fDashManager = AccessTools.Field(typeof(LevelPlayerMotor), "dashManager");
        static readonly FieldInfo fDashState = AccessTools.Field(typeof(LevelPlayerMotor.DashManager), "state");

        const long MaxVelGapTicks = 8;
        const float MaxSpeed = 6000f;  // fastest real mover: tombstone drop, ~3150 u/s

        struct Sample { public Vector2 pos; public long tick; }
        /// <summary>Body collider per entity, fixed the first time the entity is seen.</summary>
        readonly Dictionary<int, Collider2D> bodyOf = new Dictionary<int, Collider2D>();
        readonly Dictionary<int, Sample> prev = new Dictionary<int, Sample>();

        int lastMove = -1;
        float moveSince;
        float startLevelTime = -1f;
        SlimeLevel level;

        public int Parries;
        public bool Debug;

        public static LevelProperties.Slime Props(SlimeLevel l) { return (LevelProperties.Slime)fProps.GetValue(l); }
        public bool Won, Lost;

        public void OnNewLevel(SlimeLevel l)
        {
            level = l;
            prev.Clear();
            lastVel.Clear();
            bodyOf.Clear();
            lastMove = -1;
            startLevelTime = -1f;
            Parries = 0;
            Won = Lost = false;
        }

        readonly Dictionary<int, Vector2> lastVel = new Dictionary<int, Vector2>();

        /// <summary>Finite-difference velocity of a tracked object between fixed-time samples (normally 1 tick apart).</summary>
        Vector2 Vel(int key, Vector2 pos, long tick)
        {
            // dt from the integer tick counter: differencing Time.fixedTime (an absolute float)
            // loses precision and made velocities differ between otherwise identical runs.
            Sample s;
            Vector2 v = Vector2.zero;
            if (prev.TryGetValue(key, out s))
            {
                if (tick <= s.tick) return lastVel[key];  // sampled twice in the same tick
                // A collider re-enabled after a gap, or a teleport (SetPosition), is not motion.
                if (tick - s.tick <= MaxVelGapTicks) v = (pos - s.pos) / ((tick - s.tick) * Time.fixedDeltaTime);
                if (v.sqrMagnitude > MaxSpeed * MaxSpeed) v = Vector2.zero;
            }
            prev[key] = new Sample { pos = pos, tick = tick };
            lastVel[key] = v;
            return v;
        }

        static bool Alive(Object o) { return o != null; }

        public string Read(long tick, int frame, string mode) { return Build(tick, frame, mode, true); }

        /// <summary>Track positions (and move timing) on a frame that sends no state, so the next
        /// state's velocities cover one tick, not the whole action.</summary>
        public void Track(long tick, int frame) { Build(tick, frame, "sample", false); }

        string Build(long tick, int frame, string mode, bool full)
        {
            var w = new JsonWriter();
            w.Begin().S("type", "state").L("tick", tick).I("frame", frame).S("mode", mode);
            var lvl = Level.Current as SlimeLevel;
            if (lvl == null || !ReferenceEquals(lvl, level))
            {
                w.B("in_fight", false).S("scene", UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
                return w.End().ToString();
            }

            long t = tick;
            float groundY = lvl.Ground;
            var props = (LevelProperties.Slime)fProps.GetValue(lvl);
            var small = (SlimeLevelSlime)fSmall.GetValue(lvl);
            var big = (SlimeLevelSlime)fBig.GetValue(lvl);
            var tomb = (SlimeLevelTombstone)fTomb.GetValue(lvl);

            if (lvl.Started && startLevelTime < 0) startLevelTime = lvl.LevelTime;
            float time = startLevelTime < 0 ? 0f : lvl.LevelTime - startLevelTime;

            w.B("in_fight", true).B("started", lvl.Started).B("no_hit", Bridge.NoHit).S("character", Bridge.Character)
                .L("seq", Bridge.LastSeq).L("act_tick", Bridge.ActTick).I("odd_frames", Bridge.OddFrames).F("time", time).F("ground_y_unity", groundY);

            // ------------------------------------------------------------ player
            var player = PlayerManager.GetPlayer(PlayerId.PlayerOne) as LevelPlayerController;
            int hp = 0;
            if (player != null)
            {
                Vector2 pos = player.transform.position;
                Vector2 v = Vel(1, pos, t);
                var motor = player.motor;
                var dm = fDashManager.GetValue(motor);
                var dashState = (LevelPlayerMotor.DashManager.State)fDashState.GetValue(dm);
                var dr = player.damageReceiver;
                float invuln = dr.state == PlayerDamageReceiver.State.Invulnerable ? (float)fDmgTimer.GetValue(dr) : 0f;
                int facing = (int)motor.TrueLookDirection.x;
                hp = player.stats.Health;
                w.BeginObject("player")
                    .F("x", pos.x).F("y", pos.y - groundY).F("vx", v.x).F("vy", v.y)
                    .B("grounded", motor.Grounded).B("dashing", motor.Dashing)
                    .B("can_dash", dashState == LevelPlayerMotor.DashManager.State.Ready)
                    .I("facing", facing == 0 ? 1 : facing).I("hp", hp).F("invuln", invuln)
                    .F("super_meter", player.stats.SuperMeter / 10f)
                    .F("cx", player.center.x).F("cy", player.center.y - groundY)
                    .F("raw_y", pos.y).B("parrying", motor.Parrying).B("is_hit", motor.IsHit)
                    .B("ducking", motor.Ducking);
                var hb = player.GetComponent<Collider2D>();
                if (hb != null)
                {
                    var hbb = hb.bounds;
                    w.F("hb_x", hbb.center.x).F("hb_y", hbb.center.y - groundY).F("hb_half_w", hbb.extents.x).F("hb_half_h", hbb.extents.y);
                }
                w.End();
                if (player.IsDead || hp <= 0) Lost = true;
            }

            // ------------------------------------------------------------ boss
            float total = props.TotalHealth;
            float cur = Mathf.Max(0f, props.CurrentHealth);
            if (cur <= 0f) Won = true;
            // spec.BossPhase = the entity on screen (props.CurrentState flips at the HP threshold,
            // seconds before small Goopy lands and transforms / the tombstone drops): BIG from the
            // moment small Goopy starts transforming, TOMB once the tombstone is out.
            bool tombActive = Alive(tomb) && tomb.state != SlimeLevelTombstone.State.Init;
            int smallMove = Alive(small) ? SlimeMove(small) : TRANSFORM;
            int phase = tombActive ? 2 : smallMove != TRANSFORM ? 0 : 1;
            float lo = phase == 0 ? 0.76f : phase == 1 ? 0.31f : 0f;
            float hi = phase == 0 ? 1f : phase == 1 ? 0.76f : 0.31f;
            float phaseFrac = Mathf.Clamp01((cur / total - lo) / (hi - lo));

            AbstractLevelEntity active;
            int move;
            string entity;
            bool tombDealing = false;
            if (tombActive)
            {
                active = tomb;
                entity = "tomb";
                bool dealing = (bool)fTombDealDamage.GetValue(tomb);
                tombDealing = dealing;
                // smash_cr calls StartMove() as soon as the smash animation is triggered, so the
                // state machine says Move while the slam is still playing: use the clip name.
                string clip = ClipName(tomb);
                if (tomb.state == SlimeLevelTombstone.State.Intro) move = TRANSFORM;
                else if (dealing || clip.EndsWith("_smash")) move = TOMB_SLAM;
                else if (tomb.state == SlimeLevelTombstone.State.Smash || clip.Contains("smash")) move = TOMB_SLAM_WINDUP;
                else move = TOMB_MOVE;
            }
            else
            {
                var s = Alive(small) ? small : big;
                active = s;
                entity = Alive(small) ? "small" : "big";
                move = Alive(small) ? smallMove : Alive(big) ? SlimeMove(big) : TRANSFORM;
            }
            if (!lvl.Started) move = INTRO;
            if (Won) move = DEAD;
            if (move != lastMove) { lastMove = move; moveSince = time; }

            string anim = "";
            Vector2 bc = Vector2.zero, bh = Vector2.zero;
            int facingBoss = -1;
            bool vulnerable = false;
            if (Alive(active))
            {
                var an = active.GetComponent<Animator>();
                if (an != null && an.isActiveAndEnabled)
                {
                    var clips = an.GetCurrentAnimatorClipInfo(0);
                    if (clips.Length > 0 && clips[0].clip != null) anim = clips[0].clip.name;
                }
                var bodyCol = Body(active);
                if (bodyCol != null && bodyCol.enabled) { bc = bodyCol.bounds.center; bh = bodyCol.bounds.extents; }
                else bc = active.transform.position;
                facingBoss = active.transform.localScale.x > 0 ? -1 : 1;
                var rec = active.GetComponent<DamageReceiver>();
                vulnerable = lvl.Started && !Won && rec != null && rec.enabled && bh.x > 0;
            }
            // keyed by the body collider: a new entity (small -> big -> tomb) starts at v = 0
            var bodyKey = Alive(active) && Body(active) != null ? Body(active).GetInstanceID() : 2;
            Vector2 bv = Vel(bodyKey * 8 + 7, bc, t);
            if (tombActive) facingBoss = bv.x > 1f ? 1 : bv.x < -1f ? -1 : facingBoss;
            w.BeginObject("boss")
                .I("phase", phase).I("state", move).F("x", bc.x).F("y", bc.y - groundY)
                .F("half_w", bh.x).F("half_h", bh.y).F("vx", bv.x).F("vy", bv.y)
                .F("state_time", time - moveSince).I("facing", facingBoss)
                .F("hp", cur).F("max_hp", total).F("phase_hp_frac", phaseFrac).B("vulnerable", vulnerable)
                .S("entity", entity).S("anim", anim)
                .End();

            // ------------------------------------------------------------ hazards
            w.BeginArray("hazards");
            if (lvl.Started && !Won)
            {
                if (Alive(small)) Hazards(w, small, false, groundY, t);
                if (Alive(big) && !(tombActive && tomb.state != SlimeLevelTombstone.State.Intro)) Hazards(w, big, false, groundY, t);
                // spec: TOMB only while the slam deals damage; the patrolling tombstone is harmless
                if (tombActive && tombDealing) Hazards(w, tomb, true, groundY, t);
            }
            w.EndArray();

            if (Debug && full)
            {
                // Player bullets and every enabled boss collider, for hitbox sampling.
                w.BeginArray("bullets");
                foreach (var pr in Object.FindObjectsOfType<AbstractProjectile>())
                {
                    if (pr.tag != "PlayerProjectile" && pr.tag != "PlayerProjectileEx" && !pr.GetType().Name.StartsWith("Weapon")) continue;
                    var pc = pr.GetComponent<Collider2D>();
                    Vector2 pp = pc != null ? (Vector2)pc.bounds.center : (Vector2)pr.transform.position;
                    w.BeginElement().I("id", pr.GetInstanceID()).S("type", pr.GetType().Name).S("tag", pr.tag)
                        .F("x", pp.x).F("y", pp.y - groundY)
                        .F("half_w", pc != null ? pc.bounds.extents.x : 0f).F("half_h", pc != null ? pc.bounds.extents.y : 0f).End();
                }
                w.EndArray();
                w.BeginArray("colliders");
                foreach (var e in new AbstractLevelEntity[] { small, big, tomb })
                {
                    if (!Alive(e)) continue;
                    int idx = 0;
                    foreach (var c in e.GetComponentsInChildren<Collider2D>(true))
                    {
                        idx++;
                        var b = c.bounds;
                        var circ = c as CircleCollider2D;
                        w.BeginElement().S("owner", e.name).S("name", c.name + "#" + idx + ":" + c.GetType().Name).S("type", c.GetType().Name)
                            .B("enabled", c.enabled && c.gameObject.activeInHierarchy)
                            .F("x", b.center.x).F("y", b.center.y - groundY).F("half_w", b.extents.x).F("half_h", b.extents.y)
                            .F("lossy_sx", c.transform.lossyScale.x).F("lossy_sy", c.transform.lossyScale.y)
                            .F("radius", circ != null ? circ.radius : 0f)
                            .F("off_x", c.offset.x).F("off_y", c.offset.y)
                            .F("tf_x", e.transform.position.x).F("tf_y", c.transform.position.y - groundY)
                            .F("ent_y", e.transform.position.y - groundY).F("ent_sx", e.transform.localScale.x).End();
                    }
                }
                w.EndArray();
            }

            w.F("damage_dealt", Mathf.Min(total, total - cur)).B("won", Won).B("lost", Lost).I("parries", Parries);
            return w.End().ToString();
        }

        static int SlimeMove(SlimeLevelSlime s)
        {
            bool inAir = (bool)fInAir.GetValue(s);
            switch (s.state)
            {
                case SlimeLevelSlime.State.Intro: return INTRO;
                case SlimeLevelSlime.State.Dying: return TRANSFORM;
                case SlimeLevelSlime.State.Punch:
                    return ArmOut(s) ? PUNCH : PUNCH_WINDUP;
                default:
                    if (inAir) return JUMP;
                    if ((bool)fWantsTransform.GetValue(s) && !inAir && s.GetComponent<Animator>() != null)
                    {
                        var info = s.GetComponent<Animator>().GetCurrentAnimatorStateInfo(0);
                        if (info.IsName("Transform")) return TRANSFORM;
                    }
                    return IDLE;
            }
        }

        static string ClipName(Component c)
        {
            var an = c.GetComponent<Animator>();
            if (an == null || !an.isActiveAndEnabled) return "";
            var clips = an.GetCurrentAnimatorClipInfo(0);
            return clips.Length > 0 && clips[0].clip != null ? clips[0].clip.name : "";
        }

        static bool ArmOut(SlimeLevelSlime s)
        {
            foreach (var box in s.GetComponents<BoxCollider2D>())
                if (box.enabled && box.bounds.extents.x > 2f) return true;
            var t = s.transform.Find("ArmCollider");
            if (t == null) return false;
            var c = t.GetComponent<Collider2D>();
            return c != null && c.enabled && c.bounds.extents.x > 2f;
        }

        /// <summary>
        /// The entity's body collider, fixed by identity the first time the entity is seen: the root
        /// circle with the largest serialized radius (small: its only circle; big: the r=160 circle,
        /// not the r=80 fist circle; tomb: its only circle).
        /// </summary>
        Collider2D Body(Component e)
        {
            Collider2D c;
            int id = e.GetInstanceID();
            if (bodyOf.TryGetValue(id, out c) && c != null) return c;
            c = null;
            float best = -1f;
            e.GetComponents(circles);
            foreach (var circ in circles)
                if (circ.radius > best) { best = circ.radius; c = circ; }
            bodyOf[id] = c;
            return c;
        }

        readonly List<CircleCollider2D> circles = new List<CircleCollider2D>();
        readonly List<Collider2D> children = new List<Collider2D>();

        void Hazards(JsonWriter w, AbstractLevelEntity e, bool isTomb, float groundY, long t)
        {
            var body = Body(e);
            bool hasBody = body != null && body.enabled;
            Bounds bb = hasBody ? body.bounds : new Bounds();
            e.GetComponentsInChildren(false, children);
            foreach (var c in children)
            {
                if (!c.enabled) continue;
                var b = c.bounds;
                if (b.extents.x < 2f && b.extents.y < 2f) continue;  // idle 1x1 arm boxes
                int kind;
                bool parry = false;
                if (ReferenceEquals(c, body))
                    kind = isTomb ? TOMB : BODY;  // contact damage in every state, as in the sim
                else if (c.gameObject == e.gameObject || c.name == "ArmCollider")
                {
                    // every other damaging root collider: big Goopy's fist circle, arm boxes. The
                    // fist circle resting inside the body adds no damage area (the sim has no
                    // fist outside punch tracks), so it is only a hazard once it pokes out.
                    if (hasBody && bb.Contains(new Vector3(b.min.x, b.min.y, bb.center.z))
                        && bb.Contains(new Vector3(b.max.x, b.max.y, bb.center.z))) continue;
                    kind = FIST;
                }
                else
                {
                    // spec: the pink question marks are harmless parryable objects
                    var qm = c.GetComponent<SlimeLevelQuestionMark>();
                    kind = OTHER;
                    parry = qm != null;
                }
                Vector2 v = Vel(c.GetInstanceID() * 8 + kind, b.center, t);
                w.BeginElement().I("kind", kind)
                    .F("x", b.center.x).F("y", b.center.y - groundY).F("half_w", b.extents.x).F("half_h", b.extents.y)
                    .F("vx", v.x).F("vy", v.y).B("parryable", parry).S("name", c.name)
                    .End();
            }
        }
    }
}
