"""Read-only probe of Cuphead's Unity data files (scene objects, colliders, serialized fields).

Usage (from the repo root):
    python -m uv run --with unitypy --with TypeTreeGeneratorAPI python tools/unity_probe.py settings
    python -m uv run --with unitypy --with TypeTreeGeneratorAPI python tools/unity_probe.py scene 20 [name-filter]
    python -m uv run --with unitypy --with TypeTreeGeneratorAPI python tools/unity_probe.py mono 20 SlimeLevelSlime

The Goopy Le Grande scene is build index 20 (`scene_level_slime`). MonoBehaviours in this build
carry no typetrees, so their layouts are generated from Cuphead_Data/Managed/*.dll.
Never writes into the game folder; prints to stdout only.
"""

import os
import sys

import UnityPy

GAME = os.environ.get("CUPHEAD_DATA", r"E:\SteamLibrary\steamapps\common\Cuphead\Cuphead_Data")
UNITY_VERSION = "2017.4.9f1"
_GEN = None


def _f(name):
    return os.path.join(GAME, name)


def settings():
    env = UnityPy.load(_f("globalgamemanagers"))
    for obj in env.objects:
        if obj.type.name == "BuildSettings":
            for i, s in enumerate(obj.read_typetree().get("scenes", [])):
                print(i, s)
        elif obj.type.name == "TimeManager":
            print(obj.read_typetree())


def _load_level(index):
    global _GEN
    if _GEN is None:
        from UnityPy.helpers.TypeTreeGenerator import TypeTreeGenerator

        _GEN = TypeTreeGenerator(UNITY_VERSION)
        # Loading every DLL in Managed/ (or setting env.typetree_generator) makes the generator
        # throw; this subset is enough for the gameplay scripts we care about.
        managed = os.path.join(GAME, "Managed")
        for name in ("mscorlib.dll", "UnityEngine.CoreModule.dll", "UnityEngine.dll",
                     "UnityEngine.Physics2DModule.dll", "UnityEngine.AnimationModule.dll",
                     "Assembly-CSharp-firstpass.dll", "Assembly-CSharp.dll"):
            with open(os.path.join(managed, name), "rb") as fh:
                _GEN.load_dll(fh.read())
    env = UnityPy.load(_f("globalgamemanagers"))
    for obj in env.objects:
        if obj.type.name == "BuildSettings":
            for i, s in enumerate(obj.read_typetree().get("scenes", [])):
                print(i, s)
        elif obj.type.name == "TimeManager":
            print(obj.read_typetree())


def _load_level(index):
    global _GEN
    if _GEN is None:
        from UnityPy.helpers.TypeTreeGenerator import TypeTreeGenerator

        _GEN = TypeTreeGenerator(UNITY_VERSION)
        # Loading every DLL in Managed/ makes the generator throw; this subset is enough.
        managed = os.path.join(GAME, "Managed")
        dlls = ["mscorlib.dll", "System.dll", "System.Core.dll", "Rewired_Core.dll", "Assembly-CSharp-firstpass.dll", "Assembly-CSharp.dll"]
        dlls[4:4] = sorted(f for f in os.listdir(managed) if f.startswith("UnityEngine") and f.endswith(".dll"))
        for name in dlls:
            with open(os.path.join(managed, name), "rb") as fh:
                _GEN.load_dll(fh.read())
    env = UnityPy.load(
        _f(f"level{index}"),
        _f(f"sharedassets{index}.assets"),
        _f("globalgamemanagers.assets"),
        _f("resources.assets"),
    )
    return env


def _path(go):
    names = [go.m_Name]
    tr = go.m_Component[0].component.read() if go.m_Component else None
    while tr is not None and tr.m_Father and tr.m_Father.path_id:
        tr = tr.m_Father.read()
        names.append(tr.m_GameObject.read().m_Name)
    return "/".join(reversed(names))


def _script(obj):
    mb = obj.read(check_read=False)
    return mb, mb.m_Script.read()


def scene(index, flt=""):
    env = _load_level(index)
    for obj in env.objects:
        if obj.type.name != "GameObject":
            continue
        go = obj.read()
        p = _path(go)
        if flt and flt.lower() not in p.lower():
            continue
        comps = []
        for c in go.m_Component:
            co = c.component
            t = co.type.name
            if t == "Transform":
                tr = co.read()
                pos = (round(tr.m_LocalPosition.x, 1), round(tr.m_LocalPosition.y, 1))
                scl = (round(tr.m_LocalScale.x, 2), round(tr.m_LocalScale.y, 2))
                comps.append(f"T(pos={pos} scale={scl})")
            elif t in ("BoxCollider2D", "CircleCollider2D", "CapsuleCollider2D", "PolygonCollider2D"):
                d = co.read_typetree()
                keys = ("m_Size", "m_Offset", "m_Radius", "m_Enabled", "m_IsTrigger")
                comps.append(f"{t}{ {k: d[k] for k in keys if k in d} }")
            elif t == "MonoBehaviour":
                try:
                    comps.append("MB:" + _script(co.deref())[1].m_ClassName)
                except Exception as e:  # noqa: BLE001
                    comps.append(f"MB:?({e})")
            else:
                comps.append(t)
        print(p, "|", "; ".join(comps))


def mono(index, classname):
    """Dump serialized fields of every MonoBehaviour of the given class."""
    env = _load_level(index)
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            mb, script = _script(obj)
        except Exception:  # noqa: BLE001
            continue
        if script.m_ClassName != classname:
            continue
        print("==", _path(mb.m_GameObject.read()))
        full = (script.m_Namespace + "." if script.m_Namespace else "") + script.m_ClassName
        nodes = _GEN.get_nodes_up(script.m_AssemblyName, full)
        tree = obj.read_typetree(nodes=nodes, check_read=False)
        for k, v in tree.items():
            sv = str(v)
            print("  ", k, "=", sv if len(sv) < 300 else sv[:300] + "...")


if __name__ == "__main__":
    {"settings": settings, "scene": scene, "mono": mono}[sys.argv[1]](*sys.argv[2:])
