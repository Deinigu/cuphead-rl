"""Fingerprint Cuphead's save slots (read-only: never writes, moves or restores a save).

    python tools/check_saves.py [--out snapshot.json]
    python tools/check_saves.py --assert-unchanged snapshot.json [--slots 0 1]

Prints sha256, size and mtime of %APPDATA%/Cuphead/cuphead_player_data_v1_slot_{0,1,2}.sav.
``--assert-unchanged`` exits 1 if any of ``--slots`` (default 0 and 1, the user's own saves;
slot 2 is the bridge's and may change) differs from the snapshot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

SAVE_DIR = Path(os.environ.get("APPDATA", "")) / "Cuphead"
SLOTS = (0, 1, 2)


def fingerprint(slot: int) -> dict:
    p = SAVE_DIR / f"cuphead_player_data_v1_slot_{slot}.sav"
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "size": st.st_size,
        "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="write the snapshot as JSON")
    ap.add_argument("--assert-unchanged", type=Path, metavar="SNAPSHOT")
    ap.add_argument("--slots", type=int, nargs="+", default=[0, 1], help="slots that must be unchanged")
    args = ap.parse_args()

    now = {str(s): fingerprint(s) for s in SLOTS}
    for s, f in now.items():
        if f["exists"]:
            print(f"slot {s}: {f['sha256']}  {f['size']:>6} B  mtime {f['mtime']}")
        else:
            print(f"slot {s}: missing")
    if args.out:
        args.out.write_text(json.dumps({"taken": dt.datetime.now().isoformat(timespec="seconds"), "slots": now}, indent=1))
        print(f"snapshot -> {args.out}")
    if args.assert_unchanged:
        before = json.loads(args.assert_unchanged.read_text())["slots"]
        bad = []
        for s in map(str, args.slots):
            b, a = before[s], now[s]
            same = b["exists"] == a["exists"] and b.get("sha256") == a.get("sha256") and b.get("mtime") == a.get("mtime")
            print(f"slot {s}: {'UNCHANGED' if same else 'CHANGED'}")
            if not same:
                bad.append(s)
        if bad:
            print(f"FAIL: slot(s) {', '.join(bad)} changed since {args.assert_unchanged}")
            return 1
        print("OK: protected slots byte-identical (sha256 and mtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
