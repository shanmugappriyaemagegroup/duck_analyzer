#!/usr/bin/env python3
"""
apply_debug_patch.py -- adds debug logging to duck_analyzer.py in place.

Run this ONCE in the same folder as your duck_analyzer.py:
    python apply_debug_patch.py

It makes a backup (duck_analyzer.py.bak) then inserts three debug hooks and
is safe to run again (it detects if already patched). Turn logging on by
adding  debug: true  to config.yaml.
"""
import re, shutil, sys, os

SRC = "duck_analyzer.py"
if not os.path.exists(SRC):
    sys.exit("duck_analyzer.py not found in this folder.")

s = open(SRC, encoding="utf-8").read()
if "self.debug = bool(cfg.get(" in s:
    print("Already patched. Nothing to do.")
    sys.exit(0)

shutil.copyfile(SRC, SRC + ".bak")
print("Backup written:", SRC + ".bak")

# --- Edit 1: read the debug flag (after crowd_dist_frac line) ---
anchor1 = 'self.crowd_dist_frac = float(cfg.get("crowd_dist_frac", 0.06))'
add1 = anchor1 + '\n        self.debug = bool(cfg.get("debug", False))'
if anchor1 in s:
    s = s.replace(anchor1, add1, 1)
else:
    print("WARN: could not find crowd_dist_frac line; add self.debug manually.")

# --- Edit 2: per-rebind log (after best_app assignment inside the accept loop) ---
anchor2 = ("            best_did = assigned_slot.get(pi)\n"
           "            best_score = assigned_score.get(pi, -1.0)\n"
           "            best_app = assigned_app.get(pi, 0.0)\n")
add2 = anchor2 + (
"            if self.debug:\n"
"                _crowded = False\n"
"                if best_did is not None and best_did in slot_centers:\n"
"                    _lc = slot_centers[best_did]\n"
"                    for _d2, _c2 in slot_centers.items():\n"
"                        if _d2 != best_did and np.linalg.norm(_lc - _c2) < crowd_dist:\n"
"                            _crowded = True\n"
"                            break\n"
"                print(f\"[DBG f{self.frame_idx}] pending tid={tid} -> slot #{best_did} \"\n"
"                      f\"score={best_score:.3f} app={best_app:.3f} crowded={_crowded} \"\n"
"                      f\"missing_active={best_did in self.missing_active}\")\n")
if anchor2 in s:
    s = s.replace(anchor2, add2, 1)
else:
    print("WARN: could not find the rebind-accept loop; add Edit 2 manually.")

# --- Edit 3: frame summary before the final return self._finish( ---
anchor3 = ("        return self._finish(\n"
           "            annotated, status=status, detected=detected, fps=fps,")
add3 = (
"        if self.debug:\n"
"            print(f\"[DBG f{self.frame_idx}] status={status} detected={detected} \"\n"
"                  f\"present={sorted(present_display.keys())} pending={len(pend_meta)} \"\n"
"                  f\"unbound={len(unbound)} missing={missing_ids} reasons={reasons}\")\n"
+ anchor3)
if anchor3 in s:
    s = s.replace(anchor3, add3, 1)
else:
    print("WARN: could not find the final _finish call; add Edit 3 manually.")

open(SRC, "w", encoding="utf-8").write(s)
print("Patched", SRC)
print("Now add  debug: true  to config.yaml and run a clip.")