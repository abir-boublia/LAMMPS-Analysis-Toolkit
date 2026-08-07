#!/usr/bin/env python3
"""
Write the Packmol input xyz straight out of the LAMMPS data file, so the
structure Packmol packs is byte-identical to the one the TCL builder verified.

Needed because `moltemplate.sh -xyz full.xyz` overrides every coordinate in the
.lt files with full.xyz. If system_final.xyz is stale, the fixed geometry in
system_final.lt is silently discarded at the final step.

Usage:  python3 data2xyz.py system_final.data system_final.xyz
"""
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "system.data"
dst = sys.argv[2] if len(sys.argv) > 2 else "system.xyz"

lines = open(src).readlines()

# type -> label, taken from the Masses comments so it can't drift
lab = {}
i = next(k for k, l in enumerate(lines) if l.strip().startswith("Masses")) + 2
while i < len(lines):
    s = lines[i].strip()
    if not s:
        i += 1
        if i < len(lines) and lines[i].strip() and not lines[i].strip()[0].isdigit():
            break
        continue
    p = s.split()
    if not p[0].isdigit():
        break
    if "#" not in s:
        sys.exit(f"ERROR: Masses line has no '# label' comment: {s}")
    lab[int(p[0])] = s.split("#")[1].strip()
    i += 1

i = next(k for k, l in enumerate(lines) if l.strip().startswith("Atoms")) + 2
out = []
while i < len(lines):
    s = lines[i].strip()
    if not s:
        i += 1
        if i < len(lines) and lines[i].strip() and not lines[i].strip()[0].isdigit():
            break
        continue
    p = s.split()
    if len(p) < 7:
        break
    out.append((int(p[0]), lab[int(p[2])], float(p[4]), float(p[5]), float(p[6])))
    i += 1

out.sort()                                  # atom-ID order, same as the data file
with open(dst, "w") as f:
    f.write(f"{len(out)}\n")
    f.write(f" from {src}\n")
    for _, e, x, y, z in out:
        f.write(f"{e} {x:.6f} {y:.6f} {z:.6f}\n")

from collections import Counter
print(f"wrote {dst}: {len(out)} atoms   {dict(Counter(e for _, e, *_ in out))}")



