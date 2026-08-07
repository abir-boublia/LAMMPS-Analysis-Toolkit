#!/usr/bin/env python3
"""
Convert a LAMMPS data file to XYZ format.

The element labels are read from comments in the Masses section,
and the coordinates are read from the Atoms section.

Author:  Abir Boublia
Contact: abir.boublia@univ-lorraine.fr
ORCID:   https://orcid.org/0000-0003-1669-4951
Part of: LAMMPS-Analysis-Toolkit
License: MIT
Created: August 2026

Usage:
    python3 data2xyz.py system.data system.xyz

Requires Python 3.10 or newer.
"""

import sys
from collections import Counter


src = sys.argv[1] if len(sys.argv) > 1 else "system.data"
dst = sys.argv[2] if len(sys.argv) > 2 else "system.xyz"


with open(src, "r") as f:
    lines = f.readlines()


# ----------------------------------------------------------------------
# Read atom-type labels from the Masses section.
#
# Example:
# 1  12.011  # C
# 2  1.008   # H
#
# Produces:
# lab = {1: "C", 2: "H"}
# ----------------------------------------------------------------------

lab = {}

try:
    i = next(
        k for k, line in enumerate(lines)
        if line.strip().startswith("Masses")
    ) + 1
except StopIteration:
    sys.exit("ERROR: Could not find a 'Masses' section in the data file.")


# Skip blank lines after the Masses header
while i < len(lines) and not lines[i].strip():
    i += 1


while i < len(lines):
    s = lines[i].strip()

    if not s:
        break

    # Remove any inline comment temporarily to inspect the data columns
    data_part = s.split("#", 1)[0].strip()
    p = data_part.split()

    if not p or not p[0].isdigit():
        break

    if "#" not in s:
        sys.exit(
            f"ERROR: Masses line has no '# label' comment:\n{s}"
        )

    atom_type = int(p[0])
    label = s.split("#", 1)[1].strip().split()[0]

    lab[atom_type] = label

    i += 1


# ----------------------------------------------------------------------
# Read atoms.
#
# This assumes LAMMPS "full" atom style:
#
# atom-ID molecule-ID atom-type charge x y z
#
# Example:
# 1  1  2  -0.2  1.0  2.0  3.0
# ----------------------------------------------------------------------

try:
    i = next(
        k for k, line in enumerate(lines)
        if line.strip().startswith("Atoms")
    ) + 1
except StopIteration:
    sys.exit("ERROR: Could not find an 'Atoms' section in the data file.")


# Skip blank lines after the Atoms header
while i < len(lines) and not lines[i].strip():
    i += 1


out = []

while i < len(lines):
    s = lines[i].strip()

    if not s:
        break

    # Ignore inline comments
    data_part = s.split("#", 1)[0].strip()
    p = data_part.split()

    if not p or not p[0].isdigit():
        break

    if len(p) < 7:
        sys.exit(
            f"ERROR: Unexpected Atoms line:\n{s}"
        )

    atom_id = int(p[0])
    atom_type = int(p[2])

    if atom_type not in lab:
        sys.exit(
            f"ERROR: No label found for atom type {atom_type}"
        )

    x = float(p[4])
    y = float(p[5])
    z = float(p[6])

    out.append(
        (
            atom_id,
            lab[atom_type],
            x,
            y,
            z,
        )
    )

    i += 1


# Keep atoms in atom-ID order
out.sort(key=lambda atom: atom[0])


# ----------------------------------------------------------------------
# Write XYZ file
# ----------------------------------------------------------------------

with open(dst, "w") as f:
    f.write(f"{len(out)}\n")
    f.write(f"Converted from {src}\n")

    for _, element, x, y, z in out:
        f.write(
            f"{element} "
            f"{x:.6f} "
            f"{y:.6f} "
            f"{z:.6f}\n"
        )


counts = Counter(element for _, element, *_ in out)

print(
    f"Wrote {dst}: "
    f"{len(out)} atoms   "
    f"{dict(counts)}"
)
