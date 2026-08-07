#!/usr/bin/env python3
"""
Convert a LAMMPS data file to XYZ format.

Element symbols come from the mass of each atom type, so the output is
readable by ASE, VMD and anything else expecting real elements. The
force-field label from the Masses comment is kept as an extra column, and
the simulation box is written as a Lattice entry so periodic analysis works
downstream.

Author:  Abir Boublia
Contact: abir.boublia@univ-lorraine.fr
ORCID:   https://orcid.org/0000-0003-1669-4951
Part of: LAMMPS-Analysis-Toolkit
License: MIT
Created: August 2026

Usage:
    python3 data2xyz.py system.data system.xyz
    python3 data2xyz.py system.data system.xyz --style charge

Requires Python 3.10 or newer.
"""

import sys
from collections import Counter


# Standard atomic weights, used to turn an atom type into an element symbol.
ELEMENTS = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990,
    "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Ti": 47.867,
    "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
    "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Sr": 87.62,
    "Zr": 91.224, "Mo": 95.95, "Ag": 107.87, "Sn": 118.71, "I": 126.90,
    "Cs": 132.91, "Ba": 137.33, "W": 183.84, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Pb": 207.2, "U": 238.03,
}

# A mass must land this close to a standard weight to name the element.
# Tight enough to keep Ni (58.693) apart from Co (58.933).
TOLERANCE = 0.06

# Where the atom type and the first coordinate sit, per atom style.
STYLES = {
    "full":      (2, 4),
    "charge":    (1, 3),
    "atomic":    (1, 2),
    "molecular": (2, 3),
    "bond":      (2, 3),
    "angle":     (2, 3),
}


def element_of(mass):
    """Element whose standard weight is closest to mass, or None."""
    best, gap = None, TOLERANCE
    for symbol, weight in ELEMENTS.items():
        if abs(mass - weight) < gap:
            best, gap = symbol, abs(mass - weight)
    return best


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------

args = [a for a in sys.argv[1:] if not a.startswith("--")]
src = args[0] if len(args) > 0 else "system.data"
dst = args[1] if len(args) > 1 else "system.xyz"

forced_style = None
for a in sys.argv[1:]:
    if a.startswith("--style"):
        forced_style = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]

if forced_style and forced_style not in STYLES:
    sys.exit(f"ERROR: unknown atom style {forced_style!r}. "
             f"Choose from: {', '.join(sorted(STYLES))}")


with open(src, "r") as f:
    lines = f.readlines()


# ----------------------------------------------------------------------
# Read the box bounds from the header, for the Lattice entry.
#
# Example:
# 0.0 50.943 xlo xhi
# ----------------------------------------------------------------------

box = {}

for line in lines:
    p = line.split("#", 1)[0].split()
    if len(p) == 4 and p[2] in ("xlo", "ylo", "zlo"):
        box[p[2][0]] = (float(p[0]), float(p[1]))
    if line.strip().startswith("Atoms"):
        break


# ----------------------------------------------------------------------
# Read atom-type labels and masses from the Masses section.
#
# Example:
# 1  55.845  # feo
#
# Produces:
# lab = {1: "feo"}, elem = {1: "Fe"}
# ----------------------------------------------------------------------

lab = {}
elem = {}

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

    data_part = s.split("#", 1)[0].strip()
    p = data_part.split()

    if not p or not p[0].isdigit():
        break

    atom_type = int(p[0])
    mass = float(p[1])

    # The label is informative but optional; the mass is what identifies
    # the element, so a file without comments still converts.
    lab[atom_type] = s.split("#", 1)[1].strip().split()[0] if "#" in s else ""

    symbol = element_of(mass)
    if symbol is None:
        # Coarse-grained or united-atom type: no element matches its mass.
        # Fall back to the label so the atom is at least identifiable.
        symbol = lab[atom_type] or f"X{atom_type}"
        print(f"WARNING: mass {mass} of type {atom_type} matches no element; "
              f"writing {symbol!r}", file=sys.stderr)

    elem[atom_type] = symbol

    i += 1


# ----------------------------------------------------------------------
# Read atoms. The style is taken from the Atoms header comment when it is
# there, since LAMMPS writes "Atoms # full", and can be forced with --style.
# ----------------------------------------------------------------------

try:
    i = next(
        k for k, line in enumerate(lines)
        if line.strip().startswith("Atoms")
    )
except StopIteration:
    sys.exit("ERROR: Could not find an 'Atoms' section in the data file.")

header = lines[i]
i += 1

style = forced_style
if style is None:
    tail = header.split("#", 1)[1].split() if "#" in header else []
    style = tail[0] if tail and tail[0] in STYLES else "full"

type_col, xyz_col = STYLES[style]


# Skip blank lines after the Atoms header
while i < len(lines) and not lines[i].strip():
    i += 1


out = []

while i < len(lines):
    s = lines[i].strip()

    if not s:
        break

    data_part = s.split("#", 1)[0].strip()
    p = data_part.split()

    if not p or not p[0].isdigit():
        break

    if len(p) < xyz_col + 3:
        sys.exit(
            f"ERROR: Atoms line has {len(p)} columns, too few for atom style "
            f"'{style}'. Pass --style with the right one:\n{s}"
        )

    atom_id = int(p[0])
    atom_type = int(p[type_col])

    if atom_type not in elem:
        sys.exit(f"ERROR: No mass given for atom type {atom_type}")

    x = float(p[xyz_col])
    y = float(p[xyz_col + 1])
    z = float(p[xyz_col + 2])

    out.append(
        (
            atom_id,
            elem[atom_type],
            lab[atom_type] or elem[atom_type],
            x,
            y,
            z,
        )
    )

    i += 1


if not out:
    sys.exit(f"ERROR: No atoms read. Check the atom style; '{style}' was used.")


# Keep atoms in atom-ID order
out.sort(key=lambda atom: atom[0])


# ----------------------------------------------------------------------
# Write extended XYZ.
#
# The Lattice entry carries the simulation box, so OVITO and ASE apply the
# right periodic boundaries instead of treating the system as a cluster.
# ----------------------------------------------------------------------

comment = f"Converted from {src}"

if len(box) == 3:
    lx = box["x"][1] - box["x"][0]
    ly = box["y"][1] - box["y"][0]
    lz = box["z"][1] - box["z"][0]
    comment = (
        f'Lattice="{lx:.6f} 0.0 0.0 0.0 {ly:.6f} 0.0 0.0 0.0 {lz:.6f}" '
        f'Properties=species:S:1:pos:R:3:label:S:1 pbc="T T T" '
        f'origin="{src}"'
    )
else:
    print("WARNING: no box bounds found; the XYZ will have no cell",
          file=sys.stderr)


with open(dst, "w") as f:
    f.write(f"{len(out)}\n")
    f.write(f"{comment}\n")

    for _, element, label, x, y, z in out:
        f.write(
            f"{element} "
            f"{x:.6f} "
            f"{y:.6f} "
            f"{z:.6f} "
            f"{label}\n"
        )


counts = Counter(element for _, element, *_ in out)

print(
    f"Wrote {dst}: "
    f"{len(out)} atoms   "
    f"style {style}   "
    f"{dict(counts)}"
)

