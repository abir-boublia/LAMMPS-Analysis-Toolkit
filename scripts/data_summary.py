#!/usr/bin/env python3
"""
data_summary.py

Says what is in a LAMMPS data file, in plain language.

Opening a data file tells you very little. The header counts atoms and types,
and the rest is millions of numbers. This answers the questions you actually
have: what is this made of, what are the molecules, where does the solid stop
and the liquid start, and is anything obviously wrong with it.

Atom types are named from a label comment on the Masses lines, from an Atom
Type Labels section, or by matching the mass against the elements.

Author:  Abir Boublia
Contact: abir.boublia@univ-lorraine.fr
ORCID:   https://orcid.org/0000-0003-1669-4951
Part of: LAMMPS-Analysis-Toolkit
License: MIT

Usage:
    python data_summary.py system.lmp
    python data_summary.py system.lmp --detail
    python data_summary.py system.lmp --atom-style atomic
    python data_summary.py --help

Uses only the standard library. Requires Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

__author__ = "Abir Boublia"
__license__ = "MIT"
__version__ = "3.0.0"

# Column layout of the Atoms section per atom style:
# (type, charge, molecule, first coordinate). None means the style lacks it.
ATOM_STYLES = {
    "full":      (2, 3, 1, 4),
    "charge":    (1, 2, None, 3),
    "atomic":    (1, None, None, 2),
    "molecular": (2, None, 1, 3),
    "bond":      (2, None, 1, 3),
    "angle":     (2, None, 1, 3),
    "sphere":    (1, None, None, 4),
}

# Standard atomic weights, for naming a type from its mass.
ELEMENT_MASSES = {
    "H": 1.008, "D": 2.014, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
    "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76,
    "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33,
    "La": 138.91, "Ce": 140.12, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Th": 232.04,
    "U": 238.03,
}

# How close a mass must sit to a standard weight before the element is named.
# Tight enough to keep Ni (58.693) apart from Co (58.933).
MASS_TOLERANCE = 0.06

# Anything bigger than this is a framework, not a molecule, so a slab of a
# hundred thousand atoms is never written out as a chemical formula.
LARGEST_MOLECULE = 30

# Width of the z occupancy bars, in characters.
BAR_WIDTH = 44

HEADER_COUNTS = ("atoms", "bonds", "angles", "dihedrals", "impropers")
HEADER_TYPES = ("atom types", "bond types", "angle types",
                "dihedral types", "improper types")

SECTION_TITLES = (
    "Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals",
    "Impropers", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs",
    "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",
    "Atom Type Labels", "Bond Type Labels", "Angle Type Labels",
)

CHARGE_TOLERANCE = 1e-4
AXES = ("x", "y", "z")


def element_for(mass: float) -> str:
    """Element whose standard weight is closest to ``mass``.

    Empty when nothing is close enough, which is the right answer for a
    coarse-grained or united-atom type whose mass is a sum of several atoms.
    """
    best, distance = "", MASS_TOLERANCE
    for symbol, reference in ELEMENT_MASSES.items():
        gap = abs(mass - reference)
        if gap < distance:
            best, distance = symbol, gap
    return best


def hill_formula(symbols: list[str]) -> str:
    """Chemical formula in Hill order: carbon, hydrogen, then alphabetical."""
    counts = Counter(symbols)
    ordered: list[str] = []
    if "C" in counts:
        ordered.append("C")
        if "H" in counts:
            ordered.append("H")
    ordered += sorted(s for s in counts if s not in ordered)
    return "".join(s if counts[s] == 1 else f"{s}{counts[s]}" for s in ordered)


@dataclass
class TypeInfo:
    """What is known about one atom type."""

    number: int
    count: int = 0
    mass: float | None = None
    label: str = ""
    charge_total: float = 0.0
    z_min: float = float("inf")
    z_max: float = float("-inf")

    @property
    def element(self) -> str:
        return element_for(self.mass) if self.mass else ""

    @property
    def name(self) -> str:
        """Best available name: the label, else the element, else the number."""
        return self.label or self.element or f"type {self.number}"

    @property
    def charge_each(self) -> float:
        return self.charge_total / self.count if self.count else 0.0

    @property
    def located(self) -> bool:
        return self.z_min <= self.z_max


@dataclass
class Summary:
    """Everything read from the data file."""

    path: Path
    counts: dict[str, int] = field(default_factory=dict)
    types: dict[str, int] = field(default_factory=dict)
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    tilt: tuple[float, float, float] | None = None
    per_type: dict[int, TypeInfo] = field(default_factory=dict)
    molecule_types: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    oversized: set[int] = field(default_factory=set)
    framework_atoms: int = 0
    extent: dict[str, tuple[float, float]] = field(default_factory=dict)
    total_charge: float = 0.0
    has_charges: bool = False
    sections: list[str] = field(default_factory=list)
    atom_lines_read: int = 0

    def info(self, number: int) -> TypeInfo:
        return self.per_type.setdefault(number, TypeInfo(number))

    @property
    def lengths(self) -> dict[str, float]:
        return {axis: hi - lo for axis, (lo, hi) in self.box.items()}

    @property
    def surface_area(self) -> float:
        """Area of the xy face, the one a slab presents."""
        lengths = self.lengths
        return lengths.get("x", 0.0) * lengths.get("y", 0.0)

    def elements(self) -> dict[str, int]:
        """Atom count per element across the whole system."""
        totals: dict[str, int] = defaultdict(int)
        for info in self.per_type.values():
            totals[info.element or info.name] += info.count
        return dict(totals)

    def molecule_formulas(self) -> Counter:
        """How many molecules of each formula, for small molecules only."""
        formulas: Counter = Counter()
        for types in self.molecule_types.values():
            symbols = [self.per_type[t].element or self.per_type[t].name for t in types]
            formulas[hill_formula(symbols)] += 1
        return formulas


def parse(path: Path, atom_style: str) -> Summary:
    """Read a LAMMPS data file in a single streaming pass."""
    if atom_style not in ATOM_STYLES:
        raise ValueError(
            f"unknown atom_style {atom_style!r}. Known: {', '.join(sorted(ATOM_STYLES))}"
        )
    type_column, charge_column, molecule_column, coordinate_column = ATOM_STYLES[atom_style]

    summary = Summary(path=path)
    section = None

    with open(path, errors="replace") as handle:
        for raw_line in handle:
            body, _, comment = raw_line.partition("#")
            line = body.strip()
            comment = comment.strip()
            if not line:
                continue

            matched = next((t for t in SECTION_TITLES if line == t), None)
            if matched:
                section = matched
                summary.sections.append(matched)
                continue

            fields = line.split()

            if section is None:
                label = " ".join(fields[1:]) if len(fields) > 1 else ""
                if label in HEADER_COUNTS or label in HEADER_TYPES:
                    target = summary.counts if label in HEADER_COUNTS else summary.types
                    target[label] = int(float(fields[0]))
                elif len(fields) == 4 and fields[2].endswith("lo"):
                    summary.box[fields[2][0]] = (float(fields[0]), float(fields[1]))
                elif len(fields) == 6 and fields[3] == "xy":
                    summary.tilt = tuple(float(v) for v in fields[:3])
                continue

            if section == "Masses" and len(fields) >= 2:
                try:
                    info = summary.info(int(fields[0]))
                    info.mass = float(fields[1])
                except ValueError:
                    continue
                # ClayFF and similar force fields name the type in a trailing
                # comment, which says more than the element alone.
                if comment:
                    info.label = comment.split()[0]

            elif section == "Atom Type Labels" and len(fields) >= 2:
                try:
                    summary.info(int(fields[0])).label = fields[1]
                except ValueError:
                    continue

            elif section == "Atoms":
                if len(fields) <= type_column:
                    continue
                try:
                    atom_type = int(fields[type_column])
                except ValueError:
                    continue
                info = summary.info(atom_type)
                info.count += 1
                summary.atom_lines_read += 1

                if molecule_column is not None and len(fields) > molecule_column:
                    try:
                        molecule = int(fields[molecule_column])
                    except ValueError:
                        molecule = None
                    if molecule is not None:
                        if molecule in summary.oversized:
                            summary.framework_atoms += 1
                        else:
                            members = summary.molecule_types[molecule]
                            members.append(atom_type)
                            # Drop the member list once it is clearly a
                            # framework, rather than carrying a hundred
                            # thousand entries for a slab.
                            if len(members) > LARGEST_MOLECULE:
                                summary.oversized.add(molecule)
                                summary.framework_atoms += len(members)
                                del summary.molecule_types[molecule]

                if charge_column is not None and len(fields) > charge_column:
                    try:
                        charge = float(fields[charge_column])
                    except ValueError:
                        charge = None
                    if charge is not None:
                        summary.has_charges = True
                        summary.total_charge += charge
                        info.charge_total += charge

                if len(fields) >= coordinate_column + 3:
                    try:
                        position = [float(v) for v in
                                    fields[coordinate_column:coordinate_column + 3]]
                    except ValueError:
                        continue
                    info.z_min = min(info.z_min, position[2])
                    info.z_max = max(info.z_max, position[2])
                    for axis, value in zip(AXES, position, strict=True):
                        low, high = summary.extent.get(axis, (value, value))
                        summary.extent[axis] = (min(low, value), max(high, value))

    return summary


def occupancy_chart(summary: Summary) -> list[str]:
    """A bar per atom type showing where it sits along z.

    A picture answers "how thick is the slab and how much water is above it"
    faster than any table, and it makes a vacuum gap or a misplaced layer
    obvious at a glance.
    """
    if "z" not in summary.box:
        return []
    located = [i for i in summary.per_type.values() if i.count and i.located]
    if not located:
        return []

    low, high = summary.box["z"]
    height = high - low
    if height <= 0:
        return []

    width = max(len(i.name) for i in located)
    lines = [f"  Where things sit along z   (the box is {height:.1f} A tall)", ""]

    for info in sorted(located, key=lambda i: (i.z_min, i.z_max)):
        start = round((info.z_min - low) / height * BAR_WIDTH)
        end = round((info.z_max - low) / height * BAR_WIDTH)
        start = max(0, min(start, BAR_WIDTH - 1))
        end = max(start + 1, min(end, BAR_WIDTH))
        bar = "." * start + "#" * (end - start) + "." * (BAR_WIDTH - end)
        thickness = info.z_max - info.z_min
        lines.append(
            f"    {info.name:<{width}}  |{bar}|  "
            f"{info.z_min:6.1f} to {info.z_max:6.1f}   ({thickness:.1f} A thick)"
        )

    # Align the axis labels with the ends of the bars above them.
    indent = 4 + width + 3
    low_text, high_text = f"{low:.0f}", f"{high:.0f} A"
    padding = max(1, BAR_WIDTH - len(low_text) - len(high_text) + 2)
    lines.append(" " * indent + low_text + " " * padding + high_text)
    return lines


def plain_summary(summary: Summary) -> list[str]:
    """The short, human-readable description."""
    lines = []
    lengths = summary.lengths
    atoms = summary.atom_lines_read or summary.counts.get("atoms", 0)

    if len(lengths) == 3:
        shape = "triclinic" if summary.tilt and any(summary.tilt) else "orthogonal"
        lines.append(
            f"  A {lengths['x']:.1f} x {lengths['y']:.1f} x {lengths['z']:.1f} A "
            f"{shape} box holding {atoms:,} atoms."
        )
    else:
        lines.append(f"  {atoms:,} atoms.")

    elements = summary.elements()
    listing = ", ".join(f"{count:,} {symbol}" for symbol, count
                        in sorted(elements.items(), key=lambda kv: -kv[1]))
    lines += ["", "  Made of", f"    {listing}"]

    formulas = summary.molecule_formulas()
    framework_count = len(summary.oversized)
    if formulas or framework_count:
        lines += ["", "  Molecules"]
        for formula, number in formulas.most_common(6):
            lines.append(f"    {number:>7,} x {formula}")
        if len(formulas) > 6:
            lines.append(f"            and {len(formulas) - 6} other kinds")
        if framework_count:
            noun = "framework" if framework_count == 1 else "frameworks"
            lines.append(
                f"    {framework_count:>7,} connected {noun} of "
                f"{summary.framework_atoms:,} atoms (a slab, surface or polymer)"
            )

    lines.append("")
    bonded = ", ".join(f"{summary.counts[k]:,} {k}"
                       for k in ("bonds", "angles", "dihedrals", "impropers")
                       if summary.counts.get(k))
    if bonded:
        lines.append(f"  Bonded terms   {bonded}")
    if summary.surface_area:
        lines.append(f"  Surface area   {summary.surface_area:,.0f} A^2 in xy")
    if summary.has_charges:
        state = ("neutral" if abs(summary.total_charge) <= CHARGE_TOLERANCE
                 else f"NOT NEUTRAL, {summary.total_charge:+.4f}")
        lines.append(f"  Total charge   {state}")

    return lines


def detail_table(summary: Summary) -> list[str]:
    """The per-type table, for when the short view is not enough."""
    if not summary.per_type:
        return []
    header = f"  {'type':>4}  {'label':<8}{'element':<9}{'count':>9}{'mass':>10}"
    if summary.has_charges:
        header += f"{'charge':>10}{'total q':>13}"
    lines = ["", header, "  " + "-" * (len(header) - 2)]

    for number in sorted(summary.per_type):
        info = summary.per_type[number]
        row = (f"  {number:>4}  {info.label or '-':<8}{info.element or '-':<9}"
               f"{info.count:>9,}"
               + (f"{info.mass:>10.4f}" if info.mass else f"{'-':>10}"))
        if summary.has_charges:
            row += f"{info.charge_each:>10.4f}{info.charge_total:>13.2f}"
        lines.append(row)
    return lines


def problems(summary: Summary, atom_style: str = "full") -> list[str]:
    """Anything needing a decision before this file is used."""
    found = []

    declared = summary.counts.get("atoms")
    if declared and not summary.atom_lines_read:
        # Almost always the wrong atom style: the type column then holds a
        # coordinate, every line fails to parse, and nothing is counted.
        message = (
            f"the header declares {declared:,} atoms but none could be read with "
            f"--atom-style {atom_style}. Check which style this file was written "
            f"with; the choices are {', '.join(sorted(ATOM_STYLES))}."
        )
        return [message]

    if declared and summary.atom_lines_read != declared:
        found.append(
            f"the header declares {declared:,} atoms but the Atoms section has "
            f"{summary.atom_lines_read:,} lines"
        )

    n_types = summary.types.get("atom types", 0)
    if n_types:
        unused = [t for t in range(1, n_types + 1)
                  if t not in summary.per_type or summary.per_type[t].count == 0]
        if unused:
            found.append(
                f"atom type{'s' if len(unused) > 1 else ''} "
                f"{', '.join(map(str, unused))} declared but never used"
            )
        beyond = [t for t, i in summary.per_type.items() if t > n_types and i.count]
        if beyond:
            found.append(f"atoms use type {max(beyond)}, beyond the {n_types} declared")
        if "Masses" in summary.sections:
            missing = [t for t in range(1, n_types + 1)
                       if t not in summary.per_type or summary.per_type[t].mass is None]
            if missing:
                found.append(
                    f"no mass for type{'s' if len(missing) > 1 else ''} "
                    f"{', '.join(map(str, missing))}"
                )

    for axis in AXES:
        if axis in summary.extent and axis in summary.box:
            low, high = summary.extent[axis]
            box_low, box_high = summary.box[axis]
            if low < box_low - 1e-6 or high > box_high + 1e-6:
                found.append(
                    f"atoms lie outside the box along {axis}: {low:.3f} to "
                    f"{high:.3f} against bounds {box_low:.3f} to {box_high:.3f}"
                )

    if summary.has_charges and abs(summary.total_charge) > CHARGE_TOLERANCE:
        found.append(
            f"total charge is {summary.total_charge:+.6f}, not zero. A long-range "
            "Coulomb solver applies a neutralising background instead of stopping, "
            "so the run continues and the energies are wrong."
        )

    if not summary.per_type:
        found.append("no Atoms section was found")

    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Say what is in a LAMMPS data file.",
        epilog="Part of LAMMPS-Analysis-Toolkit by Abir Boublia.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="LAMMPS data file (.data, .lmp, any name)")
    parser.add_argument("--atom-style", default="full", choices=sorted(ATOM_STYLES),
                        help="column layout of the Atoms section")
    parser.add_argument("--detail", action="store_true",
                        help="also print the per-type table")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.data)

    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    try:
        summary = parse(path, args.atom_style)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n{path}\n")
    print("\n".join(plain_summary(summary)))

    chart = occupancy_chart(summary)
    if chart:
        print()
        print("\n".join(chart))

    if args.detail:
        print("\n".join(detail_table(summary)))

    found = problems(summary, args.atom_style)
    if found:
        print(f"\n  Worth checking ({len(found)})")
        for item in found:
            print(f"    - {item}")
    print()

    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())

