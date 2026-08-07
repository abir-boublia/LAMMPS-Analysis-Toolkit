#!/usr/bin/env python3
"""
data_summary.py

Says what is in a LAMMPS data file.

Prints the box, what it is made of, the total charge, and where each species
sits along z. The charge line matters most: a system that is not neutral
still runs under a long-range Coulomb solver, quietly, with a neutralising
background and wrong energies throughout.

Species are named from a label comment on the Masses lines, from an Atom Type
Labels section, or by matching the mass against the elements.

Author:  Abir Boublia
Contact: abir.boublia@univ-lorraine.fr
ORCID:   https://orcid.org/0000-0003-1669-4951
Part of: LAMMPS-Analysis-Toolkit
License: MIT

Usage:
    python data_summary.py system.lmp
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
__version__ = "4.0.0"

# Column layout of the Atoms section per atom style:
# (type, charge, molecule, first coordinate). None means the style lacks it.
ATOM_STYLES = {
    "full":      (2, 3, 1, 4),
    "charge":    (1, 2, None, 3),
    "atomic":    (1, None, None, 2),
    "molecular": (2, None, 1, 3),
    "bond":      (2, None, 1, 3),
    "angle":     (2, None, 1, 3),
}

# Above this size a molecule is treated as a framework and reported by
# composition rather than by formula. Set well above a surfactant or collector
# (tens of atoms) and well below a slab (thousands).
LARGEST_MOLECULE = 200

# Standard atomic weights, for naming a species from its mass.
ELEMENT_MASSES = {
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

# How close a mass must sit to a standard weight before the element is named.
# Tight enough to keep Ni (58.693) apart from Co (58.933).
MASS_TOLERANCE = 0.06

CHARGE_TOLERANCE = 1e-4

# Atoms a little outside the box are normal: LAMMPS wraps them on read under
# periodic boundaries, and a coordinate written at the boundary often lands a
# hair beyond it. Only a real excursion is worth mentioning.
OUTSIDE_BOX_TOLERANCE = 1.0

AXES = ("x", "y", "z")


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


def element_for(mass: float) -> str:
    """Element whose standard weight is closest to ``mass``.

    Empty when nothing is close enough, which is the right answer for a
    coarse-grained type whose mass is the sum of several atoms.
    """
    best, distance = "", MASS_TOLERANCE
    for symbol, reference in ELEMENT_MASSES.items():
        gap = abs(mass - reference)
        if gap < distance:
            best, distance = symbol, gap
    return best


@dataclass
class Species:
    """One atom type."""

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
        """Label if the file gave one, else the element, else the type number."""
        return self.label or self.element or str(self.number)


@dataclass
class Group:
    """One kind of molecule, or the framework, aggregated."""

    formula: str
    molecules: int = 0
    atoms: int = 0
    z_min: float = float("inf")
    z_max: float = float("-inf")
    composition: Counter = field(default_factory=Counter)

    def absorb(self, atom_count: int, z_low: float, z_high: float) -> None:
        self.molecules += 1
        self.atoms += atom_count
        self.z_min = min(self.z_min, z_low)
        self.z_max = max(self.z_max, z_high)


@dataclass
class Summary:
    """Everything read from the data file."""

    path: Path
    declared_atoms: int = 0
    declared_types: int = 0
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    triclinic: bool = False
    species: dict[int, Species] = field(default_factory=dict)
    extent: dict[str, tuple[float, float]] = field(default_factory=dict)
    total_charge: float = 0.0
    has_charges: bool = False
    atoms_read: int = 0
    has_molecules: bool = False
    molecule_types: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    molecule_z: dict[int, list[float]] = field(default_factory=dict)
    # Frameworks keep a type histogram rather than a member list, so a slab of
    # a hundred thousand atoms costs a handful of integers.
    framework_types: dict[int, Counter] = field(default_factory=dict)

    def get(self, number: int) -> Species:
        return self.species.setdefault(number, Species(number))

    @property
    def lengths(self) -> dict[str, float]:
        return {axis: hi - lo for axis, (lo, hi) in self.box.items()}

    def groups(self) -> list[Group]:
        """Molecules aggregated by formula, frameworks pooled separately.

        Grouping by molecule ID says what the system is made of rather than
        which atom types it uses: one line for the water, one for the ions,
        one for the collector, one for the slab.
        """
        by_formula: dict[str, Group] = {}
        framework = Group(formula="framework")

        for molecule, types in self.molecule_types.items():
            z_low, z_high = self.molecule_z.get(molecule, (0.0, 0.0))
            symbols = [self.species[t].element or self.species[t].name for t in types]
            group = by_formula.setdefault(hill_formula(symbols), Group(hill_formula(symbols)))
            group.absorb(len(types), z_low, z_high)

        for molecule, histogram in self.framework_types.items():
            z_low, z_high = self.molecule_z.get(molecule, (0.0, 0.0))
            framework.absorb(sum(histogram.values()), z_low, z_high)
            for atom_type, number in histogram.items():
                symbol = self.species[atom_type].element or self.species[atom_type].name
                framework.composition[symbol] += number

        ordered = sorted(by_formula.values(), key=lambda g: -g.atoms)
        if framework.molecules:
            ordered.append(framework)
        return ordered

    def composition(self) -> dict[str, int]:
        """Atom count per element, merging types that share one."""
        totals: dict[str, int] = defaultdict(int)
        for info in self.species.values():
            totals[info.element or info.name] += info.count
        return dict(totals)


def parse(path: Path, atom_style: str) -> Summary:
    """Read the file in a single streaming pass."""
    if atom_style not in ATOM_STYLES:
        raise ValueError(
            f"unknown atom_style {atom_style!r}. Known: {', '.join(sorted(ATOM_STYLES))}"
        )
    type_column, charge_column, molecule_column, coordinate_column = ATOM_STYLES[atom_style]

    summary = Summary(path=path)
    section = None
    seen_title = False

    with open(path, errors="replace") as handle:
        for raw_line in handle:
            body, _, comment = raw_line.partition("#")
            line = body.strip()
            if not line:
                continue

            # The first non-blank line is a free-form title, not a section.
            if not seen_title:
                seen_title = True
                continue

            if line[0].isalpha():
                section = line
                continue

            fields = line.split()

            if section is None:
                if len(fields) == 2 and fields[1] == "atoms":
                    summary.declared_atoms = int(float(fields[0]))
                elif len(fields) == 3 and fields[1:] == ["atom", "types"]:
                    summary.declared_types = int(float(fields[0]))
                elif len(fields) == 4 and fields[2].endswith("lo"):
                    summary.box[fields[2][0]] = (float(fields[0]), float(fields[1]))
                elif len(fields) == 6 and fields[3] == "xy":
                    summary.triclinic = any(float(v) for v in fields[:3])
                continue

            if section == "Masses" and len(fields) >= 2:
                try:
                    info = summary.get(int(fields[0]))
                    info.mass = float(fields[1])
                except ValueError:
                    continue
                # ClayFF and similar force fields name the type in a trailing
                # comment, which says more than the element alone.
                if comment.strip():
                    info.label = comment.split()[0]

            elif section.startswith("Atom Type Labels") and len(fields) >= 2:
                try:
                    summary.get(int(fields[0])).label = fields[1]
                except ValueError:
                    continue

            elif section.startswith("Atoms") and len(fields) > type_column:
                try:
                    info = summary.get(int(fields[type_column]))
                except ValueError:
                    continue
                info.count += 1
                summary.atoms_read += 1

                molecule = None
                if molecule_column is not None and len(fields) > molecule_column:
                    try:
                        molecule = int(fields[molecule_column])
                    except ValueError:
                        molecule = None
                if molecule is not None:
                    summary.has_molecules = True
                    atom_type = int(fields[type_column])
                    if molecule in summary.framework_types:
                        summary.framework_types[molecule][atom_type] += 1
                    else:
                        members = summary.molecule_types[molecule]
                        members.append(atom_type)
                        if len(members) > LARGEST_MOLECULE:
                            summary.framework_types[molecule] = Counter(members)
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
                    if molecule is not None:
                        span = summary.molecule_z.get(molecule)
                        if span is None:
                            summary.molecule_z[molecule] = [position[2], position[2]]
                        else:
                            span[0] = min(span[0], position[2])
                            span[1] = max(span[1], position[2])
                    for axis, value in zip(AXES, position, strict=True):
                        low, high = summary.extent.get(axis, (value, value))
                        summary.extent[axis] = (min(low, value), max(high, value))

    return summary


def section_title(title: str) -> list[str]:
    return [title, "-" * len(title)]


def report(summary: Summary) -> list[str]:
    """One header block and one table."""
    lines: list[str] = []
    lengths = summary.lengths

    facts = []
    if len(lengths) == 3:
        facts.append(
            f"{lengths['x']:.1f} \u00d7 {lengths['y']:.1f} \u00d7 {lengths['z']:.1f} "
            f"\u00c5 {'triclinic' if summary.triclinic else 'orthogonal'}"
        )
        facts.append(f"xy area {lengths['x'] * lengths['y']:,.0f} \u00c5\u00b2")
    facts.append(f"{summary.atoms_read:,} atoms")
    if summary.has_charges:
        neutral = abs(summary.total_charge) <= CHARGE_TOLERANCE
        value = f"{0.0:.3f}" if neutral else f"{summary.total_charge:+.3f}"
        facts.append(f"charge {value} e ({'neutral' if neutral else 'NOT NEUTRAL'})")
    lines += ["   ".join(facts), ""]

    groups = summary.groups() if summary.has_molecules else []
    if groups:
        lines += section_title("Contents")
        lines.append(f"{'':<16}{'count':>8}{'atoms':>10}{'z range':>21}")
        for group in groups:
            span = (f"{group.z_min:.1f} to {group.z_max:.1f} \u00c5"
                    if group.z_min <= group.z_max else "")
            lines.append(
                f"{group.formula:<16}{group.molecules:>8,}{group.atoms:>10,}{span:>21}"
            )
            if group.formula == "framework" and group.composition:
                parts = " ".join(f"{s}{n:,}" for s, n
                                 in sorted(group.composition.items(), key=lambda kv: -kv[1]))
                lines.append(f"{'':<16}{parts}")
        lines.append("")
    else:
        # No molecule column in this atom style: fall back to atom types.
        located = [s for s in summary.species.values() if s.count and s.z_min <= s.z_max]
        if located:
            lines += section_title("Contents")
            lines.append(f"{'Species':<16}{'atoms':>10}{'z range':>21}")
            for info in sorted(located, key=lambda s: -s.count):
                lines.append(
                    f"{info.name:<16}{info.count:>10,}"
                    f"{f'{info.z_min:.1f} to {info.z_max:.1f} \u00c5':>21}"
                )
            lines.append("")

    return lines


def problems(summary: Summary, atom_style: str) -> list[str]:
    """Only the things that would spoil a run."""
    found = []

    if summary.declared_atoms and not summary.atoms_read:
        # Almost always the wrong atom style: the type column then holds a
        # coordinate, every line fails to parse, and nothing is counted.
        message = (
            f"the header declares {summary.declared_atoms:,} atoms but none could "
            f"be read with --atom-style {atom_style}. Try another of: "
            f"{', '.join(sorted(ATOM_STYLES))}"
        )
        return [message]

    if summary.declared_atoms and summary.atoms_read != summary.declared_atoms:
        found.append(
            f"the header declares {summary.declared_atoms:,} atoms but the Atoms "
            f"section has {summary.atoms_read:,} lines"
        )

    if summary.declared_types:
        unused = [t for t in range(1, summary.declared_types + 1)
                  if t not in summary.species or summary.species[t].count == 0]
        if unused:
            found.append(
                f"atom type{'s' if len(unused) > 1 else ''} "
                f"{', '.join(map(str, unused))} declared but never used"
            )

    for axis in AXES:
        if axis in summary.extent and axis in summary.box:
            low, high = summary.extent[axis]
            box_low, box_high = summary.box[axis]
            excursion = max(box_low - low, high - box_high)
            if excursion > OUTSIDE_BOX_TOLERANCE:
                found.append(
                    f"atoms reach {excursion:.1f} \u00c5 outside the box along "
                    f"{axis}: {low:.2f} to {high:.2f} against bounds "
                    f"{box_low:.2f} to {box_high:.2f}"
                )

    if summary.has_charges and abs(summary.total_charge) > CHARGE_TOLERANCE:
        found.append(
            f"total charge is {summary.total_charge:+.6f}, not zero. A long-range "
            "Coulomb solver applies a neutralising background instead of stopping, "
            "so the run continues and the energies are wrong."
        )

    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Say what is in a LAMMPS data file.",
        epilog="Part of LAMMPS-Analysis-Toolkit by Abir Boublia.",
    )
    parser.add_argument("data", help="LAMMPS data file (.data, .lmp, any name)")
    parser.add_argument("--atom-style", default="full", choices=sorted(ATOM_STYLES),
                        help="column layout of the Atoms section (default: full)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

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
    print("\n".join(report(summary)).rstrip())

    found = problems(summary, args.atom_style)
    if found:
        print()
        print("\n".join(section_title(f"Worth checking ({len(found)})")))
        for item in found:
            print(f"- {item}")
    print()

    return 1 if found else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Normal when the output is piped into head, less or similar.
        sys.exit(0)

