#!/usr/bin/env python3
"""
lammps_check.py

Checks a LAMMPS input script before you submit it.

Most failed jobs die in the first seconds for reasons already visible in the
input file: a misspelled command, a data file that is not where the script
says it is, a group used before it is defined, an unfix of something that was
never fixed. Each one costs a queue wait. This reads the script, follows its
include files, and reports what it can see.

It does not replace running LAMMPS. It cannot check physics, force field
parameters, or anything depending on the contents of the data file. It catches
the clerical mistakes, which are most of them.

Author:  Abir Boublia
Contact: abir.boublia@univ-lorraine.fr
ORCID:   https://orcid.org/0000-0003-1669-4951
Part of: LAMMPS-Analysis-Toolkit
License: MIT

Usage:
    python lammps_check.py in.lammps
    python lammps_check.py in.lammps --strict
    python lammps_check.py --help

Exit codes: 0 clean, 1 errors found, 2 the script could not be read.

Requires Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass, field
from pathlib import Path

__author__ = "Abir Boublia"
__license__ = "MIT"
__version__ = "1.3.1"

# Commands LAMMPS understands. Not exhaustive across every optional package,
# but wide enough that anything missing is worth a second look.
KNOWN_COMMANDS = {
    "angle_coeff", "angle_style", "atom_modify", "atom_style", "balance",
    "bond_coeff", "bond_style", "boundary", "box", "change_box", "clear",
    "comm_modify", "comm_style", "compute", "compute_modify", "create_atoms",
    "create_bonds", "create_box", "delete_atoms", "delete_bonds", "dielectric",
    "dihedral_coeff", "dihedral_style", "dimension", "displace_atoms", "dump",
    "dump_modify", "echo", "fix", "fix_modify", "group", "group2ndx", "if",
    "improper_coeff", "improper_style", "include", "info", "jump",
    "kspace_modify", "kspace_style", "label", "lattice", "log", "mass",
    "min_modify", "min_style", "minimize", "molecule", "neb", "neigh_modify",
    "neighbor", "newton", "next", "package", "pair_coeff", "pair_modify",
    "pair_style", "pair_write", "partition", "print", "processors", "python",
    "quit", "read_data", "read_dump", "read_restart", "region", "replicate",
    "rerun", "reset_atom_ids", "reset_atoms", "reset_ids", "reset_mol_ids",
    "reset_timestep", "restart", "run", "run_style", "set", "shell",
    "special_bonds", "suffix", "temper", "thermo", "thermo_modify",
    "thermo_style", "timer", "timestep", "uncompute", "undump", "unfix",
    "units", "variable", "velocity", "write_coeff", "write_data", "write_dump",
    "write_restart",
}

# Keywords accepted by thermo_style custom. Values pulled from computes, fixes
# and variables use c_, f_ and v_ prefixes and are allowed through untouched.
THERMO_KEYWORDS = {
    "step", "elapsed", "elaplong", "dt", "time", "cpu", "tpcpu", "spcpu",
    "cpuremain", "part", "timeremain", "atoms", "temp", "press", "pe", "ke",
    "etotal", "enthalpy", "evdwl", "ecoul", "epair", "ebond", "eangle",
    "edihed", "eimp", "emol", "elong", "etail", "vol", "density", "lx", "ly",
    "lz", "xlo", "xhi", "ylo", "yhi", "zlo", "zhi", "xy", "xz", "yz", "xlat",
    "ylat", "zlat", "bonds", "angles", "dihedrals", "impropers", "pxx", "pyy",
    "pzz", "pxy", "pxz", "pyz", "fmax", "fnorm", "nbuild", "ndanger", "cella",
    "cellb", "cellc", "cellalpha", "cellbeta", "cellgamma", "ecouple",
    "econserve",
}

VALID_UNITS = {"lj", "real", "metal", "si", "cgs", "electron", "micro", "nano"}

# Commands naming a file LAMMPS must open, and which argument holds the name.
# "include" is deliberately absent: read_commands already reports missing
# includes, with the added detail that their contents went unchecked.
FILE_ARGUMENTS = {"read_data": 0, "read_restart": 0, "molecule": 1}

BUILTIN_GROUPS = {"all"}

# Header counts named in a LAMMPS data file, and the coefficient command and
# style command that each one requires.
TYPE_COUNTS = {
    "atom types": ("pair_coeff", "pair_style", "Pair Coeffs"),
    "bond types": ("bond_coeff", "bond_style", "Bond Coeffs"),
    "angle types": ("angle_coeff", "angle_style", "Angle Coeffs"),
    "dihedral types": ("dihedral_coeff", "dihedral_style", "Dihedral Coeffs"),
    "improper types": ("improper_coeff", "improper_style", "Improper Coeffs"),
}

# Section headers that can appear in a data file.
DATA_SECTIONS = (
    "Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
    "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs",
    "Dihedral Coeffs", "Improper Coeffs",
)

# Fix styles that integrate the equations of motion. Having two of these
# active over the same atoms means they are moved twice per step.
INTEGRATOR_PREFIXES = (
    "nve", "nvt", "npt", "nph", "rigid", "brownian", "langevin/spin",
    "sph", "peri", "nphug", "manifoldforce",
)

# Default timestep per unit system, and the time unit it is expressed in.
DEFAULT_TIMESTEP = {
    "real": (1.0, "fs"), "metal": (0.001, "ps"), "si": (1.0e-8, "s"),
    "cgs": (1.0e-8, "s"), "electron": (0.001, "fs"), "micro": (2.0, "us"),
    "nano": (0.00045, "ns"), "lj": (0.005, ""),
}

# How many of each time unit make one nanosecond, for readable durations.
PER_NANOSECOND = {"fs": 1.0e6, "ps": 1.0e3, "ns": 1.0, "us": 1.0e-3, "s": 1.0e-9}

# Fix styles that set the thermodynamic ensemble.
ENSEMBLE_STYLES = {
    "nve": "NVE", "nvt": "NVT", "npt": "NPT", "nph": "NPH",
    "nve/limit": "NVE", "nvt/sllod": "NVT", "nphug": "NPHug",
}

# Rough sizes for output estimates.
BYTES_PER_THERMO_LINE = 90


@dataclass
class Issue:
    """One problem found in the input."""

    level: str  # error, warning or note
    source: str
    line_number: int
    message: str
    hint: str = ""

    def render(self, root: Path) -> str:
        try:
            where = Path(self.source).relative_to(root)
        except ValueError:
            where = Path(self.source).name
        location = f"{where}:{self.line_number}" if self.line_number else str(where)
        head = f"  {self.level.upper():<9}{location:<22}{self.message}"
        return head + (f"\n{'':<11}{self.hint}" if self.hint else "")


@dataclass
class Command:
    """One logical command, after joining continuation lines."""

    name: str
    args: list[str]
    source: str
    line_number: int


@dataclass
class Stage:
    """One minimisation or run, with the conditions in force at the time."""

    number: int
    kind: str                 # "minimisation" or "dynamics"
    line_number: int
    steps: int = 0
    ensemble: str = ""
    temperature: str = ""
    pressure: str = ""
    timestep: float = 0.0
    time_unit: str = ""
    constraints: list[str] = field(default_factory=list)
    dumps: list[str] = field(default_factory=list)
    min_style: str = ""

    def duration(self) -> str:
        """Physical length of the run, in the most readable unit."""
        if not (self.steps and self.timestep and self.time_unit):
            return f"{self.steps:,} steps"
        total = self.steps * self.timestep
        per_ns = PER_NANOSECOND.get(self.time_unit)
        if per_ns is None:
            return f"{self.steps:,} steps"
        nanoseconds = total / per_ns
        if nanoseconds >= 1.0:
            return f"{nanoseconds:g} ns"
        return f"{nanoseconds * 1000:g} ps"

    def describe(self) -> str:
        if self.kind == "minimisation":
            detail = f"min_style {self.min_style}" if self.min_style else "minimisation"
            return f"  {self.number}  minimisation   {detail}"

        parts = [self.ensemble or "no integrator"]
        if self.temperature:
            parts.append(self.temperature)
        if self.pressure:
            parts.append(self.pressure)
        head = (f"  {self.number}  {parts[0]:<14} " + ", ".join(parts[1:])).rstrip()

        timing = (f"{self.timestep:g} {self.time_unit} x {self.steps:,} steps "
                  f"= {self.duration()}")
        lines = [f"{head}", f"{'':<21}{timing}"]
        if self.constraints:
            lines.append(f"{'':<21}constraints: {', '.join(self.constraints)}")
        for entry in self.dumps:
            lines.append(f"{'':<21}{entry}")
        return "\n".join(lines)


@dataclass
class DataFile:
    """Header information read from a LAMMPS data file."""

    path: Path
    counts: dict[str, int] = field(default_factory=dict)
    sections: set[str] = field(default_factory=set)

    def types(self, kind: str) -> int:
        return self.counts.get(kind, 0)


def parse_data_header(path: Path) -> DataFile:
    """Read counts and section names from a LAMMPS data file.

    Only the header and the section titles are read, not the millions of atom
    lines below them, so this stays fast on a large system.
    """
    data = DataFile(path=path)
    with open(path, errors="replace") as handle:
        for number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            for title in DATA_SECTIONS:
                if line == title or line.startswith(title + " "):
                    data.sections.add(title)
                    break

            fields = line.split()
            # Counts look like:  6 atom types   /   17373 atoms
            if len(fields) >= 2 and fields[0].lstrip("-").isdigit():
                label = " ".join(fields[1:])
                if label in TYPE_COUNTS or label in (
                    "atoms", "bonds", "angles", "dihedrals", "impropers"
                ):
                    data.counts[label] = int(fields[0])

            # The header ends at the first section title.
            if data.sections and number > 2:
                break
    return data


def expand_types(spec: str, maximum: int) -> set[int]:
    """Expand a LAMMPS type specifier into the type numbers it covers.

    Handles "*", "3", "2*", "*4" and "2*4" exactly as LAMMPS does.
    """
    if maximum < 1:
        return set()
    spec = spec.strip()
    if spec == "*":
        return set(range(1, maximum + 1))
    if "*" in spec:
        low, _, high = spec.partition("*")
        try:
            start = int(low) if low else 1
            end = int(high) if high else maximum
        except ValueError:
            return set()
        return set(range(max(start, 1), min(end, maximum) + 1))
    try:
        value = int(spec)
    except ValueError:
        return set()
    # Out-of-range values are returned too, so the caller can report them.
    return {value}


@dataclass
class State:
    """What the script has defined by this point."""

    units: str | None = None
    groups: set[str] = field(default_factory=lambda: set(BUILTIN_GROUPS))
    fixes: dict[str, Command] = field(default_factory=dict)
    computes: dict[str, Command] = field(default_factory=dict)
    dumps: dict[str, Command] = field(default_factory=dict)
    variables: set[str] = field(default_factory=set)
    regions: set[str] = field(default_factory=set)
    pair_style: str | None = None
    min_style: str = ""
    has_box: bool = False
    thermo_every: int | None = None
    data: DataFile | None = None
    timestep: float = 0.0
    stages: list[Stage] = field(default_factory=list)
    groups_by_type: list[Command] = field(default_factory=list)
    shake_fixes: list[Command] = field(default_factory=list)
    styles: dict[str, Command] = field(default_factory=dict)
    coeffs: dict[str, list[Command]] = field(default_factory=dict)
    masses: list[Command] = field(default_factory=list)


def read_commands(
    path: Path, seen: set[Path] | None = None
) -> tuple[list[Command], list[tuple[str, int, str]]]:
    """Read a script into commands, following include files.

    Comments and blank lines are dropped and continuation lines ending in "&"
    are joined, so every later check sees one command per entry however the
    script was laid out.

    Returns the commands and a list of includes that could not be opened, as
    (source file, line number, name).
    """
    seen = set() if seen is None else seen
    resolved = path.resolve()
    if resolved in seen:
        return [], []  # guard against a script including itself
    seen.add(resolved)

    commands: list[Command] = []
    missing: list[tuple[str, int, str]] = []
    pending = ""
    pending_line = 0

    with open(path, errors="replace") as handle:
        for number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if line.endswith("&"):
                if not pending:
                    pending_line = number
                pending += line[:-1] + " "
                continue

            full = (pending + line).strip()
            start = pending_line or number
            pending, pending_line = "", 0

            fields = full.split()
            commands.append(Command(fields[0], fields[1:], str(path), start))

            if fields[0] == "include" and len(fields) > 1:
                target = path.parent / fields[1]
                if target.exists():
                    nested, nested_missing = read_commands(target, seen)
                    commands.extend(nested)
                    missing.extend(nested_missing)
                else:
                    missing.append((str(path), start, fields[1]))

    if pending:
        fields = pending.split()
        if fields:
            commands.append(Command(fields[0], fields[1:], str(path), pending_line))

    return commands, missing


def suggest(name: str, pool: set[str], cutoff: float = 0.72) -> str:
    """Closest match from a pool, for a did-you-mean hint.

    Anagrams are checked first. Most real typos are transpositions, and
    sequence similarity handles those badly: "tmep" scores closer to "time"
    than to "temp", which is the word actually meant.
    """
    letters = sorted(name.lower())
    anagrams = [word for word in pool if sorted(word) == letters and word != name]
    if anagrams:
        return min(anagrams)
    matches = difflib.get_close_matches(name, pool, n=1, cutoff=cutoff)
    return matches[0] if matches else ""


def referenced_group(command: Command) -> str | None:
    """The group a command acts on, where its position is unambiguous.

    Only commands whose group argument sits at a fixed position are covered.
    Guessing at the others would produce false alarms, and a checker that
    cries wolf gets ignored, which is worse than no checker.
    """
    if command.name in ("fix", "compute", "dump") and len(command.args) >= 2:
        return command.args[1]
    if command.name == "velocity" and command.args:
        return command.args[0]
    return None


def describe_thermostat(fix: Command) -> tuple[str, str, str]:
    """Read ensemble, temperature and pressure targets from a fix command.

    Returns (ensemble, temperature text, pressure text). Anything not present
    comes back as an empty string rather than a guess.
    """
    style = fix.args[2] if len(fix.args) >= 3 else ""
    ensemble = ENSEMBLE_STYLES.get(style, style.upper() if style else "")
    args = fix.args[3:]

    temperature = ""
    if "temp" in args:
        index = args.index("temp")
        values = args[index + 1: index + 3]
        if len(values) == 2:
            start, stop = values
            temperature = (f"{start} K" if start == stop
                           else f"{start} to {stop} K")

    pressure = ""
    for keyword in ("iso", "aniso", "tri", "x", "y", "z"):
        if keyword in args:
            index = args.index(keyword)
            values = args[index + 1: index + 3]
            if len(values) == 2:
                start, stop = values
                label = "" if keyword == "iso" else f" {keyword}"
                pressure = (f"{start} atm{label}" if start == stop
                            else f"{start} to {stop} atm{label}")
            break

    return ensemble, temperature, pressure


def record_stage(state: State, command: Command, kind: str, steps: int = 0) -> Stage:
    """Capture the conditions in force when a run or minimise is reached."""
    stage = Stage(number=len(state.stages) + 1, kind=kind,
                  line_number=command.line_number, steps=steps)

    default_step, unit = DEFAULT_TIMESTEP.get(state.units or "lj", (0.0, ""))
    stage.timestep = state.timestep or default_step
    stage.time_unit = unit

    for fix in state.fixes.values():
        style = fix.args[2] if len(fix.args) >= 3 else ""
        if style in ENSEMBLE_STYLES:
            stage.ensemble, stage.temperature, stage.pressure = describe_thermostat(fix)
        elif style.startswith(("shake", "rattle")):
            stage.constraints.append(f"{style} on group {fix.args[1]}")

    for dump_id, dump in state.dumps.items():
        if len(dump.args) >= 5:
            try:
                interval = int(dump.args[3])
            except ValueError:
                continue
            filename = dump.args[4]
            frames = steps // interval if interval else 0
            stage.dumps.append(
                f"dump {dump_id} -> {filename}, every {interval:,} steps"
                + (f" ({frames:,} frames)" if frames else "")
            )

    state.stages.append(stage)
    return stage


def analyse(commands: list[Command], strict: bool) -> tuple[list[Issue], State]:
    """Walk the script in order and collect problems."""
    issues: list[Issue] = []
    state = State()
    total_run_steps = 0
    dump_intervals: list[tuple[Command, int]] = []

    def add(level: str, command: Command, message: str, hint: str = "") -> None:
        issues.append(Issue(level, command.source, command.line_number, message, hint))

    for command in commands:
        name = command.name

        if name not in KNOWN_COMMANDS:
            guess = suggest(name, KNOWN_COMMANDS)
            add("error", command, f"unknown command {name!r}",
                f"did you mean {guess!r}?" if guess else
                "not a LAMMPS command, or from a package this checker does not cover")
            continue

        if name == "units":
            if command.args and command.args[0] not in VALID_UNITS:
                add("error", command, f"invalid units {command.args[0]!r}",
                    f"choose from: {', '.join(sorted(VALID_UNITS))}")
            elif command.args:
                state.units = command.args[0]

        if name in FILE_ARGUMENTS:
            position = FILE_ARGUMENTS[name]
            if len(command.args) > position:
                target = Path(command.source).parent / command.args[position]
                if not target.exists():
                    add("error", command,
                        f"{name} file not found: {command.args[position]}",
                        "the job stops immediately; check the path and the "
                        "working directory the scheduler will use")

        if name == "read_data" and command.args:
            target = Path(command.source).parent / command.args[0]
            if target.exists():
                try:
                    state.data = parse_data_header(target)
                except OSError:
                    state.data = None

        if name in ("read_data", "read_restart", "create_box"):
            if state.units is None:
                add("warning", command, f"{name} appears before any units command",
                    "LAMMPS then uses lj units, which is almost never intended")
            state.has_box = True

        if name == "group" and command.args:
            state.groups.add(command.args[0])
            if len(command.args) >= 3 and command.args[1] == "type":
                state.groups_by_type.append(command)
        if name == "region" and command.args:
            state.regions.add(command.args[0])
        if name == "variable" and command.args:
            state.variables.add(command.args[0])
        if name == "pair_style" and command.args:
            state.pair_style = command.args[0]
        if name.endswith("_style") and name in (
            "pair_style", "bond_style", "angle_style",
            "dihedral_style", "improper_style",
        ):
            state.styles[name] = command
        if name.endswith("_coeff"):
            state.coeffs.setdefault(name, []).append(command)
        if name == "mass":
            state.masses.append(command)

        if name == "fix" and command.args:
            fix_id = command.args[0]
            new_style = command.args[2] if len(command.args) >= 3 else ""
            existing = state.fixes.get(fix_id)
            if existing is not None:
                old_style = existing.args[2] if len(existing.args) >= 3 else ""
                if old_style and new_style and old_style != new_style:
                    # LAMMPS only overwrites a fix in place when the style is
                    # identical. A different style is fatal, and it usually
                    # means an unfix went missing after an earlier run.
                    add("error", command,
                        f"fix {fix_id!r} redefined as {new_style!r} while still "
                        f"active as {old_style!r}",
                        f"defined at line {existing.line_number} and never removed; "
                        f"LAMMPS stops with 'Replacing a fix, but new style != old "
                        f"style'. Add 'unfix {fix_id}' before this line.")
                else:
                    add("warning", command, f"fix ID {fix_id!r} is already in use",
                        f"first defined at line {existing.line_number}; same style, "
                        "so LAMMPS replaces it silently")
            state.fixes[fix_id] = command
            if new_style.startswith(("shake", "rattle")):
                state.shake_fixes.append(command)
        if name == "compute" and command.args:
            state.computes[command.args[0]] = command
        if name == "dump" and command.args:
            state.dumps[command.args[0]] = command
            if len(command.args) >= 4:
                try:
                    dump_intervals.append((command, int(command.args[3])))
                except ValueError:
                    pass

        for removal, registry, label in (
            ("unfix", state.fixes, "fix"),
            ("undump", state.dumps, "dump"),
            ("uncompute", state.computes, "compute"),
        ):
            if name == removal and command.args:
                if command.args[0] not in registry:
                    add("error", command,
                        f"{removal} {command.args[0]!r} was never defined",
                        f"LAMMPS stops with 'Could not find {label} ID'")
                else:
                    registry.pop(command.args[0])

        group = referenced_group(command)
        if group and group not in state.groups and not group.startswith(("$", "v_")):
            add("error", command, f"group {group!r} is used but never defined",
                "define it with a group command first, or check the spelling")

        if name == "timestep" and command.args:
            try:
                state.timestep = float(command.args[0])
            except ValueError:
                pass

        if name == "min_style" and command.args:
            state.min_style = command.args[0]

        if name == "thermo" and command.args:
            try:
                state.thermo_every = int(command.args[0])
            except ValueError:
                pass

        if name == "thermo_style" and len(command.args) >= 2 and command.args[0] == "custom":
            for keyword in command.args[1:]:
                if keyword.startswith(("c_", "f_", "v_")):
                    continue
                if keyword not in THERMO_KEYWORDS:
                    guess = suggest(keyword, THERMO_KEYWORDS, cutoff=0.6)
                    add("error", command, f"unknown thermo keyword {keyword!r}",
                        f"did you mean {guess!r}?" if guess else
                        "not a thermo_style custom keyword")

        if name == "minimize":
            stage = record_stage(state, command, "minimisation")
            stage.min_style = state.min_style

        if name == "run":
            integrators = [
                (fid, fix) for fid, fix in state.fixes.items()
                if len(fix.args) >= 3 and fix.args[2].startswith(INTEGRATOR_PREFIXES)
            ]
            if len(integrators) > 1:
                listing = ", ".join(
                    f"{fid} ({fix.args[2]}, line {fix.line_number})"
                    for fid, fix in integrators
                )
                add("warning", command,
                    f"{len(integrators)} time-integration fixes are active: {listing}",
                    "atoms in more than one of these are integrated twice per step; "
                    "check an earlier unfix is missing")

        if name in ("run", "minimize") and not state.has_box:
            add("error", command, f"{name} before the simulation box exists",
                "read_data, read_restart or create_box must come first")

        if name == "run" and command.args:
            try:
                steps = int(command.args[0])
            except ValueError:
                steps = 0
            total_run_steps += steps
            record_stage(state, command, "dynamics", steps)
            if steps and state.thermo_every:
                lines = steps // state.thermo_every
                size = lines * BYTES_PER_THERMO_LINE
                if size > 50e6:
                    suggested = max(1, steps // 10000)
                    add("warning", command,
                        f"this run writes about {lines:,} thermo lines, "
                        f"roughly {size / 1e9:.1f} GB of log",
                        f"'thermo {suggested}' gives about 10,000 points, "
                        "plenty for any average or plot")

        if (name == "kspace_style" and state.pair_style
                and "long" not in state.pair_style
                and "msm" not in state.pair_style):
            add("warning", command,
                f"kspace_style used with pair_style {state.pair_style!r}",
                "kspace needs a coul/long, coul/msm or similar pair style")

    for command, interval in dump_intervals:
        if interval > 0 and total_run_steps:
            frames = total_run_steps // interval
            if frames > 50000:
                issues.append(Issue(
                    "warning", command.source, command.line_number,
                    f"this dump writes roughly {frames:,} frames",
                    "check your disk quota, and consider a larger interval"))

    issues.extend(cross_check_data_file(state))

    if strict:
        for fix_id, command in state.fixes.items():
            issues.append(Issue("note", command.source, command.line_number,
                                f"fix {fix_id!r} is never removed",
                                "harmless at the end of a script"))
        for dump_id, command in state.dumps.items():
            issues.append(Issue("note", command.source, command.line_number,
                                f"dump {dump_id!r} is never closed with undump",
                                "the file is still written correctly"))

    if not state.has_box:
        issues.append(Issue("warning", commands[0].source, 0,
                            "no read_data, read_restart or create_box found",
                            "expected unless the box is set up elsewhere"))

    return issues, state


def cross_check_data_file(state: State) -> list[Issue]:
    """Check the input against the data file it reads.

    The data file states how many atom, bond and angle types exist. The input
    and its include files must supply a style and a coefficient for every one
    of them. A missing coefficient is a common and expensive failure: LAMMPS
    reports it only when the run starts, after the job has been queued,
    scheduled and started.
    """
    issues: list[Issue] = []
    data = state.data
    if data is None:
        return issues

    where = str(data.path)

    def add(level: str, message: str, hint: str = "") -> None:
        issues.append(Issue(level, where, 0, message, hint))

    # --- masses ---
    n_atom_types = data.types("atom types")
    if n_atom_types and "Masses" not in data.sections:
        defined: set[int] = set()
        for command in state.masses:
            if command.args:
                defined |= expand_types(command.args[0], n_atom_types)
        missing = sorted(set(range(1, n_atom_types + 1)) - defined)
        if missing:
            add("error",
                f"no mass for atom type{'s' if len(missing) > 1 else ''} "
                f"{', '.join(map(str, missing))}",
                f"{data.path.name} declares {n_atom_types} atom types and has no "
                "Masses section, so every type needs a mass command")
        out_of_range = sorted(i for i in defined if i > n_atom_types)
        if out_of_range:
            add("error",
                f"mass given for atom type {out_of_range[0]}, but the data file "
                f"declares only {n_atom_types}")

    # --- styles and coefficients for each interaction ---
    for label, (coeff_name, style_name, data_section) in TYPE_COUNTS.items():
        n_types = data.types(label)
        if not n_types:
            continue

        if style_name not in state.styles:
            add("error", f"{label.split()[0]} interactions exist but no {style_name} is set",
                f"{data.path.name} declares {n_types} {label}")
            continue

        if data_section in data.sections or "PairIJ Coeffs" in data.sections:
            continue  # coefficients come from the data file itself

        covered: set[int] = set()
        for command in state.coeffs.get(coeff_name, []):
            if not command.args:
                continue
            first = expand_types(command.args[0], n_types)
            if coeff_name == "pair_coeff" and len(command.args) >= 2:
                second = expand_types(command.args[1], n_types)
                # Mixing rules generate the cross terms, so only the diagonal
                # has to be given explicitly.
                covered |= first & second
            else:
                covered |= first

        missing = sorted(set(range(1, n_types + 1)) - covered)
        if missing:
            kind = label.split()[0]            # atom, bond, angle ...
            term = coeff_name.split("_")[0]    # pair, bond, angle ...
            add("error",
                f"no {coeff_name} for {kind} type{'s' if len(missing) > 1 else ''} "
                f"{', '.join(map(str, missing))}",
                f"{data.path.name} declares {n_types} {label}; LAMMPS stops with "
                f"'All {term} coeffs are not set'")

    # --- group type numbers ---
    if n_atom_types:
        for command in state.groups_by_type:
            for token in command.args[2:]:
                for value in expand_types(token, n_atom_types):
                    if value > n_atom_types:
                        issues.append(Issue(
                            "error", command.source, command.line_number,
                            f"group {command.args[0]!r} uses atom type {value}, "
                            f"but the data file declares only {n_atom_types}"))
                        break

    # --- shake bond and angle types ---
    for command in state.shake_fixes:
        args = command.args
        for flag, label in (("b", "bond types"), ("a", "angle types")):
            if flag not in args:
                continue
            limit = data.types(label)
            if not limit:
                continue
            index = args.index(flag) + 1
            while index < len(args) and args[index].lstrip("-").isdigit():
                value = int(args[index])
                if value > limit:
                    issues.append(Issue(
                        "error", command.source, command.line_number,
                        f"fix shake refers to {label[:-1]} {value}, but the data "
                        f"file declares only {limit}"))
                index += 1

    return issues


def summarise(state: State, commands: list[Command]) -> str:
    """A few lines describing what the script does."""
    runs = [c for c in commands if c.name == "run"]
    total = 0
    for command in runs:
        try:
            total += int(command.args[0])
        except (ValueError, IndexError):
            pass

    lines = [
        f"  units          {state.units or 'not set (LAMMPS would use lj)'}",
        f"  commands read  {len(commands)}",
        f"  groups         {', '.join(sorted(state.groups))}",
    ]

    if state.data is not None:
        counts = state.data.counts
        parts = [f"{counts[k]:,} {k}" for k in ("atoms", "bonds", "angles")
                 if counts.get(k)]
        types = [f"{counts[k]} {k}" for k in ("atom types", "bond types", "angle types")
                 if counts.get(k)]
        if parts:
            lines.append(f"  system         {', '.join(parts)}")
        if types:
            lines.append(f"  types          {', '.join(types)}")

    if state.stages:
        lines.append("")
        count = len(state.stages)
        plural = "stage" if count == 1 else "stages"
        lines.append(f"  {count} {plural}, {total:,} dynamics steps total")
        lines.extend(stage.describe() for stage in state.stages)
    elif runs:
        lines.append(f"  run stages     {len(runs)}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a LAMMPS input script for mistakes before submitting it.",
        epilog="Part of LAMMPS-Analysis-Toolkit by Abir Boublia.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="LAMMPS input script, for example in.lammps")
    parser.add_argument("--strict", action="store_true",
                        help="also report tidiness notes")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the issues, without the summary")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.input)

    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    try:
        commands, missing_includes = read_commands(path)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not commands:
        print(f"error: {path} contains no commands", file=sys.stderr)
        return 2

    issues, state = analyse(commands, args.strict)
    for source, line_number, target in missing_includes:
        issues.insert(0, Issue("error", source, line_number,
                               f"include file not found: {target}",
                               "its commands could not be checked"))

    if not args.quiet:
        print(f"\n{path}")
        print(summarise(state, commands))

    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    notes = [i for i in issues if i.level == "note"]

    for group, label in ((errors, "Errors"), (warnings, "Warnings"), (notes, "Notes")):
        if group:
            print(f"\n{label} ({len(group)})")
            for issue in group:
                print(issue.render(path.parent))

    if not issues:
        print("\nNothing to report.")
    print()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
