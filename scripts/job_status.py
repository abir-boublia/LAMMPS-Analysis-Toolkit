#!/usr/bin/env python3
"""
job_status.py

Reports on a LAMMPS job while it is running.

Answers the questions you have while a job is in the queue or on a node: how
far along is it, when will it finish, and has anything gone wrong. Progress
comes from the log, and the rate is measured by reading the log twice a few
seconds apart, which is the only honest way to get it while the job is still
writing.

Large logs are read by scanning in chunks and keeping only what is needed, so
a multi-gigabyte log costs a few seconds rather than a few gigabytes of
memory.

Author:  Abir Boublia
Contact: abir.boublia@univ-lorraine.fr
ORCID:   https://orcid.org/0000-0003-1669-4951
Part of: LAMMPS-Analysis-Toolkit
License: MIT

Usage:
    python job_status.py log.lammps
    python job_status.py log.lammps --sample 30
    python job_status.py log.lammps --once
    python job_status.py --help

Uses only the standard library. Requires Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

__author__ = "Abir Boublia"
__license__ = "MIT"
__version__ = "1.0.0"

# Bytes read from the start, for the input commands LAMMPS echoes there.
HEAD_BYTES = 256 * 1024

# Bytes read from the end, for the most recent thermo lines.
TAIL_BYTES = 256 * 1024

# Chunk size for the single pass over the file.
CHUNK_BYTES = 4 * 1024 * 1024

# Default timestep per unit system, and the unit it is written in.
DEFAULT_TIMESTEP = {
    "real": (1.0, "fs"), "metal": (0.001, "ps"), "lj": (0.005, ""),
    "si": (1.0e-8, "s"), "cgs": (1.0e-8, "s"), "electron": (0.001, "fs"),
    "micro": (2.0, "us"), "nano": (0.00045, "ns"),
}

PER_NANOSECOND = {"fs": 1.0e6, "ps": 1.0e3, "ns": 1.0, "us": 1.0e-3, "s": 1.0e-9}

# A log untouched for longer than this is probably not running.
STALE_SECONDS = 300

# Temperature this far from the thermostat target is worth reporting.
TEMPERATURE_TOLERANCE = 0.1  # as a fraction of the target

PROGRESS_WIDTH = 30


@dataclass
class Plan:
    """What the input script asked for, read from the echoed commands."""

    units: str = "real"
    timestep: float | None = None
    run_targets: list[int] = field(default_factory=list)
    temperature_target: float | None = None

    @property
    def step_size(self) -> tuple[float, str]:
        default, unit = DEFAULT_TIMESTEP.get(self.units, (0.0, ""))
        return (self.timestep if self.timestep else default), unit


@dataclass
class Progress:
    """Where the job has got to."""

    columns: list[str] = field(default_factory=list)
    last_values: list[float] = field(default_factory=list)
    recent: list[list[float]] = field(default_factory=list)
    completed_runs: int = 0
    finished: bool = False
    lost_atoms: bool = False
    warnings: int = 0

    def value(self, name: str) -> float | None:
        if name in self.columns and self.last_values:
            index = self.columns.index(name)
            if index < len(self.last_values):
                return self.last_values[index]
        return None

    @property
    def step(self) -> int | None:
        value = self.value(self.columns[0]) if self.columns else None
        return int(value) if value is not None else None


def read_head(path: Path) -> Plan:
    """Read the input commands LAMMPS echoes at the top of the log."""
    plan = Plan()
    with open(path, "rb") as handle:
        text = handle.read(HEAD_BYTES).decode("utf-8", "replace")

    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == "units" and len(fields) > 1:
            plan.units = fields[1]
        elif fields[0] == "timestep" and len(fields) > 1:
            try:
                plan.timestep = float(fields[1])
            except ValueError:
                pass
        elif fields[0] == "run" and len(fields) > 1:
            try:
                plan.run_targets.append(int(fields[1]))
            except ValueError:
                pass
        elif fields[0] == "fix" and "temp" in fields:
            index = fields.index("temp")
            if len(fields) > index + 2:
                try:
                    plan.temperature_target = float(fields[index + 2])
                except ValueError:
                    pass
    return plan


def scan(path: Path) -> Progress:
    """One pass over the log, keeping only what the report needs.

    The thermo header can sit millions of lines above the end of the file, so
    it is tracked while scanning rather than looked for in the tail. Counting
    completed runs tells us which stage of a multi-run script is active.
    """
    progress = Progress()
    header_line = ""
    size = path.stat().st_size

    with open(path, "rb") as handle:
        overlap = b""
        position = 0
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            block = overlap + chunk

            progress.completed_runs += block.count(b"Loop time of")
            progress.warnings += block.count(b"WARNING")
            if b"Lost atoms" in block:
                progress.lost_atoms = True
            if b"Total wall time" in block:
                progress.finished = True

            # Find the most recent thermo header. Some LAMMPS builds indent
            # it and some do not, so search for the word and then check the
            # whole line rather than anchoring to the newline.
            search_from = len(block)
            while True:
                index = block.rfind(b"Step", 0, search_from)
                if index == -1:
                    break
                start = block.rfind(b"\n", 0, index) + 1
                end = block.find(b"\n", index)
                candidate = block[start:end if end != -1 else None]
                if candidate.strip().startswith(b"Step"):
                    header_line = candidate.decode("utf-8", "replace")
                    break
                search_from = index

            overlap = block[-256:]
            position += len(chunk)

        # The last thermo values live at the end of the file.
        handle.seek(max(0, size - TAIL_BYTES))
        tail = handle.read().decode("utf-8", "replace")

    if header_line:
        progress.columns = header_line.split()

    width = len(progress.columns)
    for line in tail.splitlines():
        fields = line.split()
        if width and len(fields) == width:
            try:
                progress.recent.append([float(v) for v in fields])
            except ValueError:
                continue

    if progress.recent:
        progress.last_values = progress.recent[-1]

    return progress


def last_step(path: Path, columns: int) -> int | None:
    """Cheaply read the most recent step from the end of the file."""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        handle.seek(max(0, size - TAIL_BYTES))
        tail = handle.read().decode("utf-8", "replace")

    for line in reversed(tail.splitlines()):
        fields = line.split()
        if len(fields) == columns:
            try:
                return int(float(fields[0]))
            except ValueError:
                continue
    return None


def tail_lines(path: Path, count: int) -> list[str]:
    """The last few non-empty lines, for showing what state a log is in."""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        handle.seek(max(0, size - TAIL_BYTES))
        text = handle.read().decode("utf-8", "replace")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-count:] if lines else ["(the file is empty)"]


def format_duration(seconds: float) -> str:
    """A rough, readable duration."""
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def progress_bar(fraction: float, width: int = PROGRESS_WIDTH) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def build_report(
    path: Path,
    plan: Plan,
    progress: Progress,
    rate: float | None,
    sampled_over: float,
) -> list[str]:
    """Assemble the report."""
    age = time.time() - path.stat().st_mtime

    if progress.finished:
        state = "finished"
    elif progress.lost_atoms:
        state = "FAILED, lost atoms"
    elif age > STALE_SECONDS:
        state = f"no output for {format_duration(age)}"
    else:
        state = "running"

    lines = [f"{path}   {state}", ""]

    step = progress.step
    stage = progress.completed_runs + 1
    total_stages = len(plan.run_targets)
    target = (plan.run_targets[progress.completed_runs]
              if progress.completed_runs < total_stages else None)

    if total_stages:
        if progress.finished:
            label = f"all {total_stages} stage{'s' if total_stages > 1 else ''} complete"
        else:
            label = f"stage {min(stage, total_stages)} of {total_stages}"
        if plan.temperature_target:
            label += f",  thermostat {plan.temperature_target:g} K"
        lines.append(label)

    if step is not None and target:
        fraction = min(1.0, step / target) if target else 0.0
        lines.append(f"step {step:,} of {target:,}      {fraction * 100:.1f}%")
        lines.append(progress_bar(fraction))

        size, unit = plan.step_size
        if size and unit in PER_NANOSECOND:
            done = step * size / PER_NANOSECOND[unit]
            whole = target * size / PER_NANOSECOND[unit]
            lines.append(f"{done:.3g} ns of {whole:.3g} ns simulated")

        if rate:
            remaining = (target - step) / rate
            finish = datetime.now().astimezone() + timedelta(seconds=remaining)
            lines.append(
                f"{rate:,.0f} steps/s   remaining {format_duration(remaining)}"
                f"   finishing about {finish:%a %H:%M}"
            )
        elif sampled_over > 0 and not progress.finished:
            lines.append(f"no progress in {format_duration(sampled_over)} of watching")
    elif step is not None:
        lines.append(f"step {step:,}")

    # --- current thermodynamic state ---
    interesting = [c for c in ("Temp", "Press", "TotEng", "PotEng", "Volume")
                   if c in progress.columns]
    if interesting:
        lines.append("")
        for name in interesting:
            value = progress.value(name)
            if value is None:
                continue
            note = ""
            if (name == "Temp" and plan.temperature_target
                    and plan.temperature_target > 0):
                drift = abs(value - plan.temperature_target) / plan.temperature_target
                if drift > TEMPERATURE_TOLERANCE:
                    note = f"   off target by {value - plan.temperature_target:+.0f} K"
            lines.append(f"{name:<8}{value:>16,.4g}{note}")

    lines.append("")
    lines.append(f"log {path.stat().st_size / 1e6:,.0f} MB, "
                 f"last written {format_duration(age)} ago")

    return lines


def health(plan: Plan, progress: Progress) -> list[str]:
    """Signs that the run is in trouble."""
    found = []

    if progress.lost_atoms:
        found.append("LAMMPS reported lost atoms; the run has stopped")

    for name, value in zip(progress.columns, progress.last_values, strict=False):
        if not math.isfinite(value):
            found.append(f"{name} is {value}; the run has gone unstable")

    temperature = progress.value("Temp")
    if (temperature is not None and plan.temperature_target
            and plan.temperature_target > 0):
        drift = abs(temperature - plan.temperature_target) / plan.temperature_target
        if drift > TEMPERATURE_TOLERANCE:
            found.append(
                f"temperature is {temperature:.1f} K against a target of "
                f"{plan.temperature_target:g} K"
            )

    # A sudden jump between consecutive frames is how a blow-up begins.
    if "TotEng" in progress.columns and len(progress.recent) > 20:
        index = progress.columns.index("TotEng")
        series = [row[index] for row in progress.recent[-50:]]
        spread = max(series) - min(series)
        typical = abs(sum(series) / len(series))
        if typical and spread > 0.05 * typical:
            found.append(
                f"total energy moved by {spread:,.0f} over the last "
                f"{len(series)} frames, {spread / typical * 100:.0f}% of its value"
            )

    if progress.warnings > 50:
        found.append(f"{progress.warnings:,} WARNING lines in the log")

    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report on a LAMMPS job while it runs.",
        epilog="Part of LAMMPS-Analysis-Toolkit by Abir Boublia.",
    )
    parser.add_argument("log", nargs="?", default="log.lammps",
                        help="LAMMPS log file (default: log.lammps)")
    parser.add_argument("--sample", type=float, default=15.0,
                        help="seconds to watch for measuring the rate (default: 15)")
    parser.add_argument("--once", action="store_true",
                        help="do not watch; skip the rate and the finish time")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    path = Path(args.log)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    plan = read_head(path)
    progress = scan(path)

    if not progress.columns:
        size = path.stat().st_size
        age = time.time() - path.stat().st_mtime
        print(f"\n{path}   {size:,} bytes, last written "
              f"{format_duration(age)} ago\n")
        print("No thermo table found. The last lines of the log were:\n")
        for line in tail_lines(path, 8):
            print(f"  {line}")
        print("\nIf the run has started, this is a header the scanner did not "
              "recognise;\nif it has not, the lines above say why.\n")
        return 1

    rate = None
    sampled = 0.0
    if not args.once and not progress.finished and progress.step is not None:
        first_step, first_time = progress.step, time.time()
        time.sleep(args.sample)
        second_step = last_step(path, len(progress.columns))
        sampled = time.time() - first_time
        if second_step is not None and second_step > first_step:
            rate = (second_step - first_step) / sampled
            progress.last_values = scan(path).last_values or progress.last_values

    print()
    print("\n".join(build_report(path, plan, progress, rate, sampled)))

    problems = health(plan, progress)
    if problems:
        title = f"Worth checking ({len(problems)})"
        print()
        print(title)
        print("-" * len(title))
        for item in problems:
            print(f"- {item}")
    print()

    return 1 if problems else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(0)

