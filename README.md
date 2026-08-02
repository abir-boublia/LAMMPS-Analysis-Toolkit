# LAMMPS-Analysis-Toolkit
Python tools for analysing LAMMPS molecular dynamics trajectories of mineral-water interfaces



## Why this exists

I run production simulations dumping to DCD because it is compact,
then regularly need the same trajectory in OVITO for visualisation.
Converting by hand every time got old, so this script does it in one
command and handles the two things that break silently along the way:
the box origin, which DCD does not store, and triclinic cells.

## Installation

```bash
pip install -r requirements.txt
```
Requires Python 3.10 or newer.

## Usage

Convert a DCD trajectory to LAMMPS dump format:

```bash
python scripts/dcd_to_dump.py topology.data trajectory.dcd output.dump
```

- `topology.data` — LAMMPS data file (atom types, bonds, box)
- `trajectory.dcd` — the DCD trajectory to convert
- `output.dump` — the file to write

Run with `--help` for frame selection and atom style options.

