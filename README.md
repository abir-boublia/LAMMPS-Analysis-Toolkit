<p align="center">
  <img src="docs/banner.svg" alt="LAMMPS Analysis Toolkit" width="820">
</p>

Python tools for analysing LAMMPS molecular dynamics trajectories
of mineral-water interfaces.

## About

I work on classical MD of mineral surfaces in contact with water and
dissolved ions. The same analyses come up in every project: getting
trajectories into the right format, then measuring how water and ions
are structured near the surface. These are the tools I use, cleaned up
and documented so they are useful to someone else.

Each script runs standalone from the command line and works on any
LAMMPS trajectory, not just the systems I study.

## Tools

| Script | Purpose | Status |
|---|---|---|
| `dcd_to_dump.py` | Convert DCD trajectories to LAMMPS dump format | Available |
| `density_profile.py` | Density of water and ions along the surface normal | Planned |
| `rdf.py` | Radial distribution functions | Planned |
| `msd.py` | Mean squared displacement and diffusion coefficients | Planned |
| `hydrogen_bonds.py` | Hydrogen bond counting and lifetimes | Planned |

## Installation

```bash
git clone https://github.com/abir-boublia/LAMMPS-Analysis-Toolkit.git
cd LAMMPS-Analysis-Toolkit
pip install -r requirements.txt
```

Requires Python 3.10 or newer.

---

## dcd_to_dump.py

Converts a DCD trajectory into a LAMMPS text dump.

I run production simulations dumping to DCD because it is compact, then
regularly need the same trajectory in OVITO for visualisation. Converting
by hand every time got old, so this does it in one command and handles the
two things that break silently along the way: the box origin, which DCD
does not store, and triclinic cells.

```bash
python scripts/dcd_to_dump.py topology.lmp trajectory.dcd output.dump
```

| Argument | Meaning |
|---|---|
| `topology.lmp` | LAMMPS data file (`.data`, `.lmp`, any name): atom types, bonds, box |
| `trajectory.dcd` | DCD trajectory to convert |
| `output.dump` | File to write |

Useful options:

```bash
--step 10          convert every 10th frame
--start / --stop   restrict the frame range
--atom-style       column layout of the data file Atoms section
--anchor           whether an NPT box grows about its centre or lower corner
```

Run with `--help` for the full list.

**A note on NPT runs.** Under constant pressure the cell breathes, and LAMMPS
expands the box about its centre. The default `--anchor center` accounts for
this. If your atoms end up outside the box bounds, try `--anchor lo`.

## License

MIT. See [LICENSE](LICENSE).
