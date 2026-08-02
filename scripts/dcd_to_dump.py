#!/usr/bin/env python3
"""
dcd_to_dump.py

Converts a DCD trajectory into a LAMMPS text dump.

DCD files are compact, which is why we write them, but a lot of tools read
LAMMPS dump format more reliably. The conversion itself is trivial. The two
things that go wrong are the box origin (DCD does not store it) and triclinic
cells (easy to write as if they were orthogonal). Both are handled below.

Author:  Abir Boublia
Contact: abir.boublia@univ-lorraine.fr
ORCID:   https://orcid.org/0000-0003-1669-4951
Part of: LAMMPS-Analysis-Toolkit
License: MIT
Created: August 2026

Usage:
    python dcd_to_dump.py system.lmp prod.dcd traj.dump
    python dcd_to_dump.py system.lmp prod.dcd traj.dump --step 10
    python dcd_to_dump.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

__author__ = "Abir Boublia"
__license__ = "MIT"
__version__ = "1.0.0"

# Angles this close to 90 count as orthogonal.
ORTHO_TOL_DEG = 1e-4


def read_box_origin(data_file: str | Path) -> np.ndarray:
    """Get xlo, ylo, zlo from a LAMMPS data file.

    DCD stores box lengths and angles but not where the box starts. If your
    cell runs from -20 to 20 and you assume it starts at zero, every
    coordinate in the output is off by 20 and nothing warns you. So we read
    the origin from the data file instead.

    Returns zeros for any bound that isn't found.
    """
    origin = np.zeros(3)
    keys = {"xlo": 0, "ylo": 1, "zlo": 2}

    with open(data_file) as handle:
        for line in handle:
            fields = line.split()
            # Header lines look like:  0.0 40.0 xlo xhi
            if len(fields) >= 4 and fields[2] in keys and fields[3].endswith("hi"):
                origin[keys[fields[2]]] = float(fields[0])
            # Stop once the sections start.
            if fields and fields[0] in ("Atoms", "Masses", "Velocities"):
                break

    return origin


def box_bounds(dimensions: np.ndarray, origin: np.ndarray) -> tuple[str, np.ndarray]:
    """Build the BOX BOUNDS header and values for one frame.

    Takes MDAnalysis dimensions [Lx, Ly, Lz, alpha, beta, gamma] and the
    origin from the data file. Returns the ITEM line plus a (3, 2) array for
    an orthogonal box, or (3, 3) with tilt factors for a triclinic one.

    Worth knowing: a LAMMPS dump stores the *bounding box* for triclinic
    cells, meaning the limits already widened by the tilt factors. Writing the
    plain cell edges there is the usual mistake, and it breaks distances near
    the boundary in whatever reads the file afterwards.
    """
    from MDAnalysis.lib.mdamath import triclinic_vectors

    lengths = np.asarray(dimensions[:3], dtype=float)
    angles = np.asarray(dimensions[3:6], dtype=float)
    lo = np.asarray(origin, dtype=float)

    if np.all(np.abs(angles - 90.0) < ORTHO_TOL_DEG):
        return "ITEM: BOX BOUNDS pp pp pp", np.column_stack([lo, lo + lengths])

    vectors = triclinic_vectors(dimensions)
    xy, xz, yz = vectors[1, 0], vectors[2, 0], vectors[2, 1]
    hi = lo + np.array([vectors[0, 0], vectors[1, 1], vectors[2, 2]])

    xlo_bound = lo[0] + min(0.0, xy, xz, xy + xz)
    xhi_bound = hi[0] + max(0.0, xy, xz, xy + xz)
    ylo_bound = lo[1] + min(0.0, yz)
    yhi_bound = hi[1] + max(0.0, yz)

    rows = np.array([
        [xlo_bound, xhi_bound, xy],
        [ylo_bound, yhi_bound, xz],
        [lo[2], hi[2], yz],
    ])
    return "ITEM: BOX BOUNDS xy xz yz pp pp pp", rows


def convert(
    data_file: str | Path,
    dcd_file: str | Path,
    out_file: str | Path,
    atom_style: str = "id resid type charge x y z",
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> int:
    """Convert the DCD and return how many frames were written."""
    import MDAnalysis as mda

    universe = mda.Universe(
        str(data_file),
        str(dcd_file),
        topology_format="DATA",
        atom_style=atom_style,
        format="DCD",
    )

    origin = read_box_origin(data_file)
    n_atoms = universe.atoms.n_atoms

    # Ids and types never change, so allocate once and only swap in the
    # coordinates each frame. Much faster than writing atom by atom.
    frame_data = np.empty((n_atoms, 5))
    frame_data[:, 0] = universe.atoms.ids
    frame_data[:, 1] = universe.atoms.types.astype(int)

    n_written = 0
    with open(out_file, "w") as out:
        for ts in universe.trajectory[start:stop:step]:
            if ts.dimensions is None:
                raise ValueError(
                    f"Frame {ts.frame} has no box information. Either the DCD is "
                    "corrupt or it was written without unit cell records."
                )

            header, bounds = box_bounds(ts.dimensions, origin)

            out.write("ITEM: TIMESTEP\n")
            out.write(f"{ts.frame}\n")
            out.write("ITEM: NUMBER OF ATOMS\n")
            out.write(f"{n_atoms}\n")
            out.write(f"{header}\n")
            np.savetxt(out, bounds, fmt="%.6f")
            out.write("ITEM: ATOMS id type x y z\n")

            frame_data[:, 2:5] = universe.atoms.positions
            np.savetxt(out, frame_data, fmt="%d %d %.6f %.6f %.6f")
            n_written += 1

    return n_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a DCD trajectory to a LAMMPS dump file.",
        epilog="Part of LAMMPS-Analysis-Toolkit by Abir Boublia.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="LAMMPS data file with the topology")
    parser.add_argument("dcd", help="DCD trajectory to convert")
    parser.add_argument("out", help="output dump file")
    parser.add_argument(
        "--atom-style",
        default="id resid type charge x y z",
        help="column layout of the Atoms section in the data file",
    )
    parser.add_argument("--start", type=int, default=0, help="first frame to convert")
    parser.add_argument("--stop", type=int, default=None, help="stop before this frame")
    parser.add_argument("--step", type=int, default=1, help="convert every Nth frame")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for path in (args.data, args.dcd):
        if not Path(path).exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    try:
        n_frames = convert(
            args.data, args.dcd, args.out,
            atom_style=args.atom_style,
            start=args.start, stop=args.stop, step=args.step,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {n_frames} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
