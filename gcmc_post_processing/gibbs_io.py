from __future__ import annotations

"""
Gibbs ensemble I/O: paired two-box loaders for (i) LAMMPS-style dump trajectories
and (ii) per-timestep thermodynamic data files. Designed as a small external
library to import from analysis scripts.

This module provides:

1) LAMMPSDumpReader
   - Streams frames from a single dump file with blocks like:
       ITEM: TIMESTEP\n
       <int>\n
       ITEM: NUMBER OF ATOMS\n
       <int>\n
       ITEM: BOX BOUNDS pp pp ff\n
       xlo xhi\n
       ylo yhi\n
       zlo zhi\n
       ITEM: ATOMS id x y z\n
       <id> <x> <y> <z>\n
       ...
   - Robust to 2D data (z=0 or zlo=zhi=0) and different PBC flags.
   - Yields DumpFrame(step, box, ids, pos).

2) GibbsDumpPair
   - Wraps two LAMMPSDumpReader instances (box1, box2) and yields synchronized
     frames by TIMESTEP (strict, default) or by order.

3) DataFileReader
   - Streams records from a per-timestep CSV-like text with key:value pairs,
     e.g. "Timestep:409600000 ,\tEnergy:533.16350, ..." (commas may have spaces).
   - Yields DataRecord(step, energy, temperature, pressure, volume, density, n).

4) GibbsTwoBoxLoader
   - Pairs trajectory frames from both boxes and optionally attaches matching
     DataRecord entries from the two data files (by TIMESTEP).

All objects are iterable / streaming (low memory). NumPy arrays are returned for
positions. Intended usage:

    from gibbs_io import GibbsTwoBoxLoader
    loader = GibbsTwoBoxLoader(
        dump1_path="box1.dump", dump2_path="box2.dump",
        data1_path="box1.data.txt", data2_path="box2.data.txt",
        strict_sync=True,
    )
    for fr in loader:
        # fr.pos1, fr.pos2 -> (N1,3)/(N2,3)
        # fr.data1, fr.data2 -> optional thermodynamic metadata
        ...

Notes
-----
- TIMESTEP values may be negative (e.g., 32-bit overflow in some exports). We
  parse them as Python int without constraints and use them for synchronization.
- Column order after "ITEM: ATOMS" is respected. We require at least id,x,y and
  accept optional z; if z missing, we set z=0.
- BOX BOUNDS line is used to determine periodicity (p=periodic, f=fixed).

This file has no external dependencies beyond NumPy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import io
import re
import numpy as np

# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Box:
    Lx: float
    Ly: float
    Lz: float
    pbc: Tuple[bool, bool, bool]  # True if periodic in x,y,z

    @staticmethod
    def from_bounds(xlo: float, xhi: float, ylo: float, yhi: float, zlo: float, zhi: float,
                    flags: Tuple[str, str, str]) -> "Box":
        Lx = float(xhi) - float(xlo)
        Ly = float(yhi) - float(ylo)
        Lz = float(zhi) - float(zlo)
        pbc = tuple(f in ("p", "pp", "s", "sp") for f in flags)  # treat shrink-wrapped as periodic for imaging
        return Box(Lx=Lx, Ly=Ly, Lz=Lz, pbc=(bool(pbc[0]), bool(pbc[1]), bool(pbc[2])))


@dataclass(frozen=True)
class DumpFrame:
    step: int
    box: Box
    ids: np.ndarray      # (N,), int64
    pos: np.ndarray      # (N,3), float64


@dataclass(frozen=True)
class GibbsFrame:
    step: int
    # Box 1
    box1: Box
    ids1: np.ndarray
    pos1: np.ndarray
    # Box 2
    box2: Box
    ids2: np.ndarray
    pos2: np.ndarray
    # Optional thermodynamic metadata
    data1: Optional["DataRecord"] = None
    data2: Optional["DataRecord"] = None


@dataclass(frozen=True)
class DataRecord:
    step: int
    energy: Optional[float]
    temperature: Optional[float]
    pressure: Optional[float]
    volume: Optional[float]
    density: Optional[float]
    n_particles: Optional[int]


# -----------------------------
# LAMMPS dump reader (single box)
# -----------------------------

class LAMMPSDumpReader:
    """Stream frames from a LAMMPS-style dump file.

    Parameters
    ----------
    path : str | Path
    id_field : str, default "id"
        Name of the id column in the ATOMS header (usually "id").
    x_field,y_field,z_field : str
        Coordinate column names (default "x","y","z"). If z_field is missing
        in file, z is set to 0.
    dtype : np.dtype
        Float dtype for coordinates.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        id_field: str = "id",
        x_field: str = "x",
        y_field: str = "y",
        z_field: str = "z",
        dtype: np.dtype = np.float64,
    ) -> None:
        self.path = Path(path)
        self.id_field = id_field
        self.x_field = x_field
        self.y_field = y_field
        self.z_field = z_field
        self.dtype = np.dtype(dtype)

    def __iter__(self) -> Iterator[DumpFrame]:
        with self.path.open("r", encoding="utf-8") as fh:
            while True:
                # Expect: ITEM: TIMESTEP
                line = _read_nonempty(fh)
                if line is None:
                    return
                if not line.startswith("ITEM: TIMESTEP"):
                    raise ValueError(f"Unexpected line (expect 'ITEM: TIMESTEP'): {line!r}")
                step_line = _read_nonempty(fh)
                if step_line is None:
                    return
                try:
                    step = int(step_line.strip())
                except Exception as e:
                    raise ValueError(f"Invalid TIMESTEP value: {step_line!r}") from e

                # NUMBER OF ATOMS
                line = _read_nonempty(fh)
                if line is None or not line.startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError("Missing 'ITEM: NUMBER OF ATOMS' block")
                n_line = _read_nonempty(fh)
                if n_line is None:
                    raise ValueError("Unexpected EOF reading atom count")
                try:
                    n_atoms = int(n_line.strip())
                except Exception as e:
                    raise ValueError(f"Invalid atom count: {n_line!r}") from e

                # BOX BOUNDS ... flags
                line = _read_nonempty(fh)
                if line is None or not line.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError("Missing 'ITEM: BOX BOUNDS' block")
                flags = _parse_bounds_flags(line)
                xlo_xhi = _read_floats_line(fh)
                ylo_yhi = _read_floats_line(fh)
                zlo_zhi = _read_floats_line(fh)
                box = Box.from_bounds(xlo_xhi[0], xlo_xhi[1], ylo_yhi[0], ylo_yhi[1], zlo_zhi[0], zlo_zhi[1], flags)

                # ATOMS header
                line = _read_nonempty(fh)
                if line is None or not line.startswith("ITEM: ATOMS"):
                    raise ValueError("Missing 'ITEM: ATOMS' block")
                fields = line.split()[2:]  # after 'ITEM:' 'ATOMS'
                # indices for id/x/y/z
                try:
                    i_id = fields.index(self.id_field)
                except ValueError:
                    raise ValueError(f"Required id field '{self.id_field}' not found in ATOMS header: {fields}")
                i_x = _safe_index(fields, self.x_field)
                i_y = _safe_index(fields, self.y_field)
                i_z = _safe_index(fields, self.z_field, required=False)

                ids = np.empty(n_atoms, dtype=np.int64)
                pos = np.zeros((n_atoms, 3), dtype=self.dtype)
                for i in range(n_atoms):
                    line = _read_nonempty(fh)
                    if line is None:
                        raise ValueError("Unexpected EOF in ATOMS block")
                    parts = line.split()
                    try:
                        ids[i] = int(parts[i_id])
                        pos[i, 0] = float(parts[i_x])
                        pos[i, 1] = float(parts[i_y])
                        if i_z is not None and i_z < len(parts):
                            pos[i, 2] = float(parts[i_z])
                        else:
                            pos[i, 2] = 0.0
                    except Exception as e:
                        raise ValueError(f"Invalid ATOMS row: {line!r}") from e

                yield DumpFrame(step=step, box=box, ids=ids, pos=pos)


def _read_nonempty(fh: io.TextIOBase) -> Optional[str]:
    while True:
        s = fh.readline()
        if s == "":
            return None
        s = s.strip()
        if s != "":
            return s


def _read_floats_line(fh: io.TextIOBase) -> Tuple[float, float]:
    s = _read_nonempty(fh)
    if s is None:
        raise ValueError("Unexpected EOF while reading bounds line")
    parts = s.split()
    if len(parts) < 2:
        raise ValueError(f"Expected two floats in bounds line, got: {s!r}")
    return float(parts[0]), float(parts[1])


def _parse_bounds_flags(line: str) -> Tuple[str, str, str]:
    # Example: "ITEM: BOX BOUNDS pp pp ff" -> ("pp","pp","ff")
    parts = line.split()
    if len(parts) < 6:
        # Some dumps use: ITEM: BOX BOUNDS xx yy zz tilt factors  (not supported here)
        # Keep simple and raise for unsupported formats.
        raise ValueError(f"Unsupported BOX BOUNDS header: {line!r}")
    return parts[-3], parts[-2], parts[-1]


def _safe_index(seq: List[str], key: str, required: bool = True) -> Optional[int]:
    try:
        return seq.index(key)
    except ValueError:
        if required:
            raise
        return None


# -----------------------------
# Pair two dump streams
# -----------------------------

class GibbsDumpPair:
    """Synchronize two dump readers.

    Parameters
    ----------
    dump1_path, dump2_path : str | Path
    strict_sync : bool (default True)
        If True, TIMESTEP must match for the two boxes; if False, frames are
        paired by order.
    """

    def __init__(self, dump1_path: str | Path, dump2_path: str | Path, *, strict_sync: bool = True) -> None:
        self.r1 = LAMMPSDumpReader(dump1_path)
        self.r2 = LAMMPSDumpReader(dump2_path)
        self.strict = strict_sync

    def __iter__(self) -> Iterator[GibbsFrame]:
        i1 = iter(self.r1)
        i2 = iter(self.r2)
        while True:
            try:
                f1 = next(i1)
                f2 = next(i2)
            except StopIteration:
                return
            if self.strict and f1.step != f2.step:
                raise ValueError(f"TIMESTEP mismatch: box1={f1.step} box2={f2.step}")
            step = f1.step if self.strict else f1.step
            yield GibbsFrame(step=step, box1=f1.box, ids1=f1.ids, pos1=f1.pos,
                             box2=f2.box, ids2=f2.ids, pos2=f2.pos)


# -----------------------------
# Thermodynamic data reader (single box)
# -----------------------------

_DATA_KV_RE = re.compile(r"\s*([A-Za-z][A-Za-z ]*[A-Za-z])\s*:\s*([^,\t]+)")


class DataFileReader:
    """Stream key:value records from a text file (one record per line).

    Expected keys (case-insensitive, extra spaces tolerated in keys):
      Timestep, Energy, Tempreture/Temperature, Pressure, Volume,
      density of particles / Density, NumberofParticles / Number of Particles

    Unknown keys are ignored. Values are parsed as float except Timestep and
    Number* which are parsed as int when possible.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[DataRecord]:
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                kv = _parse_kv_line(line)
                step = _to_int(kv.get("timestep"))
                rec = DataRecord(
                    step=step if step is not None else 0,
                    energy=_to_float(kv.get("energy")),
                    temperature=_to_float(kv.get("tempreture")) if kv.get("tempreture") is not None else _to_float(kv.get("temperature")),
                    pressure=_to_float(kv.get("pressure")),
                    volume=_to_float(kv.get("volume")),
                    density=_to_float(kv.get("density of particles")) if kv.get("density of particles") is not None else _to_float(kv.get("density")),
                    n_particles=_to_int(kv.get("numberofparticles")) if kv.get("numberofparticles") is not None else _to_int(kv.get("number of particles")),
                )
                yield rec


def _parse_kv_line(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _DATA_KV_RE.finditer(line):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        # Drop trailing commas if any remainder
        if val.endswith(','):
            val = val[:-1].strip()
        out[key] = val
    return out


def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(x))  # tolerate "123.0"
    except Exception:
        return None


# -----------------------------
# Two-box orchestrator that attaches thermo data
# -----------------------------

class GibbsTwoBoxLoader:
    """Pair two dumps and (optionally) attach matching data-file records.

    Parameters
    ----------
    dump1_path, dump2_path : str | Path
        Trajectory dumps for Box 1 and Box 2.
    data1_path, data2_path : str | Path | None
        Optional thermo data files for Box 1 and Box 2. If provided, they are
        indexed by TIMESTEP for O(1) lookup per frame.
    strict_sync : bool, default True
        Require equal TIMESTEP per frame for the two dumps. If False, pair by
        order; data records are then looked up by each frame's own step.
    """

    def __init__(
        self,
        *,
        dump1_path: str | Path,
        dump2_path: str | Path,
        data1_path: str | Path | None = None,
        data2_path: str | Path | None = None,
        strict_sync: bool = True,
    ) -> None:
        self.pair = GibbsDumpPair(dump1_path, dump2_path, strict_sync=strict_sync)
        self.data1 = _build_data_index(data1_path) if data1_path is not None else None
        self.data2 = _build_data_index(data2_path) if data2_path is not None else None

    def __iter__(self) -> Iterator[GibbsFrame]:
        for fr in self.pair:
            d1 = self.data1.get(fr.step) if self.data1 is not None else None
            d2 = self.data2.get(fr.step) if self.data2 is not None else None
            yield GibbsFrame(step=fr.step, box1=fr.box1, ids1=fr.ids1, pos1=fr.pos1,
                             box2=fr.box2, ids2=fr.ids2, pos2=fr.pos2,
                             data1=d1, data2=d2)


def _build_data_index(path: str | Path) -> Dict[int, DataRecord]:
    idx: Dict[int, DataRecord] = {}
    for rec in DataFileReader(path):
        idx[rec.step] = rec  # last-wins if duplicates
    return idx


