from __future__ import annotations

"""
Gibbs ensemble I/O: robust, streaming, two-box loaders for (i) LAMMPS-style dump
trajectories and (ii) per-timestep thermodynamic data files.

Key robustness features requested:
- Malformed/corrupted dump frames are **skipped** (on_error="skip").
- Two dump streams are **merged by TIMESTEP**; steps missing in either stream
  are **dropped** (inner-join behavior).
- Data files are optional; if provided and `skip_if_missing_data=True`, steps
  missing in either data file are **dropped** as well.

Public API (import from package root if exported there):
    LAMMPSDumpReader, GibbsDumpPair, DataFileReader, GibbsTwoBoxLoader
    load_gibbs(...), iter_gibbs(...), index_datafile(path)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
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
    pbc: Tuple[bool, bool, bool]

    @staticmethod
    def from_bounds(xlo: float, xhi: float, ylo: float, yhi: float, zlo: float, zhi: float,
                    flags: Tuple[str, str, str]) -> "Box":
        Lx = float(xhi) - float(xlo)
        Ly = float(yhi) - float(ylo)
        Lz = float(zhi) - float(zlo)
        pbc = tuple(f in ("p", "pp", "s", "sp") for f in flags)
        return Box(Lx=Lx, Ly=Ly, Lz=Lz, pbc=(bool(pbc[0]), bool(pbc[1]), bool(pbc[2])))


@dataclass(frozen=True)
class DumpFrame:
    step: int
    box: Box
    ids: np.ndarray
    pos: np.ndarray


@dataclass(frozen=True)
class GibbsFrame:
    step: int
    box1: Box
    ids1: np.ndarray
    pos1: np.ndarray
    box2: Box
    ids2: np.ndarray
    pos2: np.ndarray
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
# Helpers
# -----------------------------

def _read_until_prefix(fh: io.TextIOBase, prefix: str) -> Optional[str]:
    while True:
        s = fh.readline()
        if s == "":
            return None
        s = s.strip()
        if s.startswith(prefix):
            return s


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
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"Unsupported BOX BOUNDS header: {line!r}")
    return parts[-3], parts[-2], parts[-1]


def _safe_index(seq: List[str], key: str, required: bool = True) -> Optional[int]:
    try:
        return seq.index(key)
    except ValueError:
        if required:
            raise
        return None


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
        return int(float(x))
    except Exception:
        return None

# -----------------------------
# LAMMPS dump reader (robust)
# -----------------------------

class LAMMPSDumpReader:
    """Stream frames from a LAMMPS-style dump. Malformed frames are skippable.

    on_error: {"skip","raise"} controls behavior. Default is "skip".
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
        on_error: str = "skip",
    ) -> None:
        self.path = Path(path)
        self.id_field = id_field
        self.x_field = x_field
        self.y_field = y_field
        self.z_field = z_field
        self.dtype = np.dtype(dtype)
        if on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be 'skip' or 'raise'")
        self.on_error = on_error

    def __iter__(self) -> Iterator[DumpFrame]:
        with self.path.open("r", encoding="utf-8") as fh:
            while True:
                if _read_until_prefix(fh, "ITEM: TIMESTEP") is None:
                    return
                step_line = _read_nonempty(fh)
                if step_line is None:
                    return
                try:
                    step = int(step_line)
                except Exception as e:
                    if self.on_error == "skip":
                        continue
                    raise ValueError(f"Invalid TIMESTEP value: {step_line!r}") from e
                try:
                    line = _read_nonempty(fh)
                    if line is None or not line.startswith("ITEM: NUMBER OF ATOMS"):
                        raise ValueError("Missing 'ITEM: NUMBER OF ATOMS'")
                    n_atoms = int(_read_nonempty(fh))

                    line = _read_nonempty(fh)
                    if line is None or not line.startswith("ITEM: BOX BOUNDS"):
                        raise ValueError("Missing 'ITEM: BOX BOUNDS'")
                    flags = _parse_bounds_flags(line)
                    xlo_xhi = _read_floats_line(fh)
                    ylo_yhi = _read_floats_line(fh)
                    zlo_zhi = _read_floats_line(fh)
                    box = Box.from_bounds(xlo_xhi[0], xlo_xhi[1], ylo_yhi[0], ylo_yhi[1], zlo_zhi[0], zlo_zhi[1], flags)

                    line = _read_nonempty(fh)
                    if line is None or not line.startswith("ITEM: ATOMS"):
                        raise ValueError("Missing 'ITEM: ATOMS'")
                    fields = line.split()[2:]
                    i_id = fields.index(self.id_field)
                    i_x = _safe_index(fields, self.x_field)
                    i_y = _safe_index(fields, self.y_field)
                    i_z = _safe_index(fields, self.z_field, required=False)

                    ids = np.empty(n_atoms, dtype=np.int64)
                    pos = np.zeros((n_atoms, 3), dtype=self.dtype)
                    for i in range(n_atoms):
                        row = _read_nonempty(fh)
                        if row is None:
                            raise ValueError("Unexpected EOF in ATOMS block")
                        parts = row.split()
                        ids[i] = int(parts[i_id])
                        pos[i, 0] = float(parts[i_x])
                        pos[i, 1] = float(parts[i_y])
                        pos[i, 2] = float(parts[i_z]) if (i_z is not None and i_z < len(parts)) else 0.0

                    yield DumpFrame(step=step, box=box, ids=ids, pos=pos)
                except Exception:
                    if self.on_error == "skip":
                        continue
                    raise


# -----------------------------
# Pair two dump streams (merge by TIMESTEP)
# -----------------------------

class GibbsDumpPair:
    """Yield synchronized frames; drop steps missing in either stream if strict_sync=True."""

    def __init__(self, dump1_path: str | Path, dump2_path: str | Path, *, strict_sync: bool = True, reader_on_error: str = "skip") -> None:
        self.r1 = LAMMPSDumpReader(dump1_path, on_error=reader_on_error)
        self.r2 = LAMMPSDumpReader(dump2_path, on_error=reader_on_error)
        self.strict = strict_sync

    def __iter__(self) -> Iterator[GibbsFrame]:
        if not self.strict:
            i1, i2 = iter(self.r1), iter(self.r2)
            while True:
                try:
                    f1 = next(i1); f2 = next(i2)
                except StopIteration:
                    return
                yield GibbsFrame(step=f1.step, box1=f1.box, ids1=f1.ids, pos1=f1.pos,
                                  box2=f2.box, ids2=f2.ids, pos2=f2.pos)
        i1, i2 = iter(self.r1), iter(self.r2)
        try:
            f1 = next(i1); f2 = next(i2)
        except StopIteration:
            return
        while True:
            if f1.step == f2.step:
                yield GibbsFrame(step=f1.step, box1=f1.box, ids1=f1.ids, pos1=f1.pos,
                                  box2=f2.box, ids2=f2.ids, pos2=f2.pos)
                try:
                    f1 = next(i1); f2 = next(i2)
                except StopIteration:
                    return
            elif f1.step < f2.step:
                try:
                    f1 = next(i1)
                except StopIteration:
                    return
            else:
                try:
                    f2 = next(i2)
                except StopIteration:
                    return


# -----------------------------
# Thermodynamic data reader
# -----------------------------

_DATA_KV_RE = re.compile(r"\s*([A-Za-z][A-Za-z ]*[A-Za-z])\s*:\s*([^,\t]+)")

class DataFileReader:
    """Stream key:value records; malformed lines are skipped."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[DataRecord]:
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    kv = {}
                    for m in _DATA_KV_RE.finditer(line):
                        k = m.group(1).strip().lower()
                        v = m.group(2).strip().rstrip(',')
                        kv[k] = v
                    step = _to_int(kv.get("timestep"))
                    yield DataRecord(
                        step=step if step is not None else 0,
                        energy=_to_float(kv.get("energy")),
                        temperature=_to_float(kv.get("tempreture")) if kv.get("tempreture") is not None else _to_float(kv.get("temperature")),
                        pressure=_to_float(kv.get("pressure")),
                        volume=_to_float(kv.get("volume")),
                        density=_to_float(kv.get("density of particles")) if kv.get("density of particles") is not None else _to_float(kv.get("density")),
                        n_particles=_to_int(kv.get("numberofparticles")) if kv.get("numberofparticles") is not None else _to_int(kv.get("number of particles")),
                    )
                except Exception:
                    continue


# -----------------------------
# Two-box orchestrator
# -----------------------------

class GibbsTwoBoxLoader:
    """Pair two dumps; optionally attach data records with drop-on-missing policy."""

    def __init__(
        self,
        *,
        dump1_path: str | Path,
        dump2_path: str | Path,
        data1_path: str | Path | None = None,
        data2_path: str | Path | None = None,
        strict_sync: bool = True,
        skip_if_missing_data: bool = True,
        reader_on_error: str = "skip",
    ) -> None:
        self.pair = GibbsDumpPair(dump1_path, dump2_path, strict_sync=strict_sync, reader_on_error=reader_on_error)
        self.data1 = _build_data_index(data1_path) if data1_path is not None else None
        self.data2 = _build_data_index(data2_path) if data2_path is not None else None
        self.skip_if_missing_data = skip_if_missing_data

    def __iter__(self) -> Iterator[GibbsFrame]:
        for fr in self.pair:
            d1 = self.data1.get(fr.step) if self.data1 is not None else None
            d2 = self.data2.get(fr.step) if self.data2 is not None else None
            if self.skip_if_missing_data and ((self.data1 is not None and d1 is None) or (self.data2 is not None and d2 is None)):
                continue
            yield GibbsFrame(step=fr.step, box1=fr.box1, ids1=fr.ids1, pos1=fr.pos1,
                             box2=fr.box2, ids2=fr.ids2, pos2=fr.pos2,
                             data1=d1, data2=d2)


def _build_data_index(path: str | Path) -> Dict[int, DataRecord]:
    idx: Dict[int, DataRecord] = {}
    for rec in DataFileReader(path):
        idx[rec.step] = rec
    return idx


# -----------------------------
# Convenience helpers
# -----------------------------

def load_gibbs(
    *,
    dump1_path: str | Path,
    dump2_path: str | Path,
    data1_path: Optional[str | Path] = None,
    data2_path: Optional[str | Path] = None,
    strict_sync: bool = True,
    skip_if_missing_data: bool = True,
    reader_on_error: str = "skip",
) -> GibbsTwoBoxLoader:
    return GibbsTwoBoxLoader(
        dump1_path=dump1_path,
        dump2_path=dump2_path,
        data1_path=data1_path,
        data2_path=data2_path,
        strict_sync=strict_sync,
        skip_if_missing_data=skip_if_missing_data,
        reader_on_error=reader_on_error,
    )


def iter_gibbs(
    *,
    dump1_path: str | Path,
    dump2_path: str | Path,
    data1_path: Optional[str | Path] = None,
    data2_path: Optional[str | Path] = None,
    strict_sync: bool = True,
    skip_if_missing_data: bool = True,
    reader_on_error: str = "skip",
):
    return iter(load_gibbs(
        dump1_path=dump1_path,
        dump2_path=dump2_path,
        data1_path=data1_path,
        data2_path=data2_path,
        strict_sync=strict_sync,
        skip_if_missing_data=skip_if_missing_data,
        reader_on_error=reader_on_error,
    ))


def index_datafile(path: str | Path) -> Dict[int, DataRecord]:
    idx: Dict[int, DataRecord] = {}
    for rec in DataFileReader(path):
        idx[rec.step] = rec
    return idx
