from __future__ import annotations
"""
Gibbs analysis utilities — publication‑ready plots for reduced variables
----------------------------------------------------------------------
This module computes and visualizes the reduced variables for two‑box Gibbs
ensemble simulations:
    x = v_local / v_global
    y = n_local / n_global

Features
- Streaming computation from paired dumps via `GibbsTwoBoxLoader` (low memory).
- Robust to 2D (uses area Lx*Ly when Lz≈0); prefers data‑file Volume if present.
- Publication‑ready scatter plotting with consistent styling (fonts, sizes,
  grid, guides, axis limits), light/dark themes, and optional saving.

Public API
    collect_reduced_nv_for_folder(folder, max_frames=None) -> (X, Y)
    collect_reduced_nv_across(root, pattern, max_frames_per_folder=None) -> (X, Y)
    set_plot_style(theme='light', *, base_fontsize=11) -> None
    plot_reduced_nv_scatter(X, Y, *, ax=None, color='#1f77b4', s=10, alpha=0.7,
                            show_guides=True, annotate=True, title=None) -> Axes
    analyze_and_plot_reduced_nv(root, pattern, *, max_frames_per_folder=None,
                                theme='light', save=None, dpi=300, **plot_kwargs) -> Axes

Notes
- Deterministic: no randomness; identical inputs produce identical figures.
- No I/O beyond optional figure saving; callers control paths.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .gibbs_io import GibbsTwoBoxLoader

# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------

def _box_measure(Lx: float, Ly: float, Lz: float) -> float:
    """Geometric measure: area for 2D (Lz≈0), else volume for 3D."""
    if Lz == 0.0 or abs(Lz) < 1e-12:
        return float(Lx) * float(Ly)
    return float(Lx) * float(Ly) * float(Lz)


def _choose_volume(data_rec, box) -> float:
    """Prefer data‑file volume if present; otherwise use geometric measure."""
    if data_rec is not None and data_rec.volume is not None:
        try:
            return float(data_rec.volume)
        except Exception:
            pass
    return _box_measure(box.Lx, box.Ly, box.Lz)


def collect_reduced_nv_for_folder(
    folder: Path | str,
    *,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reduced (v_local/v_global, n_local/n_global) for all frames in one folder.

    Required files in `folder`:
      particle_positions_1.xyz, particle_positions_2.xyz
      simulation_data_1.dat (optional), simulation_data_2.dat (optional)

    Returns
    -------
    X : np.ndarray, shape (2*T,)
        Concatenated v_local/v_global for both boxes over T frames.
    Y : np.ndarray, shape (2*T,)
        Concatenated n_local/n_global for both boxes over T frames.
    """
    folder = Path(folder)
    dump1 = folder / "particle_positions_1.xyz"
    dump2 = folder / "particle_positions_2.xyz"
    data1 = folder / "simulation_data_1.dat"
    data2 = folder / "simulation_data_2.dat"

    if not dump1.exists() or not dump2.exists():
        raise FileNotFoundError(f"Missing dump files in {folder}")

    loader = GibbsTwoBoxLoader(
        dump1_path=dump1,
        dump2_path=dump2,
        data1_path=data1 if data1.exists() else None,
        data2_path=data2 if data2.exists() else None,
        strict_sync=True,
        skip_if_missing_data=True,   # drop steps without both data records if provided
        reader_on_error="skip",     # skip malformed dump frames
    )

    x_list: list[float] = []
    y_list: list[float] = []
    for i, fr in enumerate(loader):
        N1, N2 = fr.pos1.shape[0], fr.pos2.shape[0]
        V1 = _choose_volume(fr.data1, fr.box1)
        V2 = _choose_volume(fr.data2, fr.box2)

        Vg = V1 + V2
        Ng = N1 + N2
        if Vg <= 0:
            # Skip pathological frames deterministically
            continue
        ng = Ng / Vg

        for N, V in ((N1, V1), (N2, V2)):
            n_local = N / V
            x_list.append(V / Vg)
            y_list.append(n_local / ng)

        if max_frames is not None and (i + 1) >= max_frames:
            break

    return np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=float)


def collect_reduced_nv_across(
    root: Path | str,
    pattern: str = "Medium_sim_density*_f*_l*_*/",
    *,
    max_frames_per_folder: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate reduced variables across many simulation folders."""
    root = Path(root)
    folders = sorted(root.glob(pattern))
    if not folders:
        raise FileNotFoundError(f"No folders match pattern '{pattern}' under {root}")

    X_all, Y_all = [], []
    for folder in folders:
        x, y = collect_reduced_nv_for_folder(folder, max_frames=max_frames_per_folder)
        if x.size:
            X_all.append(x)
            Y_all.append(y)
    if not X_all:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    return np.concatenate(X_all), np.concatenate(Y_all)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

def set_plot_style(*, theme: str = "light", base_fontsize: int = 11) -> None:
    """Apply a clean, publication‑ready Matplotlib style.

    Parameters
    ----------
    theme : {'light','dark'}
    base_fontsize : int
    """
    if theme not in {"light", "dark"}:
        raise ValueError("theme must be 'light' or 'dark'")

    fc = "#ffffff" if theme == "light" else "#121212"
    tc = "#222222" if theme == "light" else "#e5e5e5"
    grid_c = "#cfcfcf" if theme == "light" else "#333333"

    mpl.rcParams.update({
        "figure.facecolor": fc,
        "axes.facecolor": fc,
        "savefig.facecolor": fc,
        "text.color": tc,
        "axes.labelcolor": tc,
        "axes.edgecolor": tc,
        "axes.titlecolor": tc,
        "xtick.color": tc,
        "ytick.color": tc,
        "grid.color": grid_c,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.frameon": False,
        "figure.dpi": 120,
    })

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_reduced_nv_scatter(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    color: str = "#1f77b4",
    s: float = 10.0,
    alpha: float = 0.7,
    show_guides: bool = True,
    annotate: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """Scatter plot of reduced density vs reduced volume (same color for both boxes)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sc = ax.scatter(X, Y, s=s, alpha=alpha, linewidths=0, color=color)

    ax.set_xlabel(r"$v_{\mathrm{local}}/v_{\mathrm{global}}$")
    ax.set_ylabel(r"$n_{\mathrm{local}}/n_{\mathrm{global}}$")
    if title:
        ax.set_title(title)

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    if show_guides:
        ax.axhline(1.0, color="#888888", lw=0.9, alpha=0.8)
        ax.axvline(0.5, color="#888888", lw=0.8, alpha=0.6)

    if annotate:
        ax.annotate("equal density", xy=(ax.get_xlim()[1]*0.98, 1.0), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom", color="#666666")
        ax.annotate("equal volume", xy=(0.5, ax.get_ylim()[1]*0.02), xytext=(4, 4),
                    textcoords="offset points", ha="left", va="bottom", color="#666666")

    ax.tick_params(direction="out", length=4, width=0.8)
    return ax


def analyze_and_plot_reduced_nv(
    root: Path | str,
    pattern: str = "Medium_sim_density*_f*_l*_*/",
    *,
    max_frames_per_folder: Optional[int] = None,
    theme: str = "light",
    save: Optional[Path | str] = None,
    dpi: int = 300,
    **plot_kwargs,
) -> plt.Axes:
    """Aggregate reduced variables across folders and plot with styling.

    If `save` is given, the figure is saved to that path (format inferred).
    """
    set_plot_style(theme=theme)
    X, Y = collect_reduced_nv_across(root, pattern, max_frames_per_folder=max_frames_per_folder)
    ax = plot_reduced_nv_scatter(X, Y, **plot_kwargs)
    if save is not None:
        ax.figure.savefig(save, dpi=dpi, bbox_inches="tight")
    return ax

# -----------------------------------------------------------------------------
# Coexistence detection (bimodality in local densities)
# -----------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass(frozen=True)
class CoexistenceResult:
    exists: bool                    # True if two separated peaks detected
    peaks: Tuple[float, ...]        # density values at peaks (low, high) if exists else ()
    heights: Tuple[float, ...]      # corresponding histogram heights
    bin_centers: np.ndarray         # histogram x-axis (centers)
    counts_smooth: np.ndarray       # smoothed histogram counts used for detection


def _gaussian_kernel_1d(sigma_bins: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k


def _smooth_counts(counts: np.ndarray, sigma_bins: float = 1.0) -> np.ndarray:
    if sigma_bins <= 0:
        return counts.astype(float)
    radius = max(1, int(3.0 * sigma_bins))
    k = _gaussian_kernel_1d(sigma_bins, radius)
    return np.convolve(counts, k, mode="same")


def collect_local_densities_for_folder(
    folder: Path | str,
    *,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Return 1D array of local densities rho=N/V for both boxes over frames in a folder."""
    folder = Path(folder)
    dump1 = folder / "particle_positions_1.xyz"
    dump2 = folder / "particle_positions_2.xyz"
    data1 = folder / "simulation_data_1.dat"
    data2 = folder / "simulation_data_2.dat"

    if not dump1.exists() or not dump2.exists():
        raise FileNotFoundError(f"Missing dump files in {folder}")

    loader = GibbsTwoBoxLoader(
        dump1_path=dump1,
        dump2_path=dump2,
        data1_path=data1 if data1.exists() else None,
        data2_path=data2 if data2.exists() else None,
        strict_sync=True,
        skip_if_missing_data=True,
        reader_on_error="skip",
    )

    rhos: list[float] = []
    for i, fr in enumerate(loader):
        V1 = _choose_volume(fr.data1, fr.box1)
        V2 = _choose_volume(fr.data2, fr.box2)
        if V1 > 0:
            rhos.append(fr.pos1.shape[0] / V1)
        if V2 > 0:
            rhos.append(fr.pos2.shape[0] / V2)
        if max_frames is not None and (i + 1) >= max_frames:
            break
    return np.asarray(rhos, dtype=float)


def collect_local_densities_for_pattern(
    root: Path | str,
    pattern: str,
    *,
    max_frames_per_folder: Optional[int] = None,
) -> np.ndarray:
    """Aggregate local densities rho=N/V across all folders matching `pattern`.

    Use a pattern that selects a fixed (f, density, lambda) while varying the
    sample suffix (e.g., `_0`..`_4`).
    """
    root = Path(root)
    folders = sorted(root.glob(pattern))
    if not folders:
        raise FileNotFoundError(f"No folders match pattern '{pattern}' under {root}")
    all_rhos = []
    for folder in folders:
        rho = collect_local_densities_for_folder(folder, max_frames=max_frames_per_folder)
        if rho.size:
            all_rhos.append(rho)
    if not all_rhos:
        return np.empty((0,), dtype=float)
    return np.concatenate(all_rhos)


def estimate_coexistence_from_rho(
    rho: np.ndarray,
    *,
    bins: int | str = "fd",
    sigma_bins: float = 1.0,
    min_rel_height: float = 0.10,
    min_separation_frac: float = 0.02,
    bin_width_frac: Optional[float] = 0.002,  # <--- NEW
) -> CoexistenceResult:
    """
    Detect bimodality (two peaks) in the rho distribution.

    bin_width_frac : float, optional
        If set, fixes histogram bin width to this fraction of the data range
        (e.g., 0.002 means ~500 bins over the range). Overrides 'bins' when
        the resulting number of bins > 50.
    """
    if rho.size == 0:
        return CoexistenceResult(False, tuple(), tuple(), np.empty(0), np.empty(0))

    # determine bins
    if bin_width_frac is not None and np.ptp(rho) > 0:
        width = bin_width_frac * np.ptp(rho)
        n_bins = max(50, int(np.ptp(rho) / width))
        bins = n_bins

    counts, edges = np.histogram(rho, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = _smooth_counts(counts.astype(float), sigma_bins=sigma_bins)

    if y.max() <= 0:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    # simple local-maximum finder
    locmax = []  # (idx, height)
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            locmax.append((i, y[i]))
    if not locmax:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    # filter by relative height
    hmax = max(h for _, h in locmax)
    locmax = [(i, h) for (i, h) in locmax if h >= min_rel_height * hmax]
    if len(locmax) < 2:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    # pick two best-separated strong peaks
    # enforce min separation in x units
    xmin, xmax = centers[0], centers[-1]
    min_sep = min_separation_frac * (xmax - xmin)

    # rank by height
    locmax.sort(key=lambda t: t[1], reverse=True)
    selected = []
    for idx, h in locmax:
        if not selected:
            selected.append((idx, h))
        else:
            # check separation from existing selections
            if all(abs(centers[idx] - centers[j]) >= min_sep for j, _ in selected):
                selected.append((idx, h))
        if len(selected) == 2:
            break

    if len(selected) < 2:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    sel_sorted = sorted(selected, key=lambda t: centers[t[0]])
    peaks = (centers[sel_sorted[0][0]], centers[sel_sorted[1][0]])
    heights = (sel_sorted[0][1], sel_sorted[1][1])
    return CoexistenceResult(True, peaks, heights, centers, y)


def plot_density_histogram_with_peaks(
    rho: np.ndarray,
    res: CoexistenceResult,
    *,
    ax: Optional[plt.Axes] = None,
    bins: int | str = "fd",
    color: str = "#1f77b4",
    edgecolor: str = "white",
    alpha: float = 0.65,
) -> plt.Axes:
    """Plot rho histogram and overlay detected peaks (if any)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
    counts, edges, patches = ax.hist(rho, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # overlay smoothed counts used for detection (scaled to histogram bin width)
    width = edges[1] - edges[0]
    y = res.counts_smooth
    if y.size == centers.size:
        ax.plot(centers, y, color="#333333", lw=1.2, alpha=0.9)
    if res.exists:
        ax.axvline(res.peaks[0], color="#d62728", lw=1.5)
        ax.axvline(res.peaks[1], color="#d62728", lw=1.5)
        ax.annotate(f"rho₁={res.peaks[0]:.4f}", xy=(res.peaks[0], ax.get_ylim()[1]*0.9),
                    xytext=(5, 5), textcoords="offset points", color="#d62728")
        ax.annotate(f"rho₂={res.peaks[1]:.4f}", xy=(res.peaks[1], ax.get_ylim()[1]*0.8),
                    xytext=(5, 5), textcoords="offset points", color="#d62728")
    ax.set_xlabel(r"$rho$ (local number density)")
    ax.set_ylabel("count")
    ax.set_title("Local density distribution")
    ax.grid(True, alpha=0.25)
    return ax

# -----------------------------------------------------------------------------
# Coexistence detection with reference density constraint
# -----------------------------------------------------------------------------

def _choose_bins_from_fraction(x: np.ndarray, *, bin_width_frac: Optional[float], min_bins: int = 50) -> int | str:
    if bin_width_frac is None or x.size == 0:
        return "fd"
    rng = float(np.max(x) - np.min(x))
    if rng <= 0:
        return max(min_bins, 1)
    # n_bins ~ 1 / bin_width_frac, but ensure at least min_bins
    n_bins = max(min_bins, int(round(1.0 / bin_width_frac)))
    return n_bins

def estimate_coexistence_from_rho_ref(
    rho: np.ndarray,
    ref_density: float,
    *,
    bins: int | str | None = None,
    bin_width_frac: Optional[float] = 0.0015,  # ~0.15% of range per bin (fine)
    sigma_bins: float = 1.2,
    min_rel_height: float = 0.06,
    min_separation_frac: float = 0.01,
) -> CoexistenceResult:
    """Detect two peaks such that one lies below and one above `ref_density`."""
    if rho.size == 0 or not np.isfinite(ref_density):
        return CoexistenceResult(False, tuple(), tuple(), np.empty(0), np.empty(0))

    # choose bins
    if bins is None:
        bins = _choose_bins_from_fraction(rho, bin_width_frac=bin_width_frac)
    counts, edges = np.histogram(rho, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = _smooth_counts(counts.astype(float), sigma_bins=sigma_bins)

    if y.size < 3 or y.max() <= 0:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    # local maxima
    locmax = [(i, y[i]) for i in range(1, len(y) - 1) if (y[i] > y[i - 1] and y[i] >= y[i + 1])]
    if not locmax:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    # height filter
    hmax = max(h for _, h in locmax)
    locmax = [(i, h) for (i, h) in locmax if h >= min_rel_height * hmax]
    if not locmax:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    # strongest peak below and above the reference
    below = [(i, h) for (i, h) in locmax if centers[i] < ref_density -0.02]
    above = [(i, h) for (i, h) in locmax if centers[i] > ref_density + 0.02]
    if not below or not above:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    i_lo, h_lo = max(below, key=lambda t: t[1])
    i_hi, h_hi = max(above, key=lambda t: t[1])

    # separation threshold
    xmin, xmax = centers[0], centers[-1]
    min_sep = min_separation_frac * (xmax - xmin)
    if abs(centers[i_hi] - centers[i_lo]) < min_sep:
        return CoexistenceResult(False, tuple(), tuple(), centers, y)

    return CoexistenceResult(True, (centers[i_lo], centers[i_hi]), (h_lo, h_hi), centers, y)

def plot_density_detection(
    rho: np.ndarray,
    res: CoexistenceResult,
    *,
    ref_density: Optional[float] = None,
    bins: int | str | None = None,
    bin_width_frac: Optional[float] = 0.0015,
    color: str = "#1f77b4",
    smooth_color: str = "#333333",
    peak_color: str = "#d62728",
    alpha: float = 0.6,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Histogram + smoothed curve + reference line + peak markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))

    if bins is None:
        bins = _choose_bins_from_fraction(rho, bin_width_frac=bin_width_frac)
    counts, edges, _ = ax.hist(rho, bins=bins, color=color, edgecolor="white", alpha=alpha)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # display smoothing (align with detection if possible)
    y = res.counts_smooth
    if y.size != centers.size:
        y = _smooth_counts(counts.astype(float), sigma_bins=1.2)
    ax.plot(centers, y, color=smooth_color, lw=1.2, alpha=0.9)

    if ref_density is not None and np.isfinite(ref_density):
        ax.axvline(ref_density, color="#666", lw=1.0, ls="--", alpha=0.9)
        ax.annotate(r"$\rho_\mathrm{ref}$", xy=(ref_density, ax.get_ylim()[1]*0.95),
                    xytext=(6, 4), textcoords="offset points", color="#666")

    if res.exists:
        ax.axvline(res.peaks[0], color=peak_color, lw=1.6)
        ax.axvline(res.peaks[1], color=peak_color, lw=1.6)
        ax.annotate(f"ρ₁={res.peaks[0]:.5f}", xy=(res.peaks[0], ax.get_ylim()[1]*0.85),
                    xytext=(6, 4), textcoords="offset points", color=peak_color)
        ax.annotate(f"ρ₂={res.peaks[1]:.5f}", xy=(res.peaks[1], ax.get_ylim()[1]*0.75),
                    xytext=(6, 4), textcoords="offset points", color=peak_color)

    ax.set_xlabel(r"$\rho$ (local number density)")
    ax.set_ylabel("count")
    ax.set_title("Local density distribution with detection")
    ax.grid(True, alpha=0.25)
    return ax

def parse_density_from_name(name: str) -> Optional[float]:
    for part in name.split("_"):
        if part.startswith("density"):
            try:
                return float(part.replace("density", ""))
            except ValueError:
                return None
    return None

def detect_coexistence_for_pattern(
    root: Path | str,
    pattern: str,
    *,
    ref_density: Optional[float] = None,
    max_frames_per_folder: Optional[int] = None,
    bin_width_frac: Optional[float] = 0.0015,
    sigma_bins: float = 1.2,
    min_rel_height: float = 0.06,
    min_separation_frac: float = 0.01,
) -> tuple[CoexistenceResult, float | None, np.ndarray]:
    """Aggregate rho over folders matching pattern and detect coexistence."""
    rho = collect_local_densities_for_pattern(root, pattern, max_frames_per_folder=max_frames_per_folder)
    if rho.size == 0:
        return CoexistenceResult(False, tuple(), tuple(), np.empty(0), np.empty(0)), ref_density, rho

    if ref_density is None:
        folders = sorted(Path(root).glob(pattern))
        if folders:
            ref_density = parse_density_from_name(folders[0].name)

    res = estimate_coexistence_from_rho_ref(
        rho,
        ref_density if ref_density is not None else float(np.median(rho)),
        bins=None,
        bin_width_frac=bin_width_frac,
        sigma_bins=sigma_bins,
        min_rel_height=min_rel_height,
        min_separation_frac=min_separation_frac,
    )
    return res, ref_density, rho
