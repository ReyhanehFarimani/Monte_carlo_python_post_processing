import pandas as pd
import re
from gcmc_post_processing.data_loader import load_txt_data, load_txt_data_old  # Assuming this function exists
import numpy as np


def extract_mu_and_temperature(filename):
    """
    Extracts mu and temperature from a filename by looking for floating-point numbers,
    including handling negative values.

    Parameters:
    - filename: The filename to extract from.

    Returns:
    - mu: The first floating-point number found in the filename.
    - temperature: The second floating-point number found in the filename.
    """
    # Regular expression to find all floating-point numbers including negative numbers
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", filename)
    
    if len(numbers) < 2:
        raise ValueError(f"Filename {filename} does not contain enough numeric values to extract mu and temperature.")
    
    # Convert the found numbers to floats and assume the first is mu and the second is temperature
    mu = float(numbers[0])
    temperature = float(numbers[1])
    
    return mu, temperature
def read_simulation_input_Old(input_file):
    """
    Reads the simulation input file and extracts relevant parameters.
    
    Parameters:
    - input_file: Path to the input.txt file.
    
    Returns:
    - params: Dictionary containing simulation parameters.
    """
    params = {}
    
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue  # Skip lines that don't have key-value pairs
            try:
                if "mu" in line:
                    params['mu'] = float(parts[-1])
                elif "f" in line:
                    params['f'] = float(parts[-1])
                elif "boxLengthX" in line:
                    params['boxLengthX'] = float(parts[-1])
                elif "boxLengthY" in line:
                    params['boxLengthY'] = float(parts[-1])
                elif "temperature" in line:
                    params['T'] = float(parts[-1])
                elif "kappa" in line:
                    params['kappa'] = float(parts[-1])
            except ValueError:
                print(f"Skipping line due to conversion error: {line}")
                pass
    
    return params
def read_simulation_input(input_file):
    """
    Reads the simulation input file and extracts relevant parameters.
    
    Parameters:
    - input_file: Path to the input.txt file.
    
    Returns:
    - params: Dictionary containing simulation parameters.
    """
    params = {}
    
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue  # Skip lines that don't have key-value pairs
            # try:

            try:
                if "mu" in line:
                    params['mu'] = float(parts[-1])
                elif "f" in line:
                    params['f'] = float(parts[-1])
                elif "Lx" in line:
                    params['boxLengthX'] = float(parts[-1])
                elif "Ly" in line:
                    params['boxLengthY'] = float(parts[-1])
                elif "T" in line:
                    params['T'] = float(parts[-1])
                elif "kappa" in line:
                    params['kappa'] = float(parts[-1])
                elif "f" in line:
                    params['f'] = float(parts[-1])
                elif "boxLengthX" in line:
                    params['boxLengthX'] = float(parts[-1])
                elif "boxLengthY" in line:
                    params['boxLengthY'] = float(parts[-1])
                elif "temperature" in line:
                    params['T'] = float(parts[-1])
                elif "kappa" in line:
                    params['kappa'] = float(parts[-1])
            except ValueError:
                print(f"Skipping line due to conversion error: {line}")
                pass
    
    
    return params


def process_simulation_data(data_files, input_files, lag):
    """
    Process simulation data from a list of files, extracting mu and temperature from filenames.

    Parameters:
    - data_files: List of file paths to process.
    - box_area: Area of the simulation box.

    Returns:
    - detailed_df: DataFrame containing detailed data from all simulations.
    - avg_df: DataFrame containing averaged data from all simulations.
    """
    detailed_records = []
    avg_records = []

    for filename, input_file in zip(data_files, input_files):
        # Extract mu and temperature from the filename
        # try:
        params = read_simulation_input(input_file)

        f = params['f']
        mu = params['mu']
        l = -(params['kappa'] - 6.56 )/7.71
        if (l==6.56/7.71):
            l = 0
        box_area = params['boxLengthX'] * params['boxLengthY']
        temperature = params['T']
        # Load simulation data
        try:
            _, num_particles, pressures, energies = load_txt_data(filename, 1000)
        except:
            _, num_particles, pressures, energies = load_txt_data_old(filename, 1000)
        print(pressures)
        num_particles = num_particles[lag:]
        pressures = pressures[lag:]
        energies = energies[lag:]
        if len(pressures) > 0:
            print('1')
            sim_avgN = np.mean(num_particles)
            avg_pressure = np.mean(pressures)
            avg_energy = np.mean(energies)
            stddevN = np.std(num_particles) / np.sqrt(len(num_particles))
            stddevP = np.std(pressures) / np.sqrt(len(pressures))
            stddevE = np.std(energies) / np.sqrt(len(energies))

            # Record detailed data for each timestep
            for n, p, e in zip(num_particles, pressures, energies):
                detailed_records.append({
                    'mu': mu,
                    'temperature': temperature,
                    'num_particles': n,
                    'density': n / box_area,
                    'pressure': p,
                    'energy': e,
                    'f': f,
                    'l': l,
                    'bx_area' : box_area
                })

            # Record averaged data for the entire file
            avg_records.append({
                'mu': mu,
                'temperature': temperature,
                'l': l,
                'sim_avgN': sim_avgN,
                'avg_pressure': avg_pressure,
                'avg_energy': avg_energy,
                'avg_density': sim_avgN/box_area,
                'stddevN': stddevN,
                'stddevP': stddevP,
                'stddevE': stddevE,
                'stddevrho': stddevN/box_area,
                'f': f,
                'bx_area' : box_area
            })
    
    detailed_df = pd.DataFrame(detailed_records)
    avg_df = pd.DataFrame(avg_records)
    # avg_df.sort_values(by=['mu'], inplace=True)

    return detailed_df, avg_df


def bin_data_by_density(df, density_bins, tolerance=0.01):
    binned_data = []
    for density_bin in density_bins:
        bin_indices = np.abs(df['density'] - density_bin) < tolerance
        if bin_indices.any():
            avg_pressure = df[bin_indices]['pressure'].mean()
            stddev_pressure = df[bin_indices]['pressure'].std() / np.sqrt(bin_indices.sum())
            binned_data.append({
                'density_bin': density_bin,
                'avg_pressure': avg_pressure,
                'stddev_pressure': stddev_pressure,
                'count': bin_indices.sum()
            })
    return pd.DataFrame(binned_data)


import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

@dataclass(frozen=True)
class StructureFactorResult:
    q: np.ndarray            # (Nbins,) radial wavenumbers
    Sq: np.ndarray           # (Nbins,) radial S(q)
    kvecs: np.ndarray        # (M,2) evaluated 2D wavevectors (cartesian)
    Sk: np.ndarray           # (M,) S(k) at those k-vectors (before binning)
    meta: Dict               # info: box, density, bins, etc.

def _generate_k_vectors(Lx: float, Ly: float, kmax: float) -> np.ndarray:
    """
    Generate all allowed reciprocal vectors k = (2π nx/Lx, 2π ny/Ly)
    with |k| <= kmax, excluding k=0.
    """
    kx0 = 2.0 * np.pi / Lx
    ky0 = 2.0 * np.pi / Ly
    # conservative index limits so that |k| <= kmax
    nx_max = int(np.floor(kmax / kx0))
    ny_max = int(np.floor(kmax / ky0))
    nxs = np.arange(-nx_max, nx_max + 1, dtype=int)
    nys = np.arange(-ny_max, ny_max + 1, dtype=int)
    grid = np.stack(np.meshgrid(nxs, nys, indexing="ij"), axis=-1).reshape(-1, 2)
    kvecs = np.empty_like(grid, dtype=float)
    kvecs[:, 0] = grid[:, 0] * kx0
    kvecs[:, 1] = grid[:, 1] * ky0
    # remove k=0
    mask = ~((grid[:, 0] == 0) & (grid[:, 1] == 0))
    kvecs = kvecs[mask]
    # enforce |k|<=kmax
    kmag = np.linalg.norm(kvecs, axis=1)
    return kvecs[kmag <= kmax]

def _batched_rho_k(positions: np.ndarray, kvecs: np.ndarray, batch: int = 4096) -> np.ndarray:
    """
    Compute rho_k = sum_j exp(i k·r_j) for batches of k to stay memory-safe.
    """
    N = positions.shape[0]
    M = kvecs.shape[0]
    rho_k = np.empty(M, dtype=np.complex128)
    for start in range(0, M, batch):
        stop = min(start + batch, M)
        kb = kvecs[start:stop]                         # (B,2)
        phases = positions @ kb.T                      # (N,B)
        # sum over particles -> (B,)
        block = np.exp(1j * phases).sum(axis=0)
        rho_k[start:stop] = block
    # normalize as S(k) = |rho_k|^2 / N
    return rho_k

def radial_bin(values: np.ndarray, radii: np.ndarray, nbins: int, rmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic radial averaging <values> over |r| with linear bins.
    """
    if rmax is None:
        rmax = float(np.max(radii))
    edges = np.linspace(0.0, rmax, nbins + 1)
    idx = np.digitize(radii, edges) - 1
    valid = (idx >= 0) & (idx < nbins)
    sums = np.bincount(idx[valid], weights=values[valid].real, minlength=nbins).astype(float)
    counts = np.bincount(idx[valid], minlength=nbins).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(counts > 0, sums / counts, 0.0)
    # bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, avg

def structure_factor(
    positions: np.ndarray,
    box: Tuple[float, float],
    kmax: Optional[float] = None,
    nbins: int = 200,
    batch: int = 4096,
) -> StructureFactorResult:
    """
    Compute static structure factor S(k) on allowed reciprocal vectors of a 2D periodic box,
    and its radial average S(q).

    Parameters
    ----------
    positions : (N,2) array, particle positions in [0,Lx)×[0,Ly)
    box : (Lx,Ly)
    kmax : float, optional
        Maximum |k| to evaluate. Default: min(20 * 2π/L, π/a_nyquist) ~ 2π * nbins / L.
        If None, use kmax = π * min(nx/Lx, ny/Ly) with nx=ny=nbins.
    nbins : int, number of radial bins for S(q).
    batch : int, number of k-vectors per batch in the plane-wave sum.

    Returns
    -------
    StructureFactorResult
    """
    positions = np.ascontiguousarray(positions, dtype=float)
    Lx, Ly = float(box[0]), float(box[1])
    N = positions.shape[0]
    area = Lx * Ly
    rho = N / area

    # Default kmax: use up to ~nbins fundamental spacings (safe and dense)
    if kmax is None:
        dkx = 2.0 * np.pi / Lx
        dky = 2.0 * np.pi / Ly
        kmax = float(nbins) * min(dkx, dky)

    kvecs = _generate_k_vectors(Lx, Ly, kmax)  # (M,2)
    # exact plane-wave sum on the allowed k grid
    rk = _batched_rho_k(positions, kvecs, batch=batch)  # (M,)
    Sk = (rk.real**2 + rk.imag**2) / N                  # (M,)

    kmag = np.linalg.norm(kvecs, axis=1)
    q, Sq = radial_bin(Sk, kmag, nbins=nbins, rmax=kmax)

    meta = dict(Lx=Lx, Ly=Ly, N=N, rho=rho, kmax=kmax, nbins=nbins, note="plane-wave exact S(k) on PBC grid")
    return StructureFactorResult(q=q, Sq=Sq, kvecs=kvecs, Sk=Sk, meta=meta)

def structure_factor_from_gr(
    r: np.ndarray, g_r: np.ndarray, density: float, q: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    S(q) from g(r) via Fourier–Bessel transform in 2D:
        S(q) = 1 + 2πρ ∫_0^∞ [g(r)-1] r J0(q r) dr
    Deterministic trapezoidal integration.

    Parameters
    ----------
    r : (M,) radii, uniform spacing recommended
    g_r : (M,) pair correlation
    density : float, number density
    q : (K,), optional q-grid. If None, uses K=len(r), q_k = k * (2π)/(r_max)

    Returns
    -------
    q, Sq : arrays
    """
    from scipy.special import j0

    r = np.asarray(r, dtype=float)
    g_r = np.asarray(g_r, dtype=float)
    dr = float(r[1] - r[0])
    if q is None:
        q = np.linspace(0.0, 2.0 * np.pi / max(dr, 1e-12), num=len(r))
    Sq = np.empty_like(q)
    fr = (g_r - 1.0) * r
    for i, qv in enumerate(q):
        Sq[i] = 1.0 + 2.0 * np.pi * density * np.trapz(fr * j0(qv * r), dx=dr)
    return q, Sq



from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

@dataclass(frozen=True)
class StructureFactor2DResult:
    kx: np.ndarray            # (Nkx,) 1D kx grid (2π n / Lx)
    ky: np.ndarray            # (Nky,) 1D ky grid (2π n / Ly)
    S: np.ndarray             # (Nkx, Nky) S(kx,ky)
    meta: Dict                # info: Lx, Ly, N, rho, nmax, etc.

def _rho_k_batched(positions: np.ndarray, klist: np.ndarray, batch: int = 4096) -> np.ndarray:
    """
    Compute rho_k = sum_j exp(i k · r_j) for flattened list of k-vectors (M,2),
    batching k’s to keep memory bounded.
    """
    M = klist.shape[0]
    rho = np.empty(M, dtype=np.complex128)
    for s in range(0, M, batch):
        e = min(s + batch, M)
        kb = klist[s:e]                              # (B,2)
        phases = positions @ kb.T                    # (N,B)
        rho[s:e] = np.exp(1j * phases).sum(axis=0)   # (B,)
    return rho

def structure_factor_2d(
    positions: np.ndarray,
    box: Tuple[float, float],
    nmax: int = 64,
    exclude_k0: bool = True,
    batch: int = 4096,
) -> StructureFactor2DResult:
    """
    Compute the 2D static structure factor S(kx,ky) on the **allowed PBC wavevectors**:
        kx = 2π n_x / Lx,  ky = 2π n_y / Ly,  n_x,n_y ∈ [-nmax, …, nmax].

    S(k) = |∑_j e^{i k·r_j}|^2 / N  (k=0 optionally excluded)

    Parameters
    ----------
    positions : (N,2) array in [0,Lx)×[0,Ly)
    box       : (Lx, Ly)
    nmax      : maximum |n| for each axis (grid side = 2*nmax+1)
    exclude_k0: if True, set S(0,0) = np.nan (NVT conserved mode)
    batch     : k-batch size for the plane-wave sum

    Returns
    -------
    StructureFactor2DResult with kx, ky, S grid and metadata.
    """
    positions = np.ascontiguousarray(positions, dtype=float)
    Lx, Ly = float(box[0]), float(box[1])
    N = positions.shape[0]
    rho = N / (Lx * Ly)

    nx = np.arange(-nmax, nmax + 1, dtype=int)
    ny = np.arange(-nmax, nmax + 1, dtype=int)
    kx = (2.0 * np.pi / Lx) * nx      # (Nkx,)
    ky = (2.0 * np.pi / Ly) * ny      # (Nky,)

    # Flattened k-list with mapping (ix,iy) -> flat index
    KX, KY = np.meshgrid(kx, ky, indexing="ij")      # (Nkx,Nky)
    klist = np.column_stack([KX.ravel(), KY.ravel()])  # (M,2), M=(2nmax+1)^2

    rk = _rho_k_batched(positions, klist, batch=batch)     # (M,)
    Sk_flat = (rk.real**2 + rk.imag**2) / N
    S = Sk_flat.reshape(KX.shape)                           # (Nkx,Nky)

    if exclude_k0:
        # k=0 sits at index where kx=0 and ky=0
        iz = np.where(nx == 0)[0]
        jz = np.where(ny == 0)[0]
        if iz.size and jz.size:
            S[iz[0], jz[0]] = np.nan

    meta = dict(Lx=Lx, Ly=Ly, N=N, rho=rho, nmax=nmax,
                note="exact plane-wave S(kx,ky) on PBC grid; center is k=0")
    return StructureFactor2DResult(kx=kx, ky=ky, S=S, meta=meta)


import numpy as np
from typing import Tuple, Optional, List, Dict

def _quadratic_subpixel_refine_2d(Z: np.ndarray, dx: float, dy: float) -> Tuple[float, float]:
    """
    Refine the peak position of a 3x3 patch Z, centered at the integer maximum,
    by fitting a quadratic surface:
        f(x,y) = a x^2 + b y^2 + c x y + d x + e y + f0
    on coordinates x ∈ {-dx,0,dx}, y ∈ {-dy,0,dy}.
    Returns (x_hat, y_hat) offsets relative to the center (in same units as dx,dy).
    If fit is ill-conditioned, returns (0,0).
    """
    assert Z.shape == (3, 3)
    xs = np.array([-dx, 0.0, dx])
    ys = np.array([-dy, 0.0, dy])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # Design matrix for 9 points
    G = np.column_stack([
        X.ravel()**2,      # a
        Y.ravel()**2,      # b
        (X*Y).ravel(),     # c
        X.ravel(),         # d
        Y.ravel(),         # e
        np.ones(9),        # f0
    ])
    z = Z.ravel()
    try:
        coef, *_ = np.linalg.lstsq(G, z, rcond=None)
        a, b, c, d, e, _f0 = coef
        # Stationary point: ∂f/∂x = 2a x + c y + d = 0 ; ∂f/∂y = 2b y + c x + e = 0
        M = np.array([[2*a, c],
                      [c,   2*b]], dtype=float)
        rhs = -np.array([d, e], dtype=float)
        xy = np.linalg.solve(M, rhs)  # in units of dx,dy
        # Guard: keep within half a pixel in each direction to avoid runaway
        xh = float(np.clip(xy[0], -0.5*dx, 0.5*dx))
        yh = float(np.clip(xy[1], -0.5*dy, 0.5*dy))
        return xh, yh
    except Exception:
        return 0.0, 0.0

def extract_q_vectors_from_S2D(
    kx: np.ndarray,
    ky: np.ndarray,
    S: np.ndarray,
    *,
    topk: Optional[int] = None,
    min_prominence: float = 0.0,
    exclude_center_radius: float = 0.0,
    ring_center: Optional[float] = None,
    ring_halfwidth: Optional[float] = None,
    subpixel: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract q-vectors (kx,ky) at local maxima of S(kx,ky).

    Parameters
    ----------
    kx, ky : 1D arrays of wavevector axes (uniformly spaced).
    S      : 2D array S[kx_index, ky_index].
    topk   : If given, keep only the top-k peaks by S value (after filters).
    min_prominence : minimum S value above the local 8-neighborhood average to accept a peak.
                     Use 0 to turn off.
    exclude_center_radius : exclude peaks with |k| < this radius (to drop the central pixel/low-q).
    ring_center, ring_halfwidth : if set, only keep peaks with | |k| - ring_center | <= ring_halfwidth.
                                  Useful to isolate the first shell.
    subpixel : refine peak positions by a 3x3 quadratic fit.

    Returns
    -------
    dict with:
        "qvecs" : (M,2) array of peak vectors [ (qx, qy), ... ]
        "q"     : (M,)  magnitudes
        "S"     : (M,)  peak values at refined positions (interpolated as central pixel)
        "ij"    : (M,2) integer grid indices for reference
    """
    # Basic checks and spacings
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    S = np.asarray(S, dtype=float)
    assert S.shape == (kx.size, ky.size)
    if kx.size < 3 or ky.size < 3:
        return dict(qvecs=np.zeros((0,2)), q=np.zeros(0), S=np.zeros(0), ij=np.zeros((0,2), dtype=int))
    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])

    # Local-maximum map (8-neighborhood)
    # Avoid borders to allow 3x3 refinement
    core = S[1:-1, 1:-1]
    # Compare to neighbors
    nbrs = np.stack([
        S[0:-2, 0:-2], S[0:-2, 1:-1], S[0:-2, 2:  ],
        S[1:-1, 0:-2],                 S[1:-1, 2:  ],
        S[2:  , 0:-2], S[2:  , 1:-1], S[2:  , 2:  ],
    ], axis=0)  # (8, Nkx-2, Nky-2)
    is_peak = (core >= nbrs).all(axis=0)

    # Prominence filter (relative to neighbor mean)
    if min_prominence > 0.0:
        mean_nbr = nbrs.mean(axis=0)
        is_peak &= (core - mean_nbr) >= min_prominence

    pi, pj = np.where(is_peak)
    # shift to full indices
    pi += 1; pj += 1

    if pi.size == 0:
        return dict(qvecs=np.zeros((0,2)), q=np.zeros(0), S=np.zeros(0), ij=np.zeros((0,2), dtype=int))

    # Subpixel refinement using 3x3 patch
    qxs = []
    qys = []
    Ss = []
    ijs = []
    for ii, jj in zip(pi, pj):
        # Exclude center region if requested
        kx0 = kx[ii]; ky0 = ky[jj]
        kmag = np.hypot(kx0, ky0)
        if exclude_center_radius > 0.0 and kmag < exclude_center_radius:
            continue

        # Ring filter if requested
        if (ring_center is not None) and (ring_halfwidth is not None):
            if abs(kmag - ring_center) > ring_halfwidth:
                continue

        # Subpixel refine
        if subpixel:
            Z = S[ii-1:ii+2, jj-1:jj+2]
            dx, dy = dkx, dky
            offx, offy = _quadratic_subpixel_refine_2d(Z, dx, dy)
        else:
            offx = offy = 0.0

        qx = kx0 + offx
        qy = ky0 + offy

        qxs.append(qx)
        qys.append(qy)
        Ss.append(S[ii, jj])
        ijs.append((ii, jj))

    if not qxs:
        return dict(qvecs=np.zeros((0,2)), q=np.zeros(0), S=np.zeros(0), ij=np.zeros((0,2), dtype=int))

    qxs = np.asarray(qxs)
    qys = np.asarray(qys)
    mags = np.hypot(qxs, qys)
    Ss = np.asarray(Ss)
    ijs = np.asarray(ijs, dtype=int)

    # Sort by descending S (strongest peaks first)
    order = np.argsort(-Ss)
    qxs, qys, mags, Ss, ijs = qxs[order], qys[order], mags[order], Ss[order], ijs[order]

    # Keep top-k if requested
    if isinstance(topk, int) and topk > 0 and topk < qxs.size:
        qxs, qys, mags, Ss, ijs = qxs[:topk], qys[:topk], mags[:topk], Ss[:topk], ijs[:topk]

    qvecs = np.column_stack([qxs, qys])
    return dict(qvecs=qvecs, q=mags, S=Ss, ij=ijs)

def extract_first_shell_q_vectors(
    res2d: "StructureFactor2DResult",
    expected_q: Optional[float] = None,
    rel_halfwidth: float = 0.15,
    exclude_center_radius: float = 0.0,
    min_prominence: float = 0.0,
    topk: Optional[int] = None,
    subpixel: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper: extract q-vectors near the FIRST shell (|k| ~ q*).

    - If expected_q is provided (e.g., 4π/(√3 a) for triangular, 2π/a for square),
      uses | |k| - expected_q | <= rel_halfwidth * expected_q.
    - Otherwise, auto-detects the strongest ring by selecting peaks at the
      most frequent magnitude bin.

    Returns same dict as extract_q_vectors_from_S2D.
    """
    kx, ky, S = res2d.kx, res2d.ky, res2d.S
    if expected_q is not None:
        return extract_q_vectors_from_S2D(
            kx, ky, S,
            topk=topk,
            min_prominence=min_prominence,
            exclude_center_radius=exclude_center_radius,
            ring_center=expected_q,
            ring_halfwidth=rel_halfwidth * expected_q,
            subpixel=subpixel,
        )

    # Auto-detect ring: get all peaks (excluding center), histogram |k|
    all_peaks = extract_q_vectors_from_S2D(
        kx, ky, S,
        topk=None,
        min_prominence=min_prominence,
        exclude_center_radius=exclude_center_radius,
        ring_center=None,
        ring_halfwidth=None,
        subpixel=subpixel,
    )
    if all_peaks["q"].size == 0:
        return all_peaks

    qmag = all_peaks["q"]
    # robust binning between percentiles
    lo, hi = np.nanpercentile(qmag, [5, 95])
    bins = max(16, int(np.sqrt(qmag.size)))
    hist, edges = np.histogram(qmag[(qmag >= lo) & (qmag <= hi)], bins=bins)
    if hist.size == 0 or hist.max() == 0:
        return all_peaks
    imax = np.argmax(hist)
    q_center = 0.5 * (edges[imax] + edges[imax+1])
    q_hw = 0.5 * (edges[imax+1] - edges[imax])

    return extract_q_vectors_from_S2D(
        kx, ky, S,
        topk=topk,
        min_prominence=min_prominence,
        exclude_center_radius=exclude_center_radius,
        ring_center=q_center,
        ring_halfwidth=max(q_hw, rel_halfwidth * q_center),
        subpixel=subpixel,
    )



# --- add to statics.py ---

import numpy as np
from typing import Optional, Dict, Tuple

def _local_maxima_8nbr(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices (i,j) of 8-neighborhood local maxima in S
    (excluding the outermost 1-pixel frame).
    """
    core = S[1:-1, 1:-1]
    nbrs = np.stack([
        S[0:-2, 0:-2], S[0:-2, 1:-1], S[0:-2, 2:  ],
        S[1:-1, 0:-2],                 S[1:-1, 2:  ],
        S[2:  , 0:-2], S[2:  , 1:-1], S[2:  , 2:  ],
    ], axis=0)  # (8, Nkx-2, Nky-2)
    mask = (core >= nbrs).all(axis=0)
    pi, pj = np.where(mask)
    return pi + 1, pj + 1  # shift to full-grid indices

def extract_q_vectors_from_S2D(
    kx: np.ndarray,
    ky: np.ndarray,
    S: np.ndarray,
    *,
    expected_q: Optional[float] = None,
    rel_halfwidth: float = 0.12,
    exclude_center_radius: float = 0.0,
    min_prominence: float = 0.0,
    topk: Optional[int] = None,
    subpixel: bool = False,  # keep False for simplicity/robustness
) -> Dict[str, np.ndarray]:
    """
    Compute q-vectors (qx,qy) at peak locations of S(kx,ky).

    - If expected_q is given, only keep peaks with | |k| - expected_q | <= rel_halfwidth*expected_q.
    - exclude_center_radius removes low-|k| peaks (e.g., set to ~0.2*q*).
    - min_prominence requires S(center) - mean(neighbors) >= threshold.
    - topk keeps the strongest K peaks by S value.

    Returns dict with:
      qvecs: (M,2) [[qx,qy],...], q: |q|, S: peak height, ij: integer (i,j) indices.
    """
    kx = np.asarray(kx, float)
    ky = np.asarray(ky, float)
    S  = np.asarray(S,  float)
    assert S.shape == (kx.size, ky.size)

    # find integer-grid local maxima
    pi, pj = _local_maxima_8nbr(S)
    if pi.size == 0:
        return dict(qvecs=np.zeros((0,2)), q=np.zeros(0), S=np.zeros(0), ij=np.zeros((0,2), int))

    # optional prominence filter
    if min_prominence > 0.0:
        # neighbor mean at peaks
        mean_n = (
            S[pi-1, pj-1] + S[pi-1, pj] + S[pi-1, pj+1] +
            S[pi,   pj-1] +                S[pi,   pj+1] +
            S[pi+1, pj-1] + S[pi+1, pj] + S[pi+1, pj+1]
        ) / 8.0
        keep = (S[pi, pj] - mean_n) >= min_prominence
        pi, pj = pi[keep], pj[keep]
        if pi.size == 0:
            return dict(qvecs=np.zeros((0,2)), q=np.zeros(0), S=np.zeros(0), ij=np.zeros((0,2), int))

    qx0 = kx[pi]
    qy0 = ky[pj]
    kmag = np.hypot(qx0, qy0)

    # drop central region (small-|k|)
    if exclude_center_radius > 0.0:
        keep = kmag >= exclude_center_radius
        qx0, qy0, kmag, pi, pj = qx0[keep], qy0[keep], kmag[keep], pi[keep], pj[keep]

    # ring gate around expected_q if provided
    if expected_q is not None:
        halfw = rel_halfwidth * expected_q
        keep = np.abs(kmag - expected_q) <= halfw
        qx0, qy0, kmag, pi, pj = qx0[keep], qy0[keep], kmag[keep], pi[keep], pj[keep]

    if qx0.size == 0:
        return dict(qvecs=np.zeros((0,2)), q=np.zeros(0), S=np.zeros(0), ij=np.zeros((0,2), int))

    # sort by S height descending
    Svals = S[pi, pj]
    order = np.argsort(-Svals)
    qx0, qy0, kmag, Svals, pi, pj = qx0[order], qy0[order], kmag[order], Svals[order], pi[order], pj[order]

    # top-k
    if isinstance(topk, int) and topk > 0 and topk < qx0.size:
        qx0, qy0, kmag, Svals, pi, pj = qx0[:topk], qy0[:topk], kmag[:topk], Svals[:topk], pi[:topk], pj[:topk]

    qvecs = np.column_stack([qx0, qy0])
    return dict(qvecs=qvecs, q=kmag, S=Svals, ij=np.column_stack([pi, pj]))
# --- end patch ---
