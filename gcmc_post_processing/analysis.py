# =============================================================================
# Reyhaneh — analysis.py
# Focus: ORIENTATIONAL ORDER (2D). Translational parts are disabled for now.
# Dependencies: numpy, freud (tested with freud >= 2.13).
# =============================================================================
from typing import Sequence, Tuple, Optional, Dict
# from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import inspect
try:
    import freud
except Exception as e:
    raise ImportError("freud is required for this analysis module.") from e


# =============================================================================
# Small utilities
# =============================================================================

def _ensure_box(box_length):
    """Return (Lx, Ly) and a 2D periodic freud.box.Box with Lz=1.0."""
    if isinstance(box_length, (float, int)):
        Lx, Ly = float(box_length), float(box_length)
    else:
        Lx, Ly = float(box_length[0]), float(box_length[1])
    return (Lx, Ly), freud.box.Box(Lx=Lx, Ly=Ly, Lz=1.0, is2D=True)



def _to_3d(positions: np.ndarray) -> np.ndarray:
    """Ensure (N,3) array for freud."""
    if positions.ndim != 2:
        raise ValueError("positions must be (N,2) or (N,3).")
    if positions.shape[1] == 3:
        return positions.astype(np.float64, copy=False)
    if positions.shape[1] == 2:
        out = np.zeros((positions.shape[0], 3), dtype=np.float64)
        out[:, :2] = positions
        return out
    raise ValueError("positions must be (N,2) or (N,3).")


# =============================================================================
# RDF (kept minimal, used occasionally for diagnostics)
# =============================================================================

def compute_rdf(all_positions, box_length, r_max=None, dr=0.02):
    """
    Radial distribution function g(r) averaged over frames.

    Parameters
    ----------
    all_positions : list[np.ndarray]  # each (N,2 or 3)
    box_length : float or (2,)
    r_max : float or None
    dr : float

    Returns
    -------
    r_centers : (nbins,)
    g_avg     : (nbins,)
    """
    (Lx, Ly), box = _ensure_box(box_length)
    if r_max is None:
        r_max = 0.49 * min(Lx, Ly)
    nbins = int(max(10, np.floor(r_max / max(dr, 1e-6))))
    rdf = freud.density.RDF(bins=nbins, r_max=float(r_max))
    accum = []
    for pos in all_positions:
        pts = _to_3d(pos)
        rdf.compute((box, pts))
        accum.append(np.array(rdf.rdf, copy=True))
    g_avg = np.mean(accum, axis=0) if accum else np.zeros(nbins)
    edges = np.linspace(0.0, r_max, nbins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, g_avg


def average_rdf_over_trajectory(all_positions, box_length, dr=0.02, rcutoff=None):
    """
    Alias for compute_rdf with the same output signature used elsewhere.
    """
    r_max = rcutoff if rcutoff is not None else None
    return compute_rdf(all_positions, box_length, r_max=r_max, dr=dr)


# =============================================================================
# Voronoi neighbors
# =============================================================================

def voronoi_neighborhood_list(positions: np.ndarray, box_length):
    """
    Return freud Voronoi neighbor list for a single frame.
    """
    (Lx, Ly), box = _ensure_box(box_length)
    pts = _to_3d(positions)
    voro = freud.locality.Voronoi()
    voro.compute((box, pts))
    return voro.nlist


# =============================================================================
# Local density (simple 2D histogram; used for quick maps)
# =============================================================================

def compute_local_density(all_positions, box_length, nbins=(64, 64)):
    """
    2D number-density map averaged over frames using histogram2d.

    Returns
    -------
    x_edges, y_edges, rho_map  # rho_map shape = (ny, nx)
    """
    (Lx, Ly), _ = _ensure_box(box_length)
    nx, ny = int(nbins[0]), int(nbins[1])
    acc = np.zeros((ny, nx), dtype=np.float64)
    T = len(all_positions)
    for pos in all_positions:
        p = np.asarray(pos)
        x = np.mod(p[:, 0], Lx)
        y = np.mod(p[:, 1], Ly)
        H, xedges, yedges = np.histogram2d(y, x, bins=[ny, nx], range=[[0, Ly], [0, Lx]])
        acc += H
    rho = acc / max(T, 1) / ((Lx / nx) * (Ly / ny))
    return xedges, yedges, rho


# =============================================================================
# Hexatic order: Ψ6 series, Binder, χ6
# =============================================================================

def compute_psi6_series(all_positions, box_length, order_k=6):
    """
    Per-frame global hexatic order parameter Ψ_k (complex).

    Returns
    -------
    psi6_series     : (T,) complex
    psi6_abs_mean   : float
    psi6_abs_stderr : float
    """
    (Lx, Ly), box = _ensure_box(box_length)
    psi_list = []
    for pos in all_positions:
        pts = _to_3d(pos)
        voro = freud.locality.Voronoi()
        voro.compute((box, pts))
        op = freud.order.Hexatic(k=order_k)
        op.compute((box, pts), neighbors=voro.nlist)
        psi_list.append(np.mean(op.particle_order))
    psi6_series = np.asarray(psi_list, dtype=np.complex128)
    abs_vals = np.abs(psi6_series)
    n = len(abs_vals)
    mean = float(np.mean(abs_vals)) if n else 0.0
    stderr = float(np.std(abs_vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return psi6_series, mean, stderr


def binder_and_susceptibility(psi6_series, N_particles):
    """
    Binder U4^(6) and susceptibility χ6 from Ψ6 series.
    """
    psi = np.asarray(psi6_series, dtype=np.complex128)
    m = np.mean(psi)
    m2 = np.mean(np.abs(psi) ** 2)
    m4 = np.mean(np.abs(psi) ** 4)
    U4 = float(1.0 - m4 / max(3.0 * m2 * m2, 1e-16))
    chi6 = float(N_particles * (m2 - np.abs(m) ** 2))
    return U4, chi6


# =============================================================================
# Bond-orientational correlation g6(r)
# =============================================================================

def compute_g6_avg(all_positions, box_length, r_max, nbins=120, return_per_frame=False):
    """
    Average g6(r) over frames using freud.density.CorrelationFunction on ψ6.

    Returns
    -------
    if return_per_frame=False:
        r, g6_avg
    else:
        r, g6_avg, g6_per_frame  # (T, nbins)
    """
    (Lx, Ly), box = _ensure_box(box_length)
    r_max = float(min(r_max, 0.49 * min(Lx, Ly)))
    nbins = int(nbins)
    cf = freud.density.CorrelationFunction(bins=nbins, r_max=r_max)
    per_frame = []
    for pos in all_positions:
        pts = _to_3d(pos)
        voro = freud.locality.Voronoi()
        voro.compute((box, pts))
        op = freud.order.Hexatic(k=6)
        op.compute((box, pts), neighbors=voro.nlist)
        vals = op.particle_order
        cf.compute(system=(box, pts), values=vals, query_points=pts, query_values=vals)
        per_frame.append(np.maximum(np.real(cf.correlation), 1e-16))
    per_frame = np.asarray(per_frame, dtype=np.float64) if len(per_frame) else np.zeros((0, nbins))
    g6_avg = np.mean(per_frame, axis=0) if per_frame.size else np.zeros(nbins)
    edges = np.linspace(0.0, r_max, nbins + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    if return_per_frame:
        return r, g6_avg, per_frame
    return r, g6_avg


# =============================================================================
# Simple eta6 fit (log–log)
# =============================================================================

def fit_eta6_from_g6(r, g6, rmin=None, rmax=None):
    """
    Fit g6(r) ~ r^{-eta6} on a specified window (log–log).
    """
    r = np.asarray(r, float)
    y = np.asarray(g6, float)
    m = (r > 0) & np.isfinite(y) & (y > 0)
    if rmin is not None:
        m &= (r >= float(rmin))
    if rmax is not None:
        m &= (r <= float(rmax))
    r_fit = r[m]; y_fit = y[m]
    if len(r_fit) < 6:
        return np.nan
    A = np.vstack([np.log(r_fit), np.ones_like(r_fit)]).T
    slope, _ = np.linalg.lstsq(A, np.log(y_fit), rcond=None)[0]
    return -float(slope)


# =============================================================================
# Automatic model selection (power vs exp) — simple
# =============================================================================

@dataclass
class G6Fit:
    kind: str       # 'power' | 'exp' | 'insufficient'
    eta6: float     # valid if 'power'
    r2_power: float
    r2_exp: float
    rmin: float
    rmax: float

def fit_g6_auto(r, g6, rmin=None, rmax=None, start_bin=10, r2_margin=0.02):
    """
    Simple selector between power law and exponential on a fixed window.
    """
    r = np.asarray(r, float)
    y = np.maximum(np.asarray(g6, float), 1e-16)
    if rmin is None or rmax is None:
        # default: drop first 'start_bin' points; use tail
        idx0 = min(start_bin, len(r))
        rr, yy = r[idx0:], y[idx0:]
    else:
        mask = (r >= rmin) & (r <= rmax)
        rr, yy = r[mask], y[mask]
    if len(rr) < 10:
        return G6Fit("insufficient", np.nan, 0.0, 0.0, np.nan, np.nan)

    # power: log y vs log r
    A = np.vstack([np.log(rr), np.ones_like(rr)]).T
    sp, ip = np.linalg.lstsq(A, np.log(yy), rcond=None)[0]
    yhat_p = np.exp(ip) * rr ** sp
    r2p = 1.0 - np.sum((yy - yhat_p) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-16)

    # exp: log y vs r
    B = np.vstack([rr, np.ones_like(rr)]).T
    se, ie = np.linalg.lstsq(B, np.log(yy), rcond=None)[0]
    yhat_e = np.exp(ie + se * rr)
    r2e = 1.0 - np.sum((np.log(yy) - np.log(yhat_e)) ** 2) / max(np.sum((np.log(yy) - np.log(yy).mean()) ** 2), 1e-16)

    if r2p >= r2e + r2_margin:
        return G6Fit("power", -float(sp), float(r2p), float(r2e), float(rr.min()), float(rr.max()))
    if r2e >= r2p + r2_margin:
        return G6Fit("exp", np.nan, float(r2p), float(r2e), float(rr.min()), float(rr.max()))
    return G6Fit("insufficient", np.nan, float(r2p), float(r2e), float(rr.min()), float(rr.max()))

# =============================================================================
# Hexatic–fluid–solid detector (robust): g6 models + Ψ6 sub-block scaling
# =============================================================================
# =============================================================================
# Robust g6(r) envelope fit with hard floor on g6
# =============================================================================
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------
# analysis.py
# ---------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class G6ModelFit:
    model: str
    c_tail: float
    eta6_mean: float
    eta6_std: float
    aic_const: float
    aic_power: float
    aic_exp: float
    best_aic: float
    delta_to_next: float
    rmin: float
    rmax: float
    n_points: int
    rel_var: float
    tail_mean: float
    note: str
    idx_used: np.ndarray
    r_used: np.ndarray
    y_used: np.ndarray
    # NEW: store regression intercepts/slopes for plotting without re-fitting
    logA_power: float        # b_p  in  log g6 = -η log r + b_p  → g6 = exp(b_p) r^{-η}
    alpha_exp: float         # α    in  log g6 =  α r + b_e       → g6 = exp(α r + b_e)
    logA_exp: float          # b_e

def _g6_upper_envelope(g6_avg: np.ndarray,
                       g6_pf: Optional[np.ndarray],
                       start_bin: int,
                       q: float = 0.85,
                       roll: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(g6_avg, float)
    i0 = max(0, int(start_bin))
    idx_tail = np.arange(i0, len(y))
    if g6_pf is not None and g6_pf.ndim == 2 and g6_pf.shape[1] == len(y):
        y_env = np.quantile(np.maximum(g6_pf[:, i0:], 1e-16), q=q, axis=0)
    else:
        z = np.maximum(y[i0:], 1e-16)
        if roll > 1:
            pad = roll
            zz = np.pad(z, (pad, pad), mode="edge")
            y_env = np.maximum.reduce([zz[j:j+len(z)] for j in range(2*pad+1)])
        else:
            y_env = z
    return np.asarray(y_env, float), idx_tail

def fit_g6_models(
    r: np.ndarray,
    g6_avg: np.ndarray,
    g6_pf: Optional[np.ndarray] = None,
    start_bin: int = 20,
    q_env: float = 0.85,
    roll_env: int = 3,
    y_floor: float = 1e-3,
    y_cap: float = 1.0,
    # keep backward-compat kwargs (ignored)
    n_boot: int = 0,
    rng_seed: int = 123,
    **kwargs,
) -> G6ModelFit:
    """
    AIC competition on the upper envelope of the tail (floor at y>=y_floor).
    Stores intercepts so plotting doesn't reconstruct from ad-hoc points.
    """
    r = np.asarray(r, float)
    y = np.asarray(g6_avg, float)

    # envelope & acceptance window
    y_env, idx_tail = _g6_upper_envelope(g6_avg, g6_pf, start_bin, q=q_env, roll=roll_env)
    r_tail = r[idx_tail]
    mask = (r_tail > 0) & np.isfinite(y_env) & (y_env >= y_floor) & (y_env <= y_cap)
    r_tail = r_tail[mask]; y_env = y_env[mask]

    if r_tail.size < 6:
        return G6ModelFit("exp", 0.0, np.nan, np.nan, 0, 0, 0, 0, 0,
                          np.nan, np.nan, 0, 0.0, 0.0,
                          f"insufficient tail after floor y>={y_floor:g}",
                          idx_tail[mask], r_tail, y_env,
                          logA_power=np.nan, alpha_exp=np.nan, logA_exp=np.nan)

    lw = np.log(y_env)

    # const (log g6 = c0)
    c0 = float(np.mean(lw))
    rss_c = float(np.sum((lw - c0)**2)); k_c = 1

    # power (log g6 = -η log r + b_p)
    Xp = np.vstack([np.log(r_tail), np.ones_like(r_tail)]).T
    sp, bp = np.linalg.lstsq(Xp, lw, rcond=None)[0]
    eta = -float(sp)         # slope is -η
    logA_power = float(bp)
    rss_p = float(np.sum((lw - (sp*np.log(r_tail) + bp))**2)); k_p = 2

    # exp (log g6 = α r + b_e)
    Xe = np.vstack([r_tail, np.ones_like(r_tail)]).T
    se, be = np.linalg.lstsq(Xe, lw, rcond=None)[0]
    alpha_exp = float(se); logA_exp = float(be)
    rss_e = float(np.sum((lw - (se*r_tail + be))**2)); k_e = 2

    def AIC(rss, n, k): return 2*k + n*np.log(max(rss, 1e-300)/n)
    n = r_tail.size
    aic_c = AIC(rss_c, n, k_c)
    aic_p = AIC(rss_p, n, k_p)
    aic_e = AIC(rss_e, n, k_e)
    aics  = np.array([aic_c, aic_p, aic_e]); labels = np.array(["const","power","exp"])
    order = np.argsort(aics); best = labels[order[0]]
    delta = float(aics[order[1]] - aics[order[0]])

    rel_var   = float((np.max(y_env) - np.min(y_env)) / max(np.mean(y_env), 1e-300))
    tail_mean = float(np.mean(np.clip(y[-max(3, len(y)//5):], 1e-300, None)))

    return G6ModelFit(
        model=best,
        c_tail=float(np.exp(c0)),
        eta6_mean=float(eta),
        eta6_std=float("nan"),
        aic_const=float(aic_c),
        aic_power=float(aic_p),
        aic_exp=float(aic_e),
        best_aic=float(aics[order[0]]),
        delta_to_next=delta,
        rmin=float(r_tail.min()),
        rmax=float(r_tail.max()),
        n_points=int(n),
        rel_var=rel_var,
        tail_mean=tail_mean,
        note=f"upper-envelope q={q_env}, floor y>={y_floor:g}, cap y<={y_cap:g}",
        idx_used=idx_tail[mask].copy(),
        r_used=r_tail.copy(),
        y_used=y_env.copy(),
        logA_power=logA_power,
        alpha_exp=alpha_exp,
        logA_exp=logA_exp
    )


def psi6_subblock_scaling(
    one_frame_xy: np.ndarray,
    box_length: Tuple[float,float],
    LB_over_L: Tuple[float,...] = (0.50, 0.40, 0.33, 0.25, 0.20, 0.167, 0.125,
                                   0.10, 0.0833, 0.071, 0.0625, 0.05),
    min_particles_per_block: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ψ6 sub-block scaling down to LB/L ≈ 0.05 with a per-block particle floor
    to control noise at small blocks.
    """
    import freud
    Lx, Ly = float(box_length[0]), float(box_length[1])
    L = np.sqrt(Lx * Ly)
    pts = one_frame_xy.copy()
    # global Ψ6
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=1.0, is2D=True)
    pts3 = np.pad(pts, ((0,0),(0,1)))
    voro = freud.locality.Voronoi(); voro.compute((box, pts3))
    op   = freud.order.Hexatic(k=6); op.compute((box, pts3), neighbors=voro.nlist)
    psi2_global = float(np.abs(np.mean(op.particle_order))**2)

    xs, ys = [], []
    xfold = np.mod(pts[:,0], Lx); yfold = np.mod(pts[:,1], Ly)
    for frac in LB_over_L:
        LB = max(1e-9, frac * L)
        # tile counts
        nx = max(1, int(round(Lx / LB)))
        ny = max(1, int(round(Ly / LB)))
        dx, dy = Lx / nx, Ly / ny

        ix = np.minimum((xfold/dx).astype(int), nx-1)
        iy = np.minimum((yfold/dy).astype(int), ny-1)

        psi2_list = []
        for bx in range(nx):
            for by in range(ny):
                mask = (ix==bx) & (iy==by)
                if np.count_nonzero(mask) < min_particles_per_block:
                    continue
                psub = np.pad(np.column_stack([xfold[mask], yfold[mask]]), ((0,0),(0,1)))
                voro.compute((box, psub))
                op.compute((box, psub), neighbors=voro.nlist)
                psi2_list.append(np.abs(np.mean(op.particle_order))**2)
        if psi2_global > 0 and len(psi2_list) > 0:
            xs.append(dx / L)
            ys.append(float(np.mean(psi2_list) / psi2_global))
    return np.asarray(xs), np.asarray(ys)


# --- Fused decision: use g6 models + Ψ6 sub-block scaling ---
@dataclass
class HexaticDecision:
    phase: str                 # 'fluid' | 'hexatic' | 'solid' | 'transition'
    reason: str
    eta6_mean: float
    eta6_std: float
    psi6_abs_mean: float
    U4: float
    chi6: float
    tail_mean: float
    x_sub: np.ndarray          # LB/L
    y_sub: np.ndarray          # Ψ6^2(LB)/Ψ6^2(L)
    fit: G6ModelFit

def classify_hexatic_fluid_solid(frames_xy: Sequence[np.ndarray],
                                 box_length: Tuple[float,float],
                                 n_particles: int,
                                 rmax_frac: float = 0.45,
                                 nbins: int = 140,
                                 start_bin: int = 25,
                                 aic_margin: float = 4.0,
                                 eta_hex: float = 0.25,
                                 eta_margin: float = 0.02,
                                 tail_liquid_cut: float = 2e-1,
                                 psi6_solid_min: float = 0.50,
                                 subblock_x: Sequence[float]=(0.5,0.4,0.33,0.25,0.2),
                                 n_boot: int = 300,
                                 rng_seed: int = 123) -> HexaticDecision:
    """
    Deterministic KTHNY-compatible classifier.
      • g6(r): const vs power vs exp (AIC, bootstrap η6).
      • Ψ6 sub-block scaling: curve above (LB/L)^{-1/4} ⇒ >= hexatic; below ⇒ liquid.  # :contentReference[oaicite:2]{index=2}
    """
    # 1) g6(r)
    (Lx, Ly) = float(box_length[0]), float(box_length[1])
    # from .analysis import compute_g6_avg, compute_psi6_series, binder_and_susceptibility  # if inside a package, adjust import
    r, g6_avg, g6_pf = compute_g6_avg(frames_xy, box_length, r_max=rmax_frac*min(Lx,Ly),
                                      nbins=nbins, return_per_frame=True)
    fit = fit_g6_models(r, g6_avg, g6_pf, start_bin=start_bin, n_boot=n_boot, rng_seed=rng_seed)

    # 2) global Ψ6, Binder, χ6
    psi6_series, psi6_abs_mean, _ = compute_psi6_series(frames_xy, box_length, order_k=6)
    U4, chi6 = binder_and_susceptibility(psi6_series, n_particles)

    # 3) Ψ6 sub-block scaling on a representative frame (middle frame)
    k = len(frames_xy)//2
    x_sub, y_sub = psi6_subblock_scaling(frames_xy[k][:,:2], (Lx, Ly), LB_over_L=subblock_x)

    # reference line (LB/L)^(-1/4) – compare mean log residual
    if len(x_sub) >= 3:
        ref = np.power(x_sub, -0.25)  # separates liquid from hexatic/solid in scaling plot.  # :contentReference[oaicite:3]{index=3}
        above_frac = float(np.mean(y_sub >= ref))
    else:
        ref = np.array([]); above_frac = np.nan

    # 4) fused rules
    # liquid: exp g6 or small tail + scaling below ref
    if (fit.model == "exp" and fit.delta_to_next >= aic_margin) or (fit.tail_mean < tail_liquid_cut):
        if np.isnan(above_frac) or (above_frac < 0.5):
            reason = "g6: exponential (or small tail) and Ψ6-scaling below ref"
            return HexaticDecision("fluid", reason, fit.eta6_mean, fit.eta6_std, psi6_abs_mean, U4, chi6,
                                   fit.tail_mean, x_sub, y_sub, fit)

    # solid: const g6 or very small η6 + strong global |Ψ6|
    if (fit.model == "const" and fit.delta_to_next >= aic_margin) or \
       ((fit.eta6_mean <= 0.05) and (fit.tail_mean >= 0.3) and (psi6_abs_mean >= psi6_solid_min)):
        reason = "g6: nearly constant / η6≈0 with strong |Ψ6|"
        return HexaticDecision("solid", reason, fit.eta6_mean, fit.eta6_std, psi6_abs_mean, U4, chi6,
                               fit.tail_mean, x_sub, y_sub, fit)

    # hexatic: power-law with η6 < 1/4 (clear AIC win) and scaling above ref
    if (fit.model == "power" and fit.delta_to_next >= aic_margin and (fit.eta6_mean < (eta_hex - eta_margin))):
        if np.isnan(above_frac) or (above_frac >= 0.5):
            reason = "g6: algebraic with η6<1/4 and Ψ6-scaling above ref"
            return HexaticDecision("hexatic", reason, fit.eta6_mean, fit.eta6_std, psi6_abs_mean, U4, chi6,
                                   fit.tail_mean, x_sub, y_sub, fit)

    # boundary / ambiguous: near η6≈1/4 or conflicting signals
    reason = "near thresholds (η6≈1/4) or g6/Ψ6 scaling disagree"
    return HexaticDecision("transition", reason, fit.eta6_mean, fit.eta6_std, psi6_abs_mean, U4, chi6,
                           fit.tail_mean, x_sub, y_sub, fit)


# =============================================================================
# Phase classifiers (orientational)
# =============================================================================

@dataclass
class PhaseResult:
    phase: str     # 'fluid' | 'hexatic' | 'solid' | 'transition'
    reason: str
    eta6: float
    psi6_abs_mean: float
    U4: float
    chi6: float
    extras: dict

def classify_orientational_phase(frames, box_length, n_particles,
                                 rmax_frac=0.45, nbins=120,
                                 eta_hex_upper=0.25, eta_solid_tol=0.05,
                                 tail_min=0.20, psi6_solid_min=0.50,
                                 r2_margin=0.02):
    """
    Simple classifier using fit_g6_auto + tail level + <|Ψ6|>.
    """
    (Lx, Ly), _ = _ensure_box(box_length)
    r, g6 = compute_g6_avg(frames, box_length, r_max=rmax_frac * min(Lx, Ly), nbins=nbins)
    psi6_series, psi6_abs_mean, _ = compute_psi6_series(frames, box_length, order_k=6)
    U4, chi6 = binder_and_susceptibility(psi6_series, n_particles)

    fit = fit_g6_auto(r, g6, start_bin=10, r2_margin=r2_margin)
    tail_mean = float(np.mean(g6[-max(3, len(g6)//5):])) if len(g6) else 0.0

    if fit.kind == "exp":
        return PhaseResult("fluid", "exp decay fits g6(r)", np.nan, psi6_abs_mean, U4, chi6,
                           {"tail_mean": tail_mean, "fit": fit})

    if fit.kind == "power":
        if (fit.eta6 <= eta_solid_tol) and (tail_mean >= tail_min) and (psi6_abs_mean >= psi6_solid_min):
            return PhaseResult("solid", "g6 ~ const; high tail; high |Ψ6|", fit.eta6, psi6_abs_mean, U4, chi6,
                               {"tail_mean": tail_mean, "fit": fit})
        if (fit.eta6 > eta_solid_tol) and (fit.eta6 < eta_hex_upper):
            return PhaseResult("hexatic", "algebraic with 0<η6<1/4", fit.eta6, psi6_abs_mean, U4, chi6,
                               {"tail_mean": tail_mean, "fit": fit})
        return PhaseResult("transition", "algebraic but near thresholds", fit.eta6, psi6_abs_mean, U4, chi6,
                           {"tail_mean": tail_mean, "fit": fit})

    return PhaseResult("transition", "insufficient or ambiguous", np.nan, psi6_abs_mean, U4, chi6,
                       {"tail_mean": tail_mean, "fit": fit})


@dataclass
class PhaseRules:
    phase: str
    reason: str
    eta6_mean: float
    eta6_std: float
    psi6_abs_mean: float
    U4: float
    chi6: float
    fit: G6ModelFit

def classify_orientational_phase_rules(frames, box_length, n_particles,
                                       rmax_frac=0.45, nbins=120,
                                       start_bin=25,
                                       n_boot=200, rng_seed=0,
                                       tail_liquid_cut=2e-1,
                                       const_rel_tol=0.01,
                                       aic_margin=4.0,
                                       eta_hex=0.25,
                                       eta_margin=0.02,
                                       psi6_solid_min=0.50):
    """
    Robust rule-based classifier (const/power/exp + thresholds).
    """
    (Lx, Ly), _ = _ensure_box(box_length)
    r, g6_avg, g6_pf = compute_g6_avg(frames, box_length, r_max=rmax_frac * min(Lx, Ly),
                                      nbins=nbins, return_per_frame=True)
    psi6_series, psi6_abs_mean, _ = compute_psi6_series(frames, box_length, order_k=6)
    U4, chi6 = binder_and_susceptibility(psi6_series, n_particles)

    fit = fit_g6_models(r, g6_avg, g6_pf, start_bin=start_bin, n_boot=n_boot, rng_seed=rng_seed)

    if fit.tail_mean < float(tail_liquid_cut):
        return PhaseRules("fluid", f"g6 tail < {tail_liquid_cut}", fit.eta6_mean, fit.eta6_std,
                          psi6_abs_mean, U4, chi6, fit)

    if (fit.rel_var <= float(const_rel_tol)) or (fit.model == "const" and fit.delta_to_next >= float(aic_margin)):
        if psi6_abs_mean >= float(psi6_solid_min):
            return PhaseRules("solid", "g6 nearly constant and |Psi6| high", fit.eta6_mean, fit.eta6_std,
                              psi6_abs_mean, U4, chi6, fit)
        return PhaseRules("transition", "g6 nearly constant but |Psi6| not high", fit.eta6_mean, fit.eta6_std,
                          psi6_abs_mean, U4, chi6, fit)

    if (fit.model == "exp") and (fit.delta_to_next >= float(aic_margin)):
        return PhaseRules("fluid", "exp better than power/const by AIC", fit.eta6_mean, fit.eta6_std,
                          psi6_abs_mean, U4, chi6, fit)

    if (fit.model == "power") and (fit.delta_to_next >= float(aic_margin)):
        low = fit.eta6_mean - (fit.eta6_std if np.isfinite(fit.eta6_std) else 0.0)
        high = fit.eta6_mean + (fit.eta6_std if np.isfinite(fit.eta6_std) else 0.0)
        if (abs(fit.eta6_mean - eta_hex) <= float(eta_margin)) or (low < eta_hex < high):
            return PhaseRules("transition", "η6 near 1/4 (hex–liquid boundary)", fit.eta6_mean, fit.eta6_std,
                              psi6_abs_mean, U4, chi6, fit)
        if fit.eta6_mean < eta_hex - float(eta_margin):
            return PhaseRules("hexatic", "algebraic with η6 < 1/4", fit.eta6_mean, fit.eta6_std,
                              psi6_abs_mean, U4, chi6, fit)
        if fit.eta6_mean > eta_hex + float(eta_margin):
            return PhaseRules("transition", "algebraic with η6 > 1/4 (liquid-side)", fit.eta6_mean, fit.eta6_std,
                              psi6_abs_mean, U4, chi6, fit)

    return PhaseRules("transition", "models too close or low confidence", fit.eta6_mean, fit.eta6_std,
                      psi6_abs_mean, U4, chi6, fit)


# =============================================================================
# Scans across state points (density or mu)
# =============================================================================

def scan_by_density_or_mu(dataset: dict, box_length, n_particles,
                          key="density", rmax_frac=0.45, nbins=120):
    """
    Compute <|Ψ6|>, U4, χ6 and g6(r) for each state point.

    Returns
    -------
    table : np.ndarray (structured with columns: key, psi6_mean_abs, psi6_stderr_abs, U4, chi6)
    g6_curves : dict[key -> (r, g6)]
    """
    (Lx, Ly), _ = _ensure_box(box_length)
    rows = []
    curves = {}
    for x in sorted(dataset.keys()):
        frames = dataset[x]
        psi6_series, psi_mean, psi_err = compute_psi6_series(frames, box_length, order_k=6)
        U4, chi6 = binder_and_susceptibility(psi6_series, n_particles)
        r, g6 = compute_g6_avg(frames, box_length, r_max=rmax_frac * min(Lx, Ly), nbins=nbins)
        rows.append((float(x), psi_mean, psi_err, float(U4), float(chi6)))
        curves[float(x)] = (r, g6)
    table = np.array(rows, dtype=[(key, 'f8'),
                                  ('psi6_mean_abs', 'f8'),
                                  ('psi6_stderr_abs', 'f8'),
                                  ('U4', 'f8'),
                                  ('chi6', 'f8')])
    return table, curves


def scan_and_classify_orientational(dataset, box_length, n_particles, key="density",
                                    rmax_frac=0.45, nbins=120, **kwargs):
    rows = []
    for x in sorted(dataset.keys()):
        pr = classify_orientational_phase(dataset[x], box_length, n_particles,
                                          rmax_frac=rmax_frac, nbins=nbins, **kwargs)
        rows.append({
            key: float(x),
            "phase": pr.phase,
            "reason": pr.reason,
            "eta6": pr.eta6,
            "psi6_mean_abs": pr.psi6_abs_mean,
            "U4": pr.U4,
            "chi6": pr.chi6,
        })
    # boundaries (adjacent changes)
    bounds = []
    rs = sorted(rows, key=lambda d: d[key])
    for i in range(len(rs) - 1):
        if rs[i]["phase"] != rs[i + 1]["phase"]:
            bounds.append({"between": (rs[i][key], rs[i + 1][key]),
                           "at": 0.5 * (rs[i][key] + rs[i + 1][key]),
                           "from": rs[i]["phase"], "to": rs[i + 1]["phase"]})
    return rs, bounds


def scan_and_classify_orientational_robust(dataset, box_length, n_particles, key="density",
                                           rmax_frac=0.45, nbins=120, **kwargs):
    rows = []
    for x in sorted(dataset.keys()):
        pr = classify_orientational_phase_rules(dataset[x], box_length, n_particles,
                                                rmax_frac=rmax_frac, nbins=nbins, **kwargs)
        rows.append({
            key: float(x),
            "phase": pr.phase,
            "reason": pr.reason,
            "eta6_mean": pr.eta6_mean,
            "eta6_std": pr.eta6_std,
            "psi6_mean_abs": pr.psi6_abs_mean,
            "U4": pr.U4,
            "chi6": pr.chi6,
            "fit_model": pr.fit.model,
            "delta_to_next": pr.fit.delta_to_next,
            "fit_rmin": pr.fit.rmin,
            "fit_rmax": pr.fit.rmax,
            "bins_used": pr.fit.n_points,
            "rel_var": pr.fit.rel_var,
            "tail_mean_g6": pr.fit.tail_mean,
        })
    bounds = []
    rs = sorted(rows, key=lambda d: d[key])
    for i in range(len(rs) - 1):
        if rs[i]["phase"] != rs[i + 1]["phase"]:
            bounds.append({"between": (rs[i][key], rs[i + 1][key]),
                           "at": 0.5 * (rs[i][key] + rs[i + 1][key]),
                           "from": rs[i]["phase"], "to": rs[i + 1]["phase"]})
    return rs, bounds


# =============================================================================
# Legacy/compat stubs (translational order — disabled now)
# =============================================================================

def compute_gG(*args, **kwargs):
    """
    Positional correlation at reciprocal vector G.
    Disabled for now. We will reintroduce after orientational pipeline is stable.
    """
    raise NotImplementedError("Translational order (compute_gG) is disabled for now.")


def sub_system_translational(*args, **kwargs):
    """
    Subsystem translational analysis.
    Disabled for now. We will reintroduce after orientational pipeline is stable.
    """
    raise NotImplementedError("Translational analysis is disabled for now.")


# =============================================================================
# Compatibility shim: compute_psi and compute_psi_density (old names)
# =============================================================================

def compute_psi(all_positions, box_length, order_number=6):
    """
    Back-compat wrapper: returns <|ψ_k|> over frames and stderr.
    (ψ_k here refers to local particle-wise order; we report frame-averaged |ψ_k|.)
    """
    (Lx, Ly), box = _ensure_box(box_length)
    per_frame_abs_mean = []
    for pos in all_positions:
        pts = _to_3d(pos)
        voro = freud.locality.Voronoi()
        voro.compute((box, pts))
        op = freud.order.Hexatic(k=order_number)
        op.compute((box, pts), neighbors=voro.nlist)
        per_frame_abs_mean.append(np.mean(np.abs(op.particle_order)))
    arr = np.asarray(per_frame_abs_mean)
    n = len(arr)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def compute_psi_density(all_positions, box_length, nbins=(64, 64)):
    """
    Produce 2D maps of:
      - number density ρ(x,y)
      - mean |ψ6|(x,y) (binned per pixel)
    Returns
    -------
    x_edges, y_edges, rho_map, psi_abs_map
    """
    (Lx, Ly), _ = _ensure_box(box_length)
    nx, ny = int(nbins[0]), int(nbins[1])
    rho_acc = np.zeros((ny, nx), dtype=np.float64)
    psi_acc = np.zeros((ny, nx), dtype=np.float64)
    cnt_acc = np.zeros((ny, nx), dtype=np.float64)

    for pos in all_positions:
        pts2 = np.asarray(pos)
        pts = _to_3d(pos)
        # ψ6 per particle
        _, box = _ensure_box(box_length)
        voro = freud.locality.Voronoi()
        voro.compute((box, pts))
        op = freud.order.Hexatic(k=6)
        op.compute((box, pts), neighbors=voro.nlist)
        psi_abs = np.abs(op.particle_order)

        # bin
        x = np.mod(pts2[:, 0], Lx); y = np.mod(pts2[:, 1], Ly)
        H, yedges, xedges = np.histogram2d(y, x, bins=[ny, nx], range=[[0, Ly], [0, Lx]])
        rho_acc += H

        # weighted sum of |ψ6|
        for i in range(pts2.shape[0]):
            xi = int(min(nx - 1, np.floor(x[i] / (Lx / nx))))
            yi = int(min(ny - 1, np.floor(y[i] / (Ly / ny))))
            psi_acc[yi, xi] += psi_abs[i]
            cnt_acc[yi, xi] += 1.0

    # number density per area
    rho_map = rho_acc / ((Lx / nx) * (Ly / ny)) / max(len(all_positions), 1)
    psi_map = np.divide(psi_acc, np.maximum(cnt_acc, 1.0), out=np.zeros_like(psi_acc), where=cnt_acc > 0)
    return xedges, yedges, rho_map, psi_map

# -----------------------------------------------------------------------------
# Back-compat: legacy compute_g6 API (for __init__ imports and old scripts)
# -----------------------------------------------------------------------------
def compute_g6(all_positions, box_length, r_max=None, nbins=120, return_per_frame=False):
    """
    Legacy wrapper around compute_g6_avg.

    Parameters
    ----------
    all_positions : list[np.ndarray]
    box_length : float or (2,)
    r_max : float or None
        If None, uses 0.45 * min(Lx, Ly).
    nbins : int
    return_per_frame : bool
        If True, also returns per-frame g6 curves (T x nbins).

    Returns
    -------
    r, g6_avg              (if return_per_frame=False)
    r, g6_avg, g6_per_frame (if return_per_frame=True)
    """
    # normalize box + default r_max
    if isinstance(box_length, (float, int)):
        Lx, Ly = float(box_length), float(box_length)
    else:
        Lx, Ly = float(box_length[0]), float(box_length[1])
    if r_max is None:
        r_max = 0.45 * min(Lx, Ly)

    return compute_g6_avg(
        all_positions,
        (Lx, Ly),
        r_max=float(r_max),
        nbins=int(nbins),
        return_per_frame=bool(return_per_frame),
    )


# =============================================================================
# NVT local-density bimodality detector (periodic coarse-graining + GMM)
# =============================================================================
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from numpy.fft import rfftn, irfftn
from scipy.special import erf
from sklearn.mixture import GaussianMixture  # optional dependency: deterministic with random_state

@dataclass
class BimodalityResult:
    bimodal: bool
    w_min: float                  # min component weight
    dprime: float                 # separation metric
    mu_lo: float                  # lower-mean component (density)
    mu_hi: float                  # higher-mean component (density)
    sigma_lo: float               # std of low-density component
    sigma_hi: float               # std of high-density component
    rho_mean: float               # global mean density N/(Lx*Ly)
    params: Dict[str, float]      # nbins, sigma (grid units), target_ppc, etc.

# ---------- Helpers: CIC deposit under PBC ----------
def _cic_deposit_pbc(positions_xy: np.ndarray, Lx: float, Ly: float, nx: int, ny: int) -> np.ndarray:
    """
    Cloud-in-cell mass deposit of N points onto an (nx,ny) grid with periodic BC.
    Returns integer counts per cell (float array).
    """
    x = np.mod(positions_xy[:, 0], Lx) * (nx / Lx)
    y = np.mod(positions_xy[:, 1], Ly) * (ny / Ly)

    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    fx = x - ix
    fy = y - iy

    # neighbor indices with PBC
    ix1 = (ix + 1) % nx
    iy1 = (iy + 1) % ny

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    H = np.zeros((ny, nx), dtype=np.float64)  # (row=y, col=x)
    # scatter-add
    np.add.at(H, (iy, ix), w00)
    np.add.at(H, (iy, ix1), w10)
    np.add.at(H, (iy1, ix), w01)
    np.add.at(H, (iy1, ix1), w11)
    return H

# ---------- Gaussian blur under PBC via FFT ----------
def _gaussian_kernel_fourier(nx: int, ny: int, sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Fourier-space Gaussian for real-FFT layout (rfftn). sigma_* in grid cells.
    """
    ky = np.fft.fftfreq(ny)[:, None]  # shape (ny,1), cycles per sample
    kx = np.fft.rfftfreq(nx)[None, :] # shape (1,nx//2+1)
    # Convert to angular frequencies: 2π * cycles
    wy2 = (2.0 * np.pi * ky) ** 2
    wx2 = (2.0 * np.pi * kx) ** 2
    # Fourier of Gaussian: exp(-0.5 * (σx^2 kx^2 + σy^2 ky^2))
    Gk = np.exp(-0.5 * (sigma_x**2 * wx2 + sigma_y**2 * wy2))
    return Gk

def _blur_pbc_fft(field: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Convolve 'field' with a separable Gaussian under periodic BC using FFTs.
    sigma_* are in grid cells (not physical units).
    """
    fk = rfftn(field, s=field.shape)
    Gk = _gaussian_kernel_fourier(field.shape[1], field.shape[0], sigma_x, sigma_y)
    out = irfftn(fk * Gk, s=field.shape)
    return out

# ---------- Main construction: ρ(x,y) on a grid ----------
def coarse_grained_density_pbc(positions_xy: np.ndarray,
                               box_length: Tuple[float, float],
                               nbins: Optional[Tuple[int, int]] = None,
                               sigma_phys: Optional[float] = None,
                               target_particles_per_cell: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Construct coarse-grained density ρ(x,y) for an NVT snapshot using:
      1) CIC deposit to counts grid H (periodic).
      2) Gaussian blur with width sigma_phys (periodic).
    Parameters
    ----------
    positions_xy : (N,2) float
    box_length   : (Lx,Ly)
    nbins        : (nx,ny). If None, auto-set for ~target_particles_per_cell on average.
    sigma_phys   : Gaussian σ in *physical units*. If None, set σ = 0.6 * a, a = 1/sqrt(ρ).
    target_particles_per_cell : average particles per cell used to pick grid if nbins is None.

    Returns
    -------
    xedges, yedges, rho_map, info
      rho_map has shape (ny,nx), with physical density units (#/area).
    """
    Lx, Ly = float(box_length[0]), float(box_length[1])
    N = positions_xy.shape[0]
    A = Lx * Ly
    rho_mean = N / A

    # grid selection
    if nbins is None:
        n_cells = max(N // max(1, target_particles_per_cell), 64)  # lower bound for resolution
        aspect = Lx / Ly
        nx = int(np.sqrt(n_cells * aspect))
        ny = max(1, int(n_cells // max(nx, 1)))
        # round to even numbers for FFT friendliness
        nx = int(2 * np.ceil(nx / 2))
        ny = int(2 * np.ceil(ny / 2))
    else:
        nx, ny = int(nbins[0]), int(nbins[1])

    # CIC deposit
    H = _cic_deposit_pbc(positions_xy, Lx, Ly, nx, ny)  # counts per cell

    # Gaussian width
    if sigma_phys is None:
        a = 1.0 / np.sqrt(rho_mean)      # mean interparticle spacing
        sigma_phys = 0.6 * a             # default: mildly smooth shot noise, keep interfaces sharp

    # convert σ to grid cells along x,y
    dx, dy = Lx / nx, Ly / ny
    sigma_x = sigma_phys / dx
    sigma_y = sigma_phys / dy

    # periodic Gaussian blur
    H_smooth = _blur_pbc_fft(H, sigma_x, sigma_y)

    # Map to density: counts / cell_area
    cell_area = dx * dy
    rho_map = H_smooth / cell_area

    xedges = np.linspace(0.0, Lx, nx + 1)
    yedges = np.linspace(0.0, Ly, ny + 1)
    info = dict(nx=float(nx), ny=float(ny),
                dx=float(dx), dy=float(dy),
                sigma_phys=float(sigma_phys),
                sigma_x=float(sigma_x), sigma_y=float(sigma_y),
                rho_mean=float(rho_mean),
                target_ppc=float(target_particles_per_cell))
    return xedges, yedges, rho_map, info

# ---------- Objective bimodality test on ρ grid ----------
def density_bimodality_gmm(rho_map: np.ndarray,
                           min_weight: float = 0.10,
                           min_dprime: float = 2.0) -> Tuple[bool, Dict[str, float]]:
    """
    Fit a 2-component Gaussian Mixture to the set {ρ(x,y)} (grid samples).
    Declare 'bimodal' if:
      - both component weights ≥ min_weight, and
      - separation d' = |μ1-μ2| / sqrt(0.5(σ1^2+σ2^2)) ≥ min_dprime.
    Returns (bimodal, diagnostics).
    """
    vals = rho_map.ravel().astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 200:
        return False, {"n": float(vals.size)}

    X = vals.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(X)
    w = gmm.weights_
    mu = gmm.means_.ravel()
    s2 = np.array([np.squeeze(c) for c in gmm.covariances_])
    order = np.argsort(mu)
    mu = mu[order]; s2 = s2[order]; w = w[order]

    dprime = abs(mu[1] - mu[0]) / np.sqrt(0.5 * (s2[0] + s2[1]))
    bimodal = (w.min() >= min_weight) and (dprime >= min_dprime)

    diag = {
        "w_min": float(w.min()),
        "w_lo": float(w[0]),
        "w_hi": float(w[1]),
        "mu_lo": float(mu[0]),
        "mu_hi": float(mu[1]),
        "sigma_lo": float(np.sqrt(s2[0])),
        "sigma_hi": float(np.sqrt(s2[1])),
        "dprime": float(dprime),
    }
    return bool(bimodal), diag

# ---------- Public API ----------
def detect_liquid_gas_bimodality_nvt(positions_xy: np.ndarray,
                                     box_length: Tuple[float, float],
                                     nbins: Optional[Tuple[int, int]] = None,
                                     sigma_phys: Optional[float] = None,
                                     gmm_min_weight: float = 0.10,
                                     gmm_min_dprime: float = 2.0,
                                     target_particles_per_cell: int = 8) -> BimodalityResult:
    """
    End-to-end NVT detector:
      - builds coarse-grained ρ(x,y) with periodic CIC + Gaussian blur,
      - runs GMM test,
      - returns a structured BimodalityResult.
    """
    xedges, yedges, rho_map, info = coarse_grained_density_pbc(
        positions_xy, box_length, nbins=nbins, sigma_phys=sigma_phys,
        target_particles_per_cell=target_particles_per_cell
    )
    bimodal, diag = density_bimodality_gmm(rho_map,
                                           min_weight=gmm_min_weight,
                                           min_dprime=gmm_min_dprime)
    res = BimodalityResult(
        bimodal=bimodal,
        w_min=float(diag.get("w_min", 0.0)),
        dprime=float(diag.get("dprime", 0.0)),
        mu_lo=float(diag.get("mu_lo", np.nan)),
        mu_hi=float(diag.get("mu_hi", np.nan)),
        sigma_lo=float(diag.get("sigma_lo", np.nan)),
        sigma_hi=float(diag.get("sigma_hi", np.nan)),
        rho_mean=float(info["rho_mean"]),
        params=dict(
            nx=info["nx"], ny=info["ny"], dx=info["dx"], dy=info["dy"],
            sigma_phys=info["sigma_phys"], sigma_x=info["sigma_x"], sigma_y=info["sigma_y"],
            gmm_min_weight=float(gmm_min_weight), gmm_min_dprime=float(gmm_min_dprime),
            target_ppc=float(target_particles_per_cell)
        )
    )
    return res, (xedges, yedges, rho_map)

# =============================================================================
# Per-particle Voronoi local density (PBC) + robust bimodality (log-space, BIC)
# =============================================================================
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# freud has PBC Voronoi; fall back to grid if unavailable
try:
    import freud
    HAS_FREUD = True
except Exception:
    HAS_FREUD = False

from sklearn.mixture import GaussianMixture

@dataclass
class BimodalityResultV:
    bimodal: bool
    method: str                  # 'voronoi' or 'grid'
    weights: Tuple[float, float] # (w_lo, w_hi) if bimodal else (1.0, 0.0)
    means: Tuple[float, float]   # component means in *density* units (not log)
    sigmas: Tuple[float, float]  # component stds (density units)
    dprime: float                # Ashman-like separation in log-space
    delta_bic: float             # BIC(1) - BIC(2), > 0 favors 2 comps
    rho_mean: float
    params: Dict[str, float]

def voronoi_local_density_pbc(positions_xy: np.ndarray,
                              box_length: Tuple[float, float]) -> np.ndarray:
    """
    Per-particle local density ρ_i = 1 / A_i where A_i is the Voronoi cell area
    computed with *periodic* boundary conditions using freud.
    Returns ρ_i with shape (N,).
    """
    if not HAS_FREUD:
        raise RuntimeError("freud not available: cannot compute Voronoi local density with PBC.")
    Lx, Ly = float(box_length[0]), float(box_length[1])
    N = positions_xy.shape[0]
    # freud box: 2D periodic in x,y; provide 3D with zero z-extent but periodic in xy only
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=1.0, is2D=True)
    pts = np.zeros((N, 3), dtype=np.float64)
    pts[:, :2] = positions_xy % np.array([Lx, Ly])
    v = freud.locality.Voronoi()
    v.compute((box, pts))
    # polygon areas are in 2D; freud returns per-particle areas
    areas = v.volumes  # shape (N,)
    rho_local = 1.0 / areas
    return rho_local

def _gmm_bimodality_log(values: np.ndarray,
                        min_weight: float = 0.02,
                        min_dprime: float = 1.5,
                        min_delta_bic: float = 6.0,
                        random_state: int = 0) -> Tuple[bool, Dict[str, float]]:
    """
    Fit GMMs to log(values). Decide bimodality if:
      - 2-component model has BIC at least 'min_delta_bic' better than 1-component, and
      - both component weights >= min_weight, and
      - separation d' >= min_dprime (on log scale).
    Returns (decision, diagnostics dict).
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 200:
        return False, {"n": float(x.size)}

    lx = np.log(x).reshape(-1, 1)

    g1 = GaussianMixture(n_components=1, covariance_type="full", random_state=random_state).fit(lx)
    g2 = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state).fit(lx)
    bic1 = g1.bic(lx)
    bic2 = g2.bic(lx)
    delta_bic = bic1 - bic2  # >0 favors 2 components

    w = g2.weights_.copy()
    mu = g2.means_.ravel().copy()
    s2 = np.array([np.squeeze(c) for c in g2.covariances_])
    order = np.argsort(mu)
    w = w[order]; mu = mu[order]; s2 = s2[order]

    # Ashman-like separation on log scale
    dprime = abs(mu[1] - mu[0]) / np.sqrt(0.5 * (s2[0] + s2[1]))

    # Decision
    bimodal = (delta_bic >= min_delta_bic) and (w.min() >= min_weight) and (dprime >= min_dprime)

    # Back-transform means/sigmas to density units for reporting
    mean_lo = float(np.exp(mu[0] + 0.5 * s2[0]))  # mean of lognormal
    mean_hi = float(np.exp(mu[1] + 0.5 * s2[1]))
    sigma_lo = float(np.sqrt((np.exp(s2[0]) - 1.0) * np.exp(2 * mu[0] + s2[0])))
    sigma_hi = float(np.sqrt((np.exp(s2[1]) - 1.0) * np.exp(2 * mu[1] + s2[1])))

    diag = {
        "bic1": float(bic1), "bic2": float(bic2), "delta_bic": float(delta_bic),
        "w_lo": float(w[0]), "w_hi": float(w[1]),
        "mu_log_lo": float(mu[0]), "mu_log_hi": float(mu[1]),
        "s2_log_lo": float(s2[0]), "s2_log_hi": float(s2[1]),
        "dprime": float(dprime),
        "mean_lo": mean_lo, "mean_hi": mean_hi,
        "sigma_lo": sigma_lo, "sigma_hi": sigma_hi,
    }
    return bool(bimodal), diag

def detect_liquid_gas_bimodality_nvt_voronoi(positions_xy: np.ndarray,
                                             box_length: Tuple[float, float],
                                             min_weight: float = 0.02,
                                             min_dprime: float = 1.5,
                                             min_delta_bic: float = 6.0,
                                             random_state: int = 0) -> Tuple[BimodalityResultV, np.ndarray]:
    """
    Deterministic NVT detector based on per-particle Voronoi local density.
    Robust to small minority phase fractions and avoids interface smearing.
    Decision uses BIC(1) vs BIC(2) on log(ρ_i), plus weight and separation cuts.
    Returns (result, rho_per_particle).
    """
    # Local densities from Voronoi (PBC)
    rho_i = voronoi_local_density_pbc(positions_xy, box_length)
    N = rho_i.size
    Lx, Ly = float(box_length[0]), float(box_length[1])
    rho_mean = N / (Lx * Ly)

    ok, diag = _gmm_bimodality_log(rho_i,
                                   min_weight=min_weight,
                                   min_dprime=min_dprime,
                                   min_delta_bic=min_delta_bic,
                                   random_state=random_state)

    if ok:
        res = BimodalityResultV(
            bimodal=True,
            method="voronoi",
            weights=(diag["w_lo"], diag["w_hi"]),
            means=(diag["mean_lo"], diag["mean_hi"]),
            sigmas=(diag["sigma_lo"], diag["sigma_hi"]),
            dprime=diag["dprime"],
            delta_bic=diag["delta_bic"],
            rho_mean=float(rho_mean),
            params=dict(min_weight=min_weight, min_dprime=min_dprime, min_delta_bic=min_delta_bic)
        )
    else:
        res = BimodalityResultV(
            bimodal=False,
            method="voronoi",
            weights=(1.0, 0.0),
            means=(float(np.mean(rho_i)), float("nan")),
            sigmas=(float(np.std(rho_i)), float("nan")),
            dprime=float(diag.get("dprime", 0.0)),
            delta_bic=float(diag.get("delta_bic", 0.0)),
            rho_mean=float(rho_mean),
            params=dict(min_weight=min_weight, min_dprime=min_dprime, min_delta_bic=min_delta_bic)
        )
    return res, rho_i



from pathlib import Path
import numpy as np

from .statics import structure_factor, structure_factor_from_gr
from .visualization import plot_structure_factor
from .utils import save_table_csv

def compute_and_save_structure_factor(
    positions: np.ndarray,
    box: tuple,
    outdir: str,
    tag: str = "frame0000",
    kmax: float = None,
    nbins: int = 200,
    save_figure: bool = True
):
    """
    Full pipeline: compute S(k)->S(q), save CSV and optional PNG.
    Outputs:
      outdir/Sq_{tag}.csv with columns [q, S(q)]
      outdir/Sq_{tag}.png
    """
    out = structure_factor(positions=positions, box=box, kmax=kmax, nbins=nbins)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"Sq_{tag}.csv"
    save_table_csv(csv_path, [out.q, out.Sq], header=["q", "S(q)"])

    if save_figure:
        import matplotlib.pyplot as plt
        ax = plot_structure_factor(out.q, out.Sq)
        fig = ax.figure
        fig.savefig(outdir / f"Sq_{tag}.png", dpi=300)
        plt.close(fig)

    # return full result for further analysis
    return out


from pathlib import Path
import numpy as np
from .statics import structure_factor_2d
from .visualization import plot_structure_factor_2d
from .utils import save_table_csv

def compute_and_save_structure_factor_2d(
    positions: np.ndarray,
    box: tuple,
    outdir: str,
    tag: str = "frame0000",
    nmax: int = 64,
    log10: bool = True,
    save_npz: bool = True,
    save_csv: bool = False,
    save_png: bool = True,
):
    """
    Full pipeline for 2D S(kx,ky). Saves:
      - NPZ (kx, ky, S) for lossless reuse,
      - optionally long-form CSV (kx, ky, S),
      - PNG heatmap (log10 by default).
    """
    out = structure_factor_2d(positions=positions, box=box, nmax=nmax)
    outpath = Path(outdir); outpath.mkdir(parents=True, exist_ok=True)

    if save_npz:
        np.savez_compressed(outpath / f"S2D_{tag}.npz", kx=out.kx, ky=out.ky, S=out.S, meta=out.meta)

    if save_csv:
        # long-form CSV: one row per pixel
        KX, KY = np.meshgrid(out.kx, out.ky, indexing="ij")
        save_table_csv(outpath / f"S2D_{tag}.csv",
                       [KX.ravel(), KY.ravel(), out.S.ravel()],
                       header=["kx", "ky", "S"])

    if save_png:
        import matplotlib.pyplot as plt
        ax = plot_structure_factor_2d(out.kx, out.ky, out.S, log10=log10,
                                      title=f"S(kx,ky) {tag}")
        ax.figure.savefig(outpath / f"S2D_{tag}.png", dpi=300)
        plt.close(ax.figure)

    return out

# translational.py
# from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .statics import structure_factor, radial_bin, extract_q_vectors_from_S2D



# ---------- 1) Select first-shell Bragg vector G from S(k) ----------
@dataclass
class GVector:
    G: np.ndarray          # (2,) chosen Bragg vector
    shell_k: float         # |G| magnitude (first ring)
    peaks: np.ndarray      # (M,2) peak positions used
def _safe_extract_q_vectors(gx, gy, Simg, **kwargs):
    """
    Call statics.extract_q_vectors_from_S2D with only the kwargs it supports.
    Returns either {'kxy': array(...)} or a plain ndarray, or None on failure.
    """
    try:
        from .statics import extract_q_vectors_from_S2D as _extract
    except Exception:
        return None

    try:
        sig = inspect.signature(_extract)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        allowed = {}

    try:
        return _extract(gx, gy, Simg, **allowed)
    except TypeError:
        # Retry with no kwargs at all
        try:
            return _extract(gx, gy, Simg)
        except Exception:
            return None

def select_G_from_Sk(positions: np.ndarray,
                     box: Tuple[float, float],
                     *,
                     kmax: Optional[float] = None,
                     nbins_radial: int = 240,
                     ring_prominence: float = 0.0,
                     topk: int = 6) -> GVector:
    """
    Determine a representative first Bragg vector G by:
      1) computing S(k) on the PBC grid,
      2) finding local maxima near the first ring,
      3) picking the strongest peak on that ring.
    Notes:
      • Do NOT constrain G // ψ6 (can fail for LJ); pick from S(k) directly.  # :contentReference[oaicite:4]{index=4}
    """
    sf = structure_factor(positions, box, kmax=kmax, nbins=nbins_radial)  # returns grid & radial S(q)  :contentReference[oaicite:5]{index=5}
    kxky = sf.kvecs
    Sk = sf.Sk
    kmag = np.linalg.norm(kxky, axis=1)

    # Estimate first-ring center from radial S(q) maximum beyond k=0
    q, Sq = sf.q, sf.Sq
    i1 = int(np.argmax(Sq[1:])) + 1  # skip q≈0
    k_ring = float(q[i1])

    # 2D peak picking restricted to a thin annulus around the first ring
    k_halfwidth = 0.10 * k_ring  # ±10% band is robust
    # Render S(k) on a regular mesh not needed here because extract_q_vectors_from_S2D expects 2D array:
    # Rebuild a coarse S(kx,ky) image (same grid used internally by structure_factor)
    # Instead of full image, use provided helper operating on S array if available:
    # We'll approximate by binning S(k) in a kx,ky grid via nearest neighbors (lightweight and robust).
    # Construct grid extents
    Lx, Ly = float(box[0]), float(box[1])
    dkx, dky = 2.0*np.pi/Lx, 2.0*np.pi/Ly
    nx = ny = 512  # coarse image for peak finding
    gx = (np.arange(-nx//2, nx//2) * dkx).astype(float)
    gy = (np.arange(-ny//2, ny//2) * dky).astype(float)
    # Map discrete allowed k-vectors to nearest pixel
    ix = np.clip(np.round((kxky[:,0] - gx[0])/(gx[1]-gx[0])).astype(int), 0, nx-1)
    iy = np.clip(np.round((kxky[:,1] - gy[0])/(gy[1]-gy[0])).astype(int), 0, ny-1)
    Simg = np.zeros((nx, ny), dtype=float)
    counts = np.zeros((nx, ny), dtype=int)
    np.add.at(Simg, (ix, iy), Sk)
    np.add.at(counts, (ix, iy), 1)
    Simg = np.where(counts>0, Simg/np.maximum(counts,1), 0.0)

        # Try your extractor; filter kwargs by supported signature
    res = _safe_extract_q_vectors(
        gx, gy, Simg,
        topk=topk,
        min_prominence=ring_prominence,
        exclude_center_radius=0.0,
        ring_center=k_ring,          # will be dropped if unsupported
        ring_halfwidth=k_halfwidth,  # will be dropped if unsupported
        subpixel=True
    )

    # Normalize return to an array of candidate peak positions
    klist = None
    if res is not None:
        if isinstance(res, dict) and "kxy" in res:
            klist = np.asarray(res["kxy"], float)
        elif isinstance(res, (list, tuple, np.ndarray)):
            arr = np.asarray(res, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                klist = arr

    if klist is None or klist.size == 0:
        # Fallback: strongest discrete allowed k near first ring
        mask = (np.abs(kmag - k_ring) <= k_halfwidth)
        if not np.any(mask):
            # very defensive: take global max away from k=0
            j = int(np.argmax(Sk * (kmag > 1e-12)))
        else:
            j = int(np.argmax(np.where(mask, Sk, -np.inf)))
        return GVector(G=kxky[j], shell_k=float(kmag[j]), peaks=np.array([kxky[j]]))


# ---------- 2) Global translational order Ψ_G ----------
def psi_G(positions: np.ndarray, G: np.ndarray) -> complex:
    """
    Ψ_G ≡ (1/N) ∑_j exp(i G · r_j), invariant to uniform translations.
    """
    z = np.exp(1j * (positions @ G))
    return np.mean(z)

# ---------- 3) Translational correlation c(r) ----------
def compute_c_of_r(frames: Sequence[np.ndarray],
                   box: Tuple[float,float],
                   G: np.ndarray,
                   *,
                   r_max: Optional[float] = None,
                   nbins: int = 140,
                   chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    c(r) = ⟨ e^{ i G·(r_k - r_l) } ⟩_{|r_k-l|∈[r,r+dr)} averaged over frames.
    O(N^2) with chunking; accurate for N≲1e4. Distances use minimum-image PBC.
    """
    Lx, Ly = float(box[0]), float(box[1])
    if r_max is None:
        r_max = 0.45 * min(Lx, Ly)
    edges = np.linspace(0.0, r_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    acc = np.zeros(nbins, dtype=np.complex128)
    cnt = np.zeros(nbins, dtype=np.int64)

    for R in frames:
        R = np.ascontiguousarray(R[:, :2], dtype=float)
        N = R.shape[0]
        # chunk over i to limit memory
        for i0 in range(0, N, chunk):
            i1 = min(N, i0 + chunk)
            Ri = R[i0:i1]
            # all j
            d = Ri[:, None, :] - R[None, :, :]
            # PBC minimum image
            d[...,0] -= Lx * np.rint(d[...,0] / Lx)
            d[...,1] -= Ly * np.rint(d[...,1] / Ly)
            rij = np.linalg.norm(d, axis=-1)
            # exclude self pairs
            mask = np.ones_like(rij, dtype=bool)
            if i1 - i0 == N:
                np.fill_diagonal(mask, False)
            # bin
            idx = np.digitize(rij[mask], edges) - 1
            valid = (idx >= 0) & (idx < nbins)
            vals = np.exp(1j * (d[mask] @ G))  # e^{i G·(r_i - r_j)}
            # accumulate real and imag separately to avoid precision bleed
            np.add.at(acc, idx[valid], vals[valid])
            np.add.at(cnt, idx[valid], 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        c_r = np.where(cnt > 0, acc / cnt, 0.0)
    return centers, c_r

# ---------- 4) Ψ_G sub-block scaling ----------
def psiG_subblock_scaling(positions: np.ndarray,
                          box: Tuple[float,float],
                          G: np.ndarray,
                          LB_over_L: Sequence[float]=(0.5,0.4,0.33,0.25,0.2),
                          grid_xy: Tuple[int,int]=(3,3)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Ψ_G^2(L_B)/Ψ_G^2(L) vs L_B/L.
    Criterion: curve below (L_B/L)^(-1/3) ⇒ solid; above ⇒ hexatic/liquid boundary region;
    liquids trend well above. (Used as a separating guide as in Zu et al. SM.)  # :contentReference[oaicite:7]{index=7}
    """
    Lx, Ly = float(box[0]), float(box[1])
    # global Ψ_G^2(L)
    psi_full = psi_G(positions, G)
    base = np.abs(psi_full)**2 + 1e-16

    x = []
    y = []
    for frac in LB_over_L:
        Bx = max(1, int(np.round(grid_xy[0] * frac)))
        By = max(1, int(np.round(grid_xy[1] * frac)))
        # tile grid of Bx×By sub-blocks
        xs = np.linspace(0.0, Lx, Bx+1)
        ys = np.linspace(0.0, Ly, By+1)
        vals = []
        for ix in range(Bx):
            for iy in range(By):
                x0,x1 = xs[ix], xs[ix+1]
                y0,y1 = ys[iy], ys[iy+1]
                # select particles in sub-box (no wrap for simplicity; affects only a tiny border set)
                selx = (positions[:,0] >= x0) & (positions[:,0] < x1)
                sely = (positions[:,1] >= y0) & (positions[:,1] < y1)
                mask = selx & sely
                if not np.any(mask):
                    continue
                psi_b = psi_G(positions[mask], G)
                vals.append(np.abs(psi_b)**2)
        if len(vals)==0:
            continue
        x.append(float(frac))
        y.append(float(np.mean(vals)/base))
    return np.array(x, float), np.array(y, float)

# ---------- 5) Lightweight model fitting & rules ----------
@dataclass
class TranslationalDecision:
    phase: str     # 'solid' | 'hexatic' | 'fluid' | 'transition'
    reason: str
    etaG: Optional[float]
    psiG_abs: float
    x_sub: np.ndarray
    y_sub: np.ndarray
    G: np.ndarray
    k_shell: float

def fit_powerlaw_eta(r: np.ndarray, c_r_abs: np.ndarray, rmin_bins: int = 20) -> Optional[float]:
    # Fit |c(r)| ~ r^{-η_G} over tail (avoid small-r oscillations)
    if len(r) < rmin_bins + 10:
        return None
    x = np.log(r[rmin_bins:])
    y = np.log(np.maximum(c_r_abs[rmin_bins:], 1e-16))
    # linear least squares
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    # y ~ a - η x ⇒ slope = -η
    return float(max(0.0, coef[1]))

def classify_translational(frames: Sequence[np.ndarray],
                           box: Tuple[float,float],
                           *,
                           rmax_frac: float = 0.45,
                           nbins: int = 160) -> TranslationalDecision:
    # pick G from middle frame
    k = len(frames)//2
    gvec = select_G_from_Sk(frames[k][:,:2], box)
    G, k_shell = gvec.G, gvec.shell_k
    psi_abs = float(np.abs(psi_G(frames[k][:,:2], G)))

    # correlation c(r)
    rmax = rmax_frac * min(box[0], box[1])
    r, c_r = compute_c_of_r(frames, box, G, r_max=rmax, nbins=nbins)
    c_abs = np.abs(c_r.real)  # real part is customary since c(r) should be real by symmetry

    # sub-block scaling on representative frame
    x_sub, y_sub = psiG_subblock_scaling(frames[k][:,:2], box, G)

    # rough model selection:
    #   solid: power-law tail with η_G ≤ 1/3 AND sub-block curve ≤ (LB/L)^(-1/3)
    #   hexatic: c(r) decays ~exp (or steeper than any r^{-1/3}); sub-block above the line
    #   fluid: strong exponential decay; Ψ_G tiny
    # Use exponent estimate:
    eta_est = fit_powerlaw_eta(r, c_abs)
    ref = np.power(np.clip(x_sub, 1e-6, 1.0), -1.0/3.0) if len(x_sub)>0 else np.array([])
    below = float(np.mean(y_sub <= ref)) if len(ref)>0 else np.nan

    reason = []
    phase = "transition"
    if eta_est is not None and eta_est <= (1.0/3.0 + 0.03) and (len(ref)==0 or below >= 0.6):
        phase = "solid"
        reason.append(f"power-law tail η_G≈{eta_est:.2f} ≤ 1/3 and Ψ_G sub-block below (LB/L)^(-1/3)")
    else:
        # check for exponential-like tail: log c(r) concave with effective slope increasing
        # simple heuristic: linear fit is poor and late-bin average < 1e-3
        tail_slice = c_abs[int(0.7 * len(c_abs)):]
        tail_mean = float(np.mean(tail_slice)) if tail_slice.size else 0.0

        if tail_mean < 1e-3:
            phase = "fluid"
            reason.append("fast (≈exponential) decay of c(r) and tiny tail amplitude")
        else:
            phase = "hexatic"
            reason.append("no stable r^{-η} with η≤1/3; sub-block above separator")

    return TranslationalDecision(phase=phase,
                                 reason="; ".join(reason),
                                 etaG=eta_est,
                                 psiG_abs=psi_abs,
                                 x_sub=x_sub, y_sub=y_sub,
                                 G=G, k_shell=k_shell)
# =========================
# Translational: c(r) fits
# =========================
from dataclasses import dataclass
import numpy as np

@dataclass
class FitCGResult:
    model: str                 # 'const' | 'power' | 'exp'
    # power law: |c(r)| ~ A * r^{-eta_G}
    etaG_mean: float = np.nan
    logA_power: float = np.nan
    # exponential: |c(r)| ~ exp(alpha * r + logA)
    alpha_exp: float = np.nan
    logA_exp: float = np.nan
    # constant tail
    c_tail: float = np.nan
    # fit window (on r)
    rmin: float = np.nan
    rmax: float = np.nan
    # model selection info
    aic: float = np.nan
    delta_to_next: float = np.nan
    # notes & diagnostic points
    note: str = ""
    r_used: np.ndarray = None
    y_used: np.ndarray = None

def _cg_make_envelope(y_matrix: np.ndarray, q_env: float = 0.85, roll_env: int = 3) -> np.ndarray:
    """
    Envelope across rows (replicas/frames) -> 1D vector (length M).
    q_env in (0,1]: quantile (>=0.5 recommended); roll_env: rolling max half-window.
    """
    if y_matrix.ndim != 2:
        raise ValueError("y_matrix must be (R, M)")
    # quantile envelope
    env = np.quantile(y_matrix, q_env, axis=0)
    # simple rolling max smoothing
    if roll_env and roll_env > 1:
        w = int(roll_env)
        pad = w // 2
        buf = np.pad(env, (pad, pad), mode="edge")
        out = np.empty_like(env)
        for i in range(env.size):
            out[i] = np.max(buf[i:i + w])
        env = out
    return env

def _aic_from_residuals(residuals: np.ndarray, k_params: int) -> float:
    n = residuals.size
    if n <= k_params or n == 0:
        return np.inf
    rss = float(np.sum(residuals**2))
    if rss <= 0.0:
        rss = 1e-300
    return 2*k_params + n * (np.log(2*np.pi*rss/n) + 1.0)

def _fit_const(y: np.ndarray) -> tuple[float, float]:
    # mean-only constant model
    c = float(np.mean(y))
    res = y - c
    aic = _aic_from_residuals(res, k_params=1)
    return c, aic

def _fit_powerlaw(r: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    # log y = logA - eta * log r  (ignore r<=0 or y<=0)
    mask = (r > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 5:
        return np.nan, np.nan, np.inf
    X = np.log(r[mask]); Y = np.log(y[mask])
    A = np.vstack([np.ones_like(X), -X]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    logA, eta = coef[0], coef[1]
    yhat = np.exp(logA) * np.power(np.clip(r[mask], 1e-300, None), -eta)
    res = Y - np.log(np.clip(yhat, 1e-300, None))
    aic = _aic_from_residuals(res, k_params=2)
    return float(eta), float(logA), float(aic)

def _fit_exponential(r: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    # log y = alpha * r + logA
    mask = (r > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 5:
        return np.nan, np.nan, np.inf
    X = r[mask]; Y = np.log(y[mask])
    A = np.vstack([X, np.ones_like(X)]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    alpha, logA = coef[0], coef[1]
    yhat = np.exp(alpha * r[mask] + logA)
    res = Y - np.log(np.clip(yhat, 1e-300, None))
    aic = _aic_from_residuals(res, k_params=2)
    return float(alpha), float(logA), float(aic)

def fit_cG_models(
    r: np.ndarray,
    c_avg: np.ndarray,
    *,
    c_pf: np.ndarray | None = None,   # optional (R, M) “per-frame” curves for envelope like g6_pf
    start_bin: int = 3,               # analogous to START_BIN you use for g6
    q_env: float = 0.85,
    roll_env: int = 3,
    y_floor: float = 1e-12,
    y_cap: float = 1.0,
) -> FitCGResult:
    """
    Model selection for translational correlation |c(r)|:
        const, power law (r^{-eta_G}), exponential.
    Uses an envelope of c_pf if provided (same idea as g6_pf in your pipeline).
    """
    r = np.asarray(r, float)
    y = np.asarray(c_avg, float)
    if r.shape != y.shape:
        raise ValueError("r and c_avg must have the same shape")

    # clip for numerical stability (same philosophy as g6)
    y_clip = np.clip(y, y_floor, y_cap)

    # envelope points used for fitting
    if c_pf is not None:
        C = np.asarray(c_pf, float)
        if C.ndim != 2 or C.shape[1] != r.size:
            raise ValueError("c_pf must be (R, M) with M=len(r)")
        env = _cg_make_envelope(np.clip(C, y_floor, y_cap), q_env=q_env, roll_env=roll_env)
        y_used = np.clip(env, y_floor, y_cap)
        note = "envelope over provided curves"
    else:
        y_used = y_clip
        note = "single-curve fit"

    # tail window
    i0 = max(0, int(round(start_bin)))
    r_used = r[i0:]
    y_used = y_used[i0:]
    if r_used.size < 8:
        return FitCGResult(model="const", c_tail=float(np.mean(y_clip)), r_used=r_used, y_used=y_used,
                           note="insufficient points for robust fit")

    # Fit all three models
    ctail, aic_const = _fit_const(y_used)
    eta, logA_p, aic_pow = _fit_powerlaw(r_used, y_used)
    alpha, logA_e, aic_exp = _fit_exponential(r_used, y_used)

    aics = np.array([aic_const, aic_pow, aic_exp], float)
    models = np.array(["const", "power", "exp"])

    order = np.argsort(aics)
    best_idx = int(order[0])
    best = models[best_idx]
    delta_to_next = float(aics[order[1]] - aics[order[0]]) if aics.size >= 2 and np.isfinite(aics[order[1]]) else np.nan

    res = FitCGResult(
        model=str(best),
        etaG_mean=float(eta) if best == "power" else np.nan,
        logA_power=float(logA_p) if best == "power" else np.nan,
        alpha_exp=float(alpha) if best == "exp" else np.nan,
        logA_exp=float(logA_e) if best == "exp" else np.nan,
        c_tail=float(ctail),
        rmin=float(r_used[0]),
        rmax=float(r_used[-1]),
        aic=float(aics[best_idx]),
        delta_to_next=delta_to_next,
        note=note,
        r_used=r_used.copy(),
        y_used=y_used.copy(),
    )
    return res

def fit_cG_auto(
    r: np.ndarray,
    c_avg: np.ndarray,
    *,
    start_bin: int = 3,
    q_env: float = 0.85,
    roll_env: int = 3,
    y_floor: float = 1e-12,
    y_cap: float = 1.0,
) -> FitCGResult:
    """Convenience wrapper when you only have a single averaged curve."""
    return fit_cG_models(
        r, c_avg,
        c_pf=None, start_bin=start_bin, q_env=q_env, roll_env=roll_env,
        y_floor=y_floor, y_cap=y_cap,
    )
# ================================
# Translational: averaged c(r)
# ================================
def compute_cG_avg(frames: Sequence[np.ndarray],
                   box: Tuple[float, float],
                   *,
                   r_max: Optional[float] = None,
                   nbins: int = 160,
                   return_per_frame: bool = False,
                   G: Optional[np.ndarray] = None,
                   return_G: bool = False):
    """
    Compute the translational correlation |c(r)| averaged over frames.
    If G is not supplied, select it from the middle frame. Optionally return G.
    Returns:
        if return_per_frame and return_G:
            r, c_avg, c_pf, G
        if return_per_frame and not return_G:
            r, c_avg, c_pf
        if not return_per_frame and return_G:
            r, c_avg, None, G
        else:
            r, c_avg, None
    """
    if len(frames) == 0:
        raise ValueError("compute_cG_avg: empty frames sequence")

    # choose G from the representative (middle) frame if not provided
    if G is None:
        k_mid = len(frames) // 2
        gsel = select_G_from_Sk(np.asarray(frames[k_mid])[:, :2], box)
        G = np.asarray(gsel.G, float)

    # averaged curve over all frames
    r, c_mean = compute_c_of_r(frames, box, G, r_max=r_max, nbins=nbins)
    c_avg = np.abs(np.real(c_mean))

    if not return_per_frame:
        if return_G:
            return r, c_avg, None, G
        else:
            return r, c_avg, None

    # per-frame curves (identical binning)
    c_list = []
    for R in frames:
        r_i, c_i = compute_c_of_r([np.asarray(R)[:, :2]], box, G, r_max=r_max, nbins=nbins)
        if r_i.shape != r.shape or np.max(np.abs(r_i - r)) > 1e-12:
            c_list.append(np.interp(r, r_i, np.abs(np.real(c_i))))
        else:
            c_list.append(np.abs(np.real(c_i)))
    c_pf = np.vstack(c_list)

    if return_G:
        return r, c_avg, c_pf, G
    else:
        return r, c_avg, c_pf
