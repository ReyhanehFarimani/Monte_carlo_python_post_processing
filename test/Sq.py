# tests/test_structure_factor_peaks.py
import os
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Public API
from gcmc_post_processing import (
    structure_factor,           # 1D S(q)
    plot_structure_factor,      # 1D plot
    save_table_csv,
    structure_factor_2d,        # 2D S(kx,ky)
    plot_structure_factor_2d,   # 2D heatmap
)

# ---------------- Lattice builders (a = NN distance) ----------------

def make_square_lattice(a: float, nx: int, ny: int):
    i, j = np.indices((nx, ny))
    x = a * i
    y = a * j
    pos = np.column_stack([x.ravel(), y.ravel()])
    Lx, Ly = nx * a, ny * a
    pos[:, 0] %= Lx; pos[:, 1] %= Ly
    return pos, (Lx, Ly)

def make_triangular_lattice(a: float, nx: int, ny: int):
    i, j = np.indices((nx, ny))
    x = a * (i + 0.5 * (j & 1))
    y = (np.sqrt(3.0) / 2.0) * a * j
    pos = np.column_stack([x.ravel(), y.ravel()])
    Lx, Ly = nx * a, ny * (np.sqrt(3.0) / 2.0) * a
    pos[:, 0] %= Lx; pos[:, 1] %= Ly
    return pos, (Lx, Ly)

def make_honeycomb_lattice(a: float, nx: int, ny: int):
    i, j = np.indices((nx, ny))
    xA = (3.0 / 2.0) * a * i
    yA = (np.sqrt(3.0) / 2.0) * a * (i + 2 * j)
    xB = xA + a
    yB = yA
    x = np.concatenate([xA.ravel(), xB.ravel()])
    y = np.concatenate([yA.ravel(), yB.ravel()])
    pos = np.column_stack([x, y])
    Lx, Ly = (3.0 / 2.0) * a * nx, (np.sqrt(3.0)) * a * ny
    pos[:, 0] %= Lx; pos[:, 1] %= Ly
    return pos, (Lx, Ly)

def expected_q1(lattice: str, a: float) -> float:
    if lattice == "square":
        return 2.0 * np.pi / a
    elif lattice in ("triangular", "honeycomb"):
        return 4.0 * np.pi / (np.sqrt(3.0) * a)
    raise ValueError(lattice)

# ---------------- Artifacts + quick real-space plot ----------------

def _artifacts_dir():
    d = os.environ.get("ARTIFACTS_DIR", "artifacts")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)

def plot_real_space(pos, box, title: str, outpath: Path, s=2.0, alpha=0.9, show=False):
    Lx, Ly = box
    fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
    ax.scatter(pos[:, 0], pos[:, 1], s=s, c="k", alpha=alpha, linewidths=0)
    ax.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], "C0-", lw=1.0, alpha=0.6)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(outpath, dpi=300)
    if show: plt.show()
    plt.close(fig)

# ---------------- Core test: 1D S(q) + 2D S with ONE q-vector ----------------

@pytest.mark.parametrize("lattice", ["triangular", "square", "honeycomb"])
@pytest.mark.parametrize("jitter_sigma, rel_tol", [
    (0.00, 0.05),   # perfect crystal
    (0.07, 0.15),   # broader peaks (hexatic-like)
])
def test_first_peak_and_single_q_vector(lattice, jitter_sigma, rel_tol):
    rng = np.random.default_rng(12345)
    a = 1.0

    if lattice == "square":
        nx, ny = 60, 60
        pos, box = make_square_lattice(a, nx, ny)
    elif lattice == "triangular":
        nx, ny = 50, 50
        pos, box = make_triangular_lattice(a, nx, ny)
    elif lattice == "honeycomb":
        nx, ny = 50, 50
        pos, box = make_honeycomb_lattice(a, nx, ny)
    else:
        raise AssertionError

    # jitter
    pos = pos + jitter_sigma * rng.standard_normal(pos.shape)

    ad = _artifacts_dir()
    plot_real_space(pos, box, f"{lattice} (a={a}, j={jitter_sigma})",
                    ad / f"realspace_{lattice}_j{jitter_sigma:.2f}.png", show=False)

    # ---- 1D S(q) check ----
    res = structure_factor(positions=pos, box=box, nbins=240)
    q, Sq = res.q, res.Sq
    mask = q > 1e-8
    pk_ids, _ = find_peaks(Sq[mask], prominence=0.2)
    assert pk_ids.size > 0, "No peaks detected in S(q)."
    j = pk_ids[np.argmax(Sq[mask][pk_ids])]
    q_meas = q[mask][j]
    q_theory = expected_q1(lattice, a)
    rel_err = abs(q_meas - q_theory) / q_theory
    assert rel_err < rel_tol, f"{lattice}: q* {q_meas:.4f} vs {q_theory:.4f} (rel {rel_err:.3f})"

    ax = plot_structure_factor(q, Sq, annotate_peak=False)
    ax.axvline(q_theory, color="C3", ls="--", lw=1.2, label=fr"expected $q^*={q_theory:.3f}$")
    ax.axvline(q_meas, color="C1", ls="-.", lw=1.2, label=fr"measured $q^*={q_meas:.3f}$")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title(f"{lattice}, j={jitter_sigma} (rel err={rel_err:.3f})")
    ax.figure.savefig(ad / f"Sq_{lattice}_j{jitter_sigma:.2f}.png", dpi=300)
    plt.close(ax.figure)
    save_table_csv(ad / f"Sq_{lattice}_j{jitter_sigma:.2f}.csv", [q, Sq], header=["q", "S(q)"])

    # ---- 2D S(kx,ky): pick ONE vector (strongest on first ring) + quiver overlay ----
    s2d = structure_factor_2d(positions=pos, box=box, nmax=64, exclude_k0=True)

    # Build magnitude grid and a ring mask around q*
    KX, KY = np.meshgrid(s2d.kx, s2d.ky, indexing="ij")  # shapes (Nkx, Nky)
    KMAG = np.hypot(KX, KY)
    ring_hw = 0.12 * q_theory           # ±12% around q*
    ring_mask = (np.abs(KMAG - q_theory) <= ring_hw) & (KMAG >= 0.2 * q_theory)

    # Fallback: if mask empty (very noisy), use global max away from center
    if not np.any(ring_mask):
        valid = KMAG >= 0.2 * q_theory
        ii, jj = np.unravel_index(np.nanargmax(np.where(valid, s2d.S, -np.inf)), s2d.S.shape)
    else:
        ii, jj = np.unravel_index(np.nanargmax(np.where(ring_mask, s2d.S, -np.inf)), s2d.S.shape)

    qx_first = float(s2d.kx[ii])
    qy_first = float(s2d.ky[jj])
    q_first  = float(np.hypot(qx_first, qy_first))
    S_first  = float(s2d.S[ii, jj])

    # Save the single vector
    save_table_csv(ad / f"q_first_{lattice}_j{jitter_sigma:.2f}.csv",
                   [np.array([qx_first]), np.array([qy_first]), np.array([q_first]), np.array([S_first])],
                   header=["qx", "qy", "q", "S"])

    # Zoomed ring view with a quiver arrow from origin to (qy, qx)
    # NOTE: plot_structure_factor_2d displays x=ky, y=kx
    ring_w = 0.4 * q_theory
    ax2 = plot_structure_factor_2d(
        s2d.kx, s2d.ky, s2d.S, log10=True,
        title=f"S2D {lattice} (j={jitter_sigma:.2f}) — first max",
        xlim=(q_theory - ring_w, q_theory + ring_w),   # ky-axis window
        ylim=(-ring_w, ring_w),                        # kx-axis window
        vmin = 0, vmax = 100
    )
    # Quiver: base at (0,0) → tip at (ky, kx) on the displayed axes
    ax2.quiver(0.0, 0.0, qy_first, qx_first, angles='xy', scale_units='xy', scale=1.0,
               color='w', width=1, headwidth=6, headlength=7)
    # Mark the tip explicitly
    ax2.plot([qy_first], [qx_first], "wo", ms=4.0, mec="k", mew=0.6, alpha=0.95)
    ax2.figure.savefig(ad / f"S2D_{lattice}_j{jitter_sigma:.2f}_first_vector.png", dpi=300)
    plt.close(ax2.figure)

# ---------------- Script mode: quick gallery ----------------

def main(show=False):
    rng = np.random.default_rng(2024)
    a = 1.0
    cases = [
        ("triangular", 50, 50, 0.01),
        ("triangular", 50, 50, 0.10),
        ("square",     60, 60, 0.01),
        ("square",     60, 60, 0.10),
        ("honeycomb",  50, 50, 0.01),
        ("honeycomb",  50, 50, 0.10),
    ]
    ad = _artifacts_dir()

    for lattice, nx, ny, jitter in cases:
        if lattice == "square":
            pos, box = make_square_lattice(a, nx, ny)
        elif lattice == "triangular":
            pos, box = make_triangular_lattice(a, nx, ny)
        elif lattice == "honeycomb":
            pos, box = make_honeycomb_lattice(a, nx, ny)
        else:
            continue

        pos_noisy = pos + jitter * rng.standard_normal(pos.shape)

        # real space
        plot_real_space(pos_noisy, box, f"{lattice} (a={a}, j={jitter})",
                        ad / f"realspace_{lattice}_j{jitter:.2f}.png", show=show)

        # 1D S(q)
        res = structure_factor(pos_noisy, box, nbins=240)
        q, Sq = res.q, res.Sq
        mask = q > 1e-8
        pk_ids, _ = find_peaks(Sq[mask], prominence=0.2)
        q_theory = expected_q1(lattice, a)
        q_meas = q[mask][pk_ids[np.argmax(Sq[mask][pk_ids])]] if pk_ids.size else np.nan

        ax = plot_structure_factor(q, Sq, annotate_peak=False)
        ax.axvline(q_theory, color="C3", ls="--", lw=1.2, label=fr"expected $q^*={q_theory:.3f}$")
        if not np.isnan(q_meas):
            ax.axvline(q_meas, color="C1", ls="-.", lw=1.2, label=fr"measured $q^*={q_meas:.3f}$")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.set_title(f"{lattice}, j={jitter}")
        ax.figure.savefig(ad / f"Sq_{lattice}_j{jitter:.2f}.png", dpi=300)
        if show: plt.show()
        plt.close(ax.figure)
        save_table_csv(ad / f"Sq_{lattice}_j{jitter:.2f}.csv", [q, Sq], header=["q", "S(q)"])

        # 2D S(kx,ky): single vector
        s2d = structure_factor_2d(pos_noisy, box, nmax=96, exclude_k0=True, batch=256)
        KX, KY = np.meshgrid(s2d.kx, s2d.ky, indexing="ij")
        KMAG = np.hypot(KX, KY)
        ring_hw = 0.12 * q_theory
        ring_mask = (np.abs(KMAG - q_theory) <= ring_hw) & (KMAG >= 0.2 * q_theory)
        if not np.any(ring_mask):
            valid = KMAG >= 0.2 * q_theory
            ii, jj = np.unravel_index(np.nanargmax(np.where(valid, s2d.S, -np.inf)), s2d.S.shape)
        else:
            ii, jj = np.unravel_index(np.nanargmax(np.where(ring_mask, s2d.S, -np.inf)), s2d.S.shape)

        qx_first = float(s2d.kx[ii]); qy_first = float(s2d.ky[jj])
        q_first  = float(np.hypot(qx_first, qy_first)); S_first = float(s2d.S[ii, jj])

        # save vector
        save_table_csv(ad / f"q_first_{lattice}_j{jitter:.2f}.csv",
                       [np.array([qx_first]), np.array([qy_first]), np.array([q_first]), np.array([S_first])],
                       header=["qx", "qy", "q", "S"])

        # ring zoom + quiver overlay
        ring_w = 1.4 * q_theory
        ax2 = plot_structure_factor_2d(
            s2d.kx, s2d.ky, s2d.S, log10=True,
            title=f"S2D {lattice} (j={jitter:.2f}) — first max",
            xlim=(- ring_w,  ring_w),
            ylim=(-ring_w, ring_w),
            vmin = 0,
            # vmax = 100,
        )
        ax2.quiver(0.0, 0.0, qy_first, qx_first, angles='xy', scale_units='xy', scale=1.0,
                   color='w', width=0.01, headwidth=6, headlength=7)
        ax2.plot([qy_first], [qx_first], "wo", ms=4.0, mec="k", mew=0.6, alpha=0.95)
        ax2.figure.savefig(ad / f"S2D_{lattice}_j{jitter:.2f}_first_vector.png", dpi=300)
        if show: plt.show()
        plt.close(ax2.figure)

if __name__ == "__main__":
    main(show=True)
