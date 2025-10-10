# tests/test_translational_order.py
import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

# Public API (must be exposed by your package)
from gcmc_post_processing import (
    structure_factor,
    structure_factor_2d,
    plot_structure_factor_2d,
    save_table_csv,
    # translational pipeline
    psi_G,
    compute_c_of_r,
    psiG_subblock_scaling,
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

# ---------------- Artifacts ----------------

def _artifacts_dir():
    d = os.environ.get("ARTIFACTS_DIR", "artifacts")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)

# ---------------- Core translational-order test ----------------

@pytest.mark.parametrize("lattice", ["triangular", "square", "honeycomb"])
def test_translational_order_crystal_vs_jitter(lattice):
    """
    For each lattice:
      - pick G from S2D first ring near theoretical q*
      - |Psi_G| (crystal) > |Psi_G| (jitter)
      - tail of c(r) (crystal) > tail of c(r) (jitter)
      - sub-block median ratio (crystal) < (jitter)
    """
    rng = np.random.default_rng(2024)
    a = 1.0
    # system sizes match your structure-factor test
    if lattice == "square":
        nx, ny = 60, 60
        pos0, box = make_square_lattice(a, nx, ny)
    elif lattice == "triangular":
        nx, ny = 50, 50
        pos0, box = make_triangular_lattice(a, nx, ny)
    elif lattice == "honeycomb":
        nx, ny = 50, 50
        pos0, box = make_honeycomb_lattice(a, nx, ny)
    else:
        raise AssertionError

    # two states: perfect crystal vs jittered (hexatic-like)
    pos_crys = pos0.copy()
    pos_jit  = pos0 + 0.07 * rng.standard_normal(pos0.shape)

    # ---- pick one G-vector from S2D, using ring around q_theory ----
    q_theory = expected_q1(lattice, a)

    def pick_G(pos, box, nmax=64):
        s2d = structure_factor_2d(positions=pos, box=box, nmax=nmax, exclude_k0=True)
        KX, KY = np.meshgrid(s2d.kx, s2d.ky, indexing="ij")
        KMAG = np.hypot(KX, KY)
        ring_hw = 0.12 * q_theory
        ring_mask = (np.abs(KMAG - q_theory) <= ring_hw) & (KMAG >= 0.2 * q_theory)
        if not np.any(ring_mask):
            valid = KMAG >= 0.2 * q_theory
            ii, jj = np.unravel_index(np.nanargmax(np.where(valid, s2d.S, -np.inf)), s2d.S.shape)
        else:
            ii, jj = np.unravel_index(np.nanargmax(np.where(ring_mask, s2d.S, -np.inf)), s2d.S.shape)
        qx, qy = float(s2d.kx[ii]), float(s2d.ky[jj])
        return np.array([qx, qy], dtype=float), s2d

    Gc, s2d_c = pick_G(pos_crys, box)
    Gj, s2d_j = pick_G(pos_jit,  box)

    # sanity: selected |G| near q_theory for crystal
    q_meas = float(np.linalg.norm(Gc))
    rel_err = abs(q_meas - q_theory) / q_theory
    assert rel_err < 0.15, f"{lattice}: |G| mismatch: {q_meas:.4f} vs {q_theory:.4f}"

    # ---- Ψ_G on each state (single frame is sufficient here) ----
    psi_c = abs(psi_G(pos_crys, Gc))
    psi_j = abs(psi_G(pos_jit,  Gj))
    # crystal should have markedly larger Ψ_G
    assert psi_c > 0.20, f"{lattice}: Ψ_G too small for crystal ({psi_c:.3f})"
    assert psi_c > 1.5 * psi_j, f"{lattice}: Ψ_G did not drop enough with jitter (crys {psi_c:.3f} vs jit {psi_j:.3f})"

    # ---- c(r) vs r on each state (use same nbins/rmax for fair comparison) ----
    rmax = 0.45 * min(box[0], box[1])
    nbins = 160
    r_c, c_c = compute_c_of_r([pos_crys], box, Gc, r_max=rmax, nbins=nbins)
    r_j, c_j = compute_c_of_r([pos_jit],  box, Gj, r_max=rmax, nbins=nbins)

    c_c_abs = np.abs(np.real(c_c))
    c_j_abs = np.abs(np.real(c_j))

    # compare tails: last 20% of bins
    t0 = int(0.8 * nbins)
    tail_c = float(np.nanmean(c_c_abs[t0:]))
    tail_j = float(np.nanmean(c_j_abs[t0:]))

    assert tail_c > 1.2 * tail_j, f"{lattice}: c(r) tail not stronger in crystal (crys {tail_c:.3e}, jit {tail_j:.3e})"

    # ---- Sub-block scaling: crystal median < jitter median ----
    x_c, y_c = psiG_subblock_scaling(pos_crys, box, Gc)
    x_j, y_j = psiG_subblock_scaling(pos_jit,  box, Gj)
    # guard against empty returns (should not happen)
    assert x_c.size and x_j.size and y_c.size and y_j.size
    med_c = float(np.median(y_c))
    med_j = float(np.median(y_j))
    assert med_c < med_j, f"{lattice}: sub-block median not reduced for crystal (crys {med_c:.3f} vs jit {med_j:.3f})"

    # ---- Artifacts: CSV + plots for inspection ----
    ad = _artifacts_dir()

    # Save selected G vectors
    save_table_csv(ad / f"G_selected_{lattice}.csv",
                   [np.array([Gc[0], Gj[0]]), np.array([Gc[1], Gj[1]])],
                   header=["qx_crystal,qx_jitter", "qy_crystal,qy_jitter"])

    # S2D ring views with quivers (crystal + jitter)
    def ring_plot(s2d, G, tag):
        qx, qy = float(G[0]), float(G[1])
        ring_w = 0.4 * q_theory
        ax = plot_structure_factor_2d(
            s2d.kx, s2d.ky, s2d.S, log10=True,
            title=f"S2D {lattice} — {tag}",
            xlim=(q_theory - ring_w, q_theory + ring_w),
            ylim=(-ring_w, ring_w),
            vmin=0, vmax=100
        )
        ax.quiver(0.0, 0.0, qy, qx, angles='xy', scale_units='xy', scale=1.0,
                  color='w', width=1, headwidth=6, headlength=7)
        ax.plot([qy], [qx], "wo", ms=4.0, mec="k", mew=0.6, alpha=0.95)
        ax.figure.savefig(ad / f"S2D_{lattice}_{tag}.png", dpi=300)
        plt.close(ax.figure)

    ring_plot(s2d_c, Gc, "crystal")
    ring_plot(s2d_j, Gj, "jitter")

    # c(r) plot
    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    ax.plot(r_c, c_c_abs, "C0-", lw=1.4, label="crystal")
    ax.plot(r_j, c_j_abs, "C1-", lw=1.4, label="jitter")
    ax.set_xlabel("r"); ax.set_ylabel(r"$|c(r)|$")
    ax.set_title(f"{lattice}: tail_crys={tail_c:.2e}, tail_jit={tail_j:.2e}")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(ad / f"c_of_r_{lattice}.png", dpi=300)
    plt.close(fig)

    # sub-block plot
    fig2, ax2 = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    ax2.plot(x_c, y_c, "o-", ms=3.0, label="crystal")
    ax2.plot(x_j, y_j, "o-", ms=3.0, label="jitter")
    xx = np.linspace(min(x_c.min(), x_j.min()), 1.0, 50)
    ax2.plot(xx, xx**(-1.0/3.0), "k--", lw=1.0, label=r"$(L_B/L)^{-1/3}$")
    ax2.set_xlabel(r"$L_B/L$"); ax2.set_ylabel(r"$\Psi_G^2(L_B)/\Psi_G^2(L)$")
    ax2.set_title(f"{lattice}: med_crys={med_c:.3f}, med_jit={med_j:.3f}")
    ax2.legend(frameon=False, fontsize=8, loc="best")
    fig2.savefig(ad / f"psiG_subblock_{lattice}.png", dpi=300)
    plt.close(fig2)


def main(show: bool = False):
    rng = np.random.default_rng(2024)
    a = 1.0
    cases = [
        ("triangular", 50, 50),
        ("square",     60, 60),
        ("honeycomb",  50, 50),
    ]
    jitters = [0.00, 0.07]  # crystal vs jittered

    def make_lattice(name: str, a: float, nx: int, ny: int):
        if name == "square":
            return make_square_lattice(a, nx, ny)
        elif name == "triangular":
            return make_triangular_lattice(a, nx, ny)
        elif name == "honeycomb":
            return make_honeycomb_lattice(a, nx, ny)
        else:
            raise ValueError(name)

    def pick_G(pos, box, q_star, nmax=100):
        s2d = structure_factor_2d(positions=pos, box=box, nmax=nmax, exclude_k0=True)
        KX, KY = np.meshgrid(s2d.kx, s2d.ky, indexing="ij")
        KMAG = np.hypot(KX, KY)
        ring_hw = 0.12 * q_star
        ring_mask = (np.abs(KMAG - q_star) <= ring_hw) & (KMAG >= 0.2 * q_star)
        if not np.any(ring_mask):
            valid = KMAG >= 0.2 * q_star
            ii, jj = np.unravel_index(np.nanargmax(np.where(valid, s2d.S, -np.inf)), s2d.S.shape)
        else:
            ii, jj = np.unravel_index(np.nanargmax(np.where(ring_mask, s2d.S, -np.inf)), s2d.S.shape)
        qx, qy = float(s2d.kx[ii]), float(s2d.ky[jj])
        return np.array([qx, qy], dtype=float), s2d

    ad = _artifacts_dir()
    print("=== Translational-order gallery ===")

    for lattice, nx, ny in cases:
        pos0, box = make_lattice(lattice, a, nx, ny)
        q_theory = expected_q1(lattice, a)
        print(f"\n[{lattice}] a={a}, size=({nx},{ny}), q*={q_theory:.6f}")

        # build states
        states = {
            "crystal": pos0.copy(),
            "jitter":  pos0 + 0.07 * rng.standard_normal(pos0.shape),
            "highjitter":  pos0 + 0.2 * rng.standard_normal(pos0.shape),
        }

        # select G for each state, compute Ψ_G, c(r), sub-blocks, and save artifacts
        results = {}
        for tag, pos in states.items():
            G, s2d = pick_G(pos, box, q_theory, nmax=64)
            psi = float(abs(psi_G(pos, G)))

            rmax = 0.45 * min(box[0], box[1])
            nbins = 160
            r, c = compute_c_of_r([pos], box, G, r_max=rmax, nbins=nbins)
            c_abs = np.abs(np.real(c))
            tail = float(np.nanmean(c_abs[int(0.8 * nbins):]))

            x_sub, y_sub = psiG_subblock_scaling(pos, box, G)

            # save G
            save_table_csv(ad / f"G_selected_{lattice}_{tag}.csv",
                           [np.array([G[0]]), np.array([G[1]])],
                           header=["qx", "qy"])

            # ring plot
            ring_w = 1.4 * q_theory
            ax = plot_structure_factor_2d(
                s2d.kx, s2d.ky, s2d.S, log10=True,
                title=f"S2D {lattice} — {tag}",
                xlim=(- ring_w, ring_w),
                ylim=(-ring_w, ring_w),
                vmin=0, 
                cmap = 'cividis',
            )
            ax.quiver(0.0, 0.0, G[1], G[0], angles='xy', scale_units='xy', scale=1.0,
                      color='w', width=0.01, headwidth=6, headlength=7)
            ax.plot([G[1]], [G[0]], "wo", ms=4.0, mec="k", mew=0.6, alpha=0.95)
            fpath = ad / f"S2D_{lattice}_{tag}.png"
            ax.figure.savefig(fpath, dpi=300)
            if show: plt.show()
            plt.close(ax.figure)

            # c(r) plot
            fig, axc = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
            axc.plot(r, c_abs, "-", lw=1.4, label=tag)
            axc.set_xlabel("r"); axc.set_ylabel(r"$|c(r)|$")
            axc.set_title(f"{lattice} — {tag}: tail={tail:.2e}")
            axc.legend(frameon=False, fontsize=8, loc="best")
            fig.savefig(ad / f"c_of_r_{lattice}_{tag}.png", dpi=300)
            if show: plt.show()
            plt.close(fig)

            # sub-block plot
            fig2, ax2 = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
            ax2.plot(x_sub, y_sub, "o-", ms=3.0, label=tag)
            xx = np.linspace(max(1e-3, x_sub.min() if x_sub.size else 1e-3), 1.0, 50)
            ax2.plot(xx, xx**(-1.0/3.0), "k--", lw=1.0, label=r"$(L_B/L)^{-1/3}$")
            ax2.set_xlabel(r"$L_B/L$"); ax2.set_ylabel(r"$\Psi_G^2(L_B)/\Psi_G^2(L)$")
            ax2.set_title(f"{lattice} — {tag}")
            ax2.legend(frameon=False, fontsize=8, loc="best")
            fig2.savefig(ad / f"psiG_subblock_{lattice}_{tag}.png", dpi=300)
            if show: plt.show()
            plt.close(fig2)

            results[tag] = dict(G=G, psi=psi, tail=tail)

        # print quick comparison summary
        Gc = results["crystal"]["G"]; Gj = results["jitter"]["G"]
        psi_c, psi_j = results["crystal"]["psi"], results["jitter"]["psi"]
        tail_c, tail_j = results["crystal"]["tail"], results["jitter"]["tail"]
        print(f"  |G_crys|={np.linalg.norm(Gc):.6f}, |G_jit|={np.linalg.norm(Gj):.6f}")
        print(f"  |Ψ_G|: crystal={psi_c:.4f}, jitter={psi_j:.4f}, ratio={psi_c/(psi_j+1e-16):.2f}")
        print(f"  tail ⟨|c(r)|⟩: crystal={tail_c:.3e}, jitter={tail_j:.3e}, ratio={(tail_c/(tail_j+1e-16)):.2f}")

if __name__ == "__main__":
    main(show=True)
