import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import freud

# Step 1: Set Global Visualization Settings
plt.rcParams.update({
    'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'grid.linestyle': '--',
    'grid.color': 'gray',
    'grid.alpha': 0.2,
})

def plot_data(ax, df, x_column, y_column, yerr_column=None, xlabel="", ylabel="", title="", plot_type="errorbar", **kwargs):
    """
    General function to plot data on a provided ax.

    Parameters:
    - ax: Matplotlib axis object to plot on.
    - df: DataFrame containing the data to plot.
    - x_column: Name of the column in df to use for the x-axis.
    - y_column: Name of the column in df to use for the y-axis.
    - yerr_column: (Optional) Name of the column in df to use for y-axis error bars.
    - xlabel: (Optional) Label for the x-axis.
    - ylabel: (Optional) Label for the y-axis.
    - title: (Optional) Title for the plot.
    - plot_type: Type of plot ("errorbar", "scatter", "line").
    - **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    
    if plot_type == "errorbar":
        ax.errorbar(df[x_column], df[y_column], yerr=df[yerr_column] if yerr_column else None, fmt=kwargs.get('fmt', 'o-'), label=kwargs.get('label', 'Data'))
    elif plot_type == "scatter":
        ax.scatter(df[x_column], df[y_column], label=kwargs.get('label', 'Data'))
    elif plot_type == "line":
        ax.plot(df[x_column], df[y_column], label=kwargs.get('label', 'Data'))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
def plot_rdf(rdf, r_bins):
    """
    Plots the radial distribution function g(r).

    Parameters
    ----------
    rdf : np.ndarray
        The computed g(r) values.
    r_bins : np.ndarray
        The radii corresponding to the g(r) values.

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(r_bins, rdf, '-o')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function (RDF)')
    plt.grid(True)
    plt.show()

def plot_voronoi_with_ids(points, box_size, neighbors):
    """
    Plots the Voronoi diagram with particle IDs and shows the Voronoi neighbors using freud.

    Parameters
    ----------
    points : np.ndarray
        Array of points with shape (N, 2), where N is the number of points.
    box_size : tuple of float
        The size of the box in the x and y dimensions.
    neighbors : dict
        A dictionary where each key is a point index, and the value is a list of neighboring point indices.

    Returns
    -------
    None
    """
    # Define the box and Voronoi analysis
    box = freud.box.Box(Lx=box_size[0], Ly=box_size[1], Lz = 0.0)
    voronoi = freud.locality.Voronoi(box)
    
    # Compute the Voronoi diagram
    voronoi.compute(system=(box, points))
    
    # Plot Voronoi diagram
    fig, ax = plt.subplots()
    for poly in voronoi.polytopes:
        polygon = np.asarray(poly)
        polygon = np.vstack([polygon, polygon[0]])  # Close the polygon
        ax.plot(polygon[:, 0], polygon[:, 1], color='orange', lw=2)
    
    # Plot points and their IDs
    for i, point in enumerate(points):
        ax.plot(point[0], point[1], 'o', color='black')
        ax.text(point[0] + 0.1, point[1] + 0.1, str(i), color='blue', fontsize=12)
        
        # Annotate Voronoi neighbors
        neighbor_ids = neighbors[i]
        neighbor_text = f"{', '.join(map(str, neighbor_ids))}"
        ax.text(point[0] + 0.1, point[1] - 0.3, neighbor_text, color='red', fontsize=8)

    ax.set_xlim([0, box_size[0]])
    ax.set_ylim([0, box_size[1]])
    
    ax.set_aspect('equal')
    ax.set_title('Voronoi Diagram with Periodic Boundary Conditions (freud)')
    plt.show()


def user_defined_voronoi_plot(box, polytopes, ax=None, color_by_sides=True, cmap=None, color_array = None):
    """Helper function to draw 2D Voronoi diagram.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        polytopes (:class:`numpy.ndarray`):
            Array containing Voronoi polytope vertices.
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).
        color_by_sides (bool):
            If :code:`True`, color cells by the number of sides.
            If :code:`False`, random colors are used for each cell.
            (Default value = :code:`True`).
        cmap (str):
            Colormap name to use (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    from matplotlib import cm
    from matplotlib.collections import PatchCollection
    from matplotlib.colorbar import Colorbar
    from matplotlib.patches import Polygon
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    # Draw Voronoi polytopes
    patches = [Polygon(poly[:, :2]) for poly in polytopes]
    patch_collection = PatchCollection(patches, edgecolors="black", alpha=0.4)

    if color_by_sides:
        colors = np.array([len(poly) for poly in polytopes])
        num_colors = np.ptp(colors) + 1
    else:
        colors = np.random.RandomState().permutation(np.arange(len(patches)))
        num_colors = np.unique(colors).size
    if color_array is not None:
        colors = color_array
        num_colors = np.unique(colors).size

    # Ensure we have enough colors to uniquely identify the cells
    if cmap is None:
        if color_by_sides and num_colors <= 10:
            cmap = "tab10"
        else:
            if num_colors > 20:
                warnings.warn(
                    "More than 20 unique colors were requested. "
                    "Consider providing a colormap to the cmap "
                    "argument.",
                    UserWarning,
                )
            cmap = "tab20"
    cmap = cm.get_cmap(cmap, num_colors)
    bounds = np.arange(-1, 2)

    patch_collection.set_array(np.array(colors))
    patch_collection.set_cmap(cmap)
    patch_collection.set_clim(bounds[0], bounds[-1] )
    ax.add_collection(patch_collection)

    # Draw box
    corners = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    # Need to copy the last point so that the box is closed.
    corners.append(corners[0])
    corners = box.make_absolute(corners)[:, :2]
    ax.plot(corners[:, 0], corners[:, 1], color="k")

    # Set title, limits, aspect
    ax.set_title("Voronoi Diagram")
    ax.set_xlim((np.min(corners[:, 0]), np.max(corners[:, 0])))
    ax.set_ylim((np.min(corners[:, 1]), np.max(corners[:, 1])))
    ax.set_aspect("equal", "datalim")

    # Add colorbar for number of sides
    if color_by_sides:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="10%")
        cb = Colorbar(cax, patch_collection)
        cb.set_label(r"$\vec{\Psi}(\rho, f).\vec{\psi}(\vec{r})$")
        cb.set_ticks(bounds)
    return ax


def plot_voro(ts, box, filename, label):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    points = ts - box/2
    points[:, 2] = 0
    box = freud.box.Box(Ly = box[1], Lx= box[0])
    voro = freud.locality.Voronoi()
    op = freud.order.Hexatic(k=6)
    cells = voro.compute((box, points)).polytopes
    op.compute( (box , points),neighbors=voro.nlist)
    sigma_6 = op.particle_order
    S = sigma_6.mean()
    colors = np.array([(s.real * S.real + s.imag * S.imag)/(np.absolute(s)*np.absolute(S)) for s in sigma_6])
    user_defined_voronoi_plot(box = box, polytopes= cells, ax = ax, color_array=colors, cmap = "terrain")
    ax.quiver(points[:, 0], points[:, 1], np.real(sigma_6),np.imag(sigma_6),color = "k",  label = label, scale = 100, alpha = 0.6)
    ax.legend(loc = "upper center")
    plt.savefig(filename + ".pdf", dpi = 800)
    plt.savefig(filename + ".eps", dpi = 800)
    plt.show()
    
    


# =============================================================================
# Reyhaneh — ORIENTATIONAL ORDER PLOTS
# =============================================================================

def plot_psi6_vs_density(table, xkey="density", xlabel=r"$\rho$", ylabel=r"$\langle |{\Psi}_6| \rangle$",
                         title="Orientational order vs density"):
    """
    Errorbar plot of <|Ψ₆|> as a function of density (or μ).

    Parameters
    ----------
    table : np.ndarray (structured)
        Output of scan_by_density_or_mu.
    xkey : str
        Name of the x column ('density' or 'mu').
    """
    x = table[xkey]
    y = table["psi6_mean_abs"]
    e = table["psi6_stderr_abs"]

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=e, fmt='o-', capsize=3, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_binder_vs_density(table, xkey="density", xlabel=r"$\rho$", ylabel=r"$U_4^{(6)}$",
                           title="Binder cumulant (hexatic) vs density"):
    """
    Plot Binder cumulant U₄ for Ψ₆ against density (or μ).
    """
    x = table[xkey]
    y = table["U4"]

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_chi6_vs_density(table, xkey="density", xlabel=r"$\rho$", ylabel=r"$\chi_6$",
                         title="Susceptibility (hexatic) vs density"):
    """
    Plot susceptibility χ₆ for Ψ₆ against density (or μ).
    """
    x = table[xkey]
    y = table["chi6"]

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Reyhaneh — ORIENTATIONAL ORDER PLOTS (single dataset)
# =============================================================================

def plot_psi6_time_series(psi6_series,
                          xlabel="frame",
                          ylabel=r"$|{\Psi}_6|$",
                          title=r"$|{\Psi}_6|$ time series"):
    """
    Plot |Ψ6| vs frame index.
    """
    y = np.abs(np.asarray(psi6_series))
    fig, ax = plt.subplots()
    ax.plot(y, "-o", lw=1, ms=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_psi6_abs_histogram(psi6_series, bins=30,
                            xlabel=r"$|{\Psi}_6|$",
                            ylabel="count",
                            title=r"Histogram of $|{\Psi}_6|$"):
    """
    Histogram of |Ψ6| across frames (useful for spotting bimodality).
    """
    y = np.abs(np.asarray(psi6_series))
    fig, ax = plt.subplots()
    ax.hist(y, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_binder_scalar(U4,
                       xlabel="",
                       ylabel=r"$U_4^{(6)}$",
                       title=r"Binder cumulant $U_4^{(6)}$ (single dataset)"):
    """
    Display the Binder cumulant as a simple bar (single number).
    """
    fig, ax = plt.subplots()
    ax.bar([0], [U4], width=0.4)
    ax.set_xticks([0])
    ax.set_xticklabels([xlabel] if xlabel else [""])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    # annotate value
    ax.text(0, U4, f"{U4:.4f}", ha="center", va="bottom")
    return fig, ax


def plot_chi6_scalar(chi6,
                     xlabel="",
                     ylabel=r"$\chi_6$",
                     title=r"Susceptibility $\chi_6$ (single dataset)"):
    """
    Display the susceptibility as a simple bar (single number).
    """
    fig, ax = plt.subplots()
    ax.bar([0], [chi6], width=0.4)
    ax.set_xticks([0])
    ax.set_xticklabels([xlabel] if xlabel else [""])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0, chi6, f"{chi6:.4f}", ha="center", va="bottom")
    return fig, ax


def plot_g6_curve(r, g6,
                  xlabel="r",
                  ylabel=r"$g_6(r)$",
                  title=r"Bond-orientational correlation $g_6(r)$"):
    """
    Quick-look g6(r) curve.
    """
    r = np.asarray(r)
    g6 = np.asarray(g6)
    fig, ax = plt.subplots()
    ax.plot(r, g6, "-o", lw=1, ms=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_density_histogram(ax, rho_map, bins=100):
    vals = rho_map.ravel()
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=bins, density=True, alpha=0.75, color='C0')
    ax.set_xlabel(r'local density $\rho$')
    ax.set_ylabel('PDF')
    ax.grid(True)


def overlay_fit_on_g6(ax, r, g6_avg, fit, y_floor=1e-3, y_cap=1.0):
    """
    Log–log plot of g6 with envelope-fit overlay.
    Always shows y-range [1e-3, 1].
    """
    r = np.asarray(r, float)
    y = np.asarray(g6_avg, float)
    y = np.clip(y, y_floor, y_cap)

    # raw
    ax.loglog(r, y, 'o-', ms=3, lw=1.0, color='C0', alpha=0.7, label=r'$g_6(r)$ avg')

    # points used for fit
    if hasattr(fit, "r_used") and hasattr(fit, "y_used") and len(fit.r_used) > 0:
        ax.loglog(fit.r_used, np.clip(fit.y_used, y_floor, y_cap),
                  'o', ms=4, color='C3', label='fit pts (upper envelope)')

    # domain
    rmin = max(fit.rmin, np.min(r[r > 0]))
    rmax = min(fit.rmax, np.max(r))
    if rmax <= rmin:
        rmin, rmax = np.min(r[r > 0]), np.max(r)
    rr = np.geomspace(max(rmin, 1e-12), rmax, 256)

    # const
    ax.loglog(rr, np.clip(0*rr + fit.c_tail, y_floor, y_cap),
              '-', color='0.5', lw=1.2, alpha=0.8, label='const')

    # power
    if np.isfinite(getattr(fit, "eta6_mean", np.nan)):
        if hasattr(fit, "r_used") and len(fit.r_used) > 0:
            A = float(fit.y_used[0] * (fit.r_used[0] ** fit.eta6_mean))
        else:
            A = float(y[-1] * (r[-1] ** fit.eta6_mean))
        ax.loglog(rr, np.clip(A * rr**(-fit.eta6_mean), y_floor, y_cap),
                  '-', color='C2', lw=1.6, label=fr'power (η₆≈{fit.eta6_mean:.3f})')

    # exp (re-fit for display)
    if hasattr(fit, "r_used") and len(fit.r_used) >= 2:
        lw = np.log(np.clip(fit.y_used, y_floor, None))
        X  = np.vstack([fit.r_used, np.ones_like(fit.r_used)]).T
        se, be = np.linalg.lstsq(X, lw, rcond=None)[0]
        ax.loglog(rr, np.clip(np.exp(se*rr + be), y_floor, y_cap),
                  '-', color='C1', lw=1.6, label='exp')

    ax.set_xlabel('r'); ax.set_ylabel(r'$g_6(r)$')
    ax.set_ylim(y_floor, y_cap)          # <---- fixed y-range [1e-3, 1]
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=8)



def plot_structure_factor(q: np.ndarray, Sq: np.ndarray, *, ax=None, annotate_peak: bool = True, peak_min_prom=0.1):
    """
    Plot S(q) with optional first-peak annotation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    ax.plot(q, Sq, lw=1.6, color="C0")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$S(q)$")
    ax.grid(True, alpha=0.3)
    if annotate_peak and len(q) > 10:
        try:
            from scipy.signal import find_peaks
            pk, props = find_peaks(Sq, prominence=peak_min_prom)
            if pk.size > 0:
                j = pk[np.argmax(Sq[pk])]
                ax.axvline(q[j], ls="--", lw=1.0, color="C1", alpha=0.7)
                ax.annotate(fr"$q^\ast={q[j]:.3f}$", (q[j], Sq[j]),
                            xytext=(5, 10), textcoords="offset points",
                            color="C1", ha="left", va="bottom", fontsize=9)
        except Exception:
            pass
    return ax



def plot_structure_factor_2d(kx: np.ndarray, ky: np.ndarray, S: np.ndarray,
                             *, ax=None, log10: bool = True,
                             vmin=None, vmax=None,  # can be numbers or percentiles (e.g., "p1","p99")
                             cmap="magma", colorbar=True, title=None,
                             xlim=None, ylim=None):
    """
    Heatmap of S(kx,ky). Supports zoom via xlim/ylim (in k-units) and
    percentile-based contrast via vmin/vmax strings: "p1","p5","p95","p99".
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    A = np.array(S, dtype=float)
    if log10:
        with np.errstate(invalid="ignore"):
            A = np.log10(A)

    # Percentile contrast (if vmin/vmax passed like "p1","p99")
    def _parse_p(v, arr):
        if isinstance(v, str) and v.startswith("p"):
            p = float(v[1:])
            return np.nanpercentile(arr, p)
        return v
    vmin = _parse_p(vmin, A)
    vmax = _parse_p(vmax, A)

    # imshow expects extents along displayed axes
    extent = [ky.min(), ky.max(), kx.min(), kx.max()]  # x=ky, y=kx
    im = ax.imshow(A, origin="lower", extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x$")
    if title:
        ax.set_title(title)
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\log_{10} S(\mathbf{k})$" if log10 else r"$S(\mathbf{k})$")

    # crosshairs at k=0
    ax.axhline(0, color="w", alpha=0.4, lw=0.6)
    ax.axvline(0, color="w", alpha=0.4, lw=0.6)

    # Apply zoom (note: displayed x-axis is ky, y-axis is kx)
    if xlim is not None:
        ax.set_xlim(xlim)  # limits in ky
    if ylim is not None:
        ax.set_ylim(ylim)  # limits in kx

    return ax

# ===============================
# Translational: c(r) visualization
# ===============================
import numpy as np
import matplotlib.pyplot as plt

def plot_cG_curve(
    r: np.ndarray,
    c_avg: np.ndarray,
    fit,                        # FitCGResult from analysis.fit_cG_models / fit_cG_auto
    *,
    y_std: np.ndarray | None = None,
    y_floor: float = 1e-12,
    y_cap: float = 1.0,
    title: str | None = None,
):
    """
    Log–log plot of |c(r)| with optional ±1σ band and model overlays (const/power/exp).
    Returns matplotlib Axes.
    """
    r = np.asarray(r, float)
    y = np.clip(np.asarray(c_avg, float), y_floor, y_cap)

    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    # 1σ band if provided
    if y_std is not None:
        ylo = np.clip(y - y_std, y_floor, y_cap)
        yhi = np.clip(y + y_std, y_floor, y_cap)
        ax.fill_between(r, ylo, yhi, color='C0', alpha=0.18, lw=0)

    # averaged curve
    ax.loglog(r, y, 'o-', ms=3, lw=1.1, color='C0', alpha=0.85, label=r'$\langle |c(r)| \rangle$')

    # points actually used by the fit
    if getattr(fit, "r_used", None) is not None and len(fit.r_used) > 0:
        ax.loglog(fit.r_used, np.clip(fit.y_used, y_floor, y_cap),
                  'o', ms=4, color='C3', alpha=0.9, label='fit pts')

    # overlays (on r∈[rmin, rmax])
    rr_all = r[(r > 0)]
    if rr_all.size == 0:
        rr_all = np.geomspace(1e-6, 1.0, 64)
    rmin = float(getattr(fit, "rmin", np.nan))
    rmax = float(getattr(fit, "rmax", np.nan))
    if np.isfinite(rmin) and np.isfinite(rmax) and rmax > rmin:
        rr = np.geomspace(max(rmin, 1e-12), rmax, 256)
    else:
        rr = np.geomspace(max(rr_all.min(), 1e-12), rr_all.max(), 256)

    # const overlay
    if np.isfinite(getattr(fit, "c_tail", np.nan)):
        ax.loglog(rr, np.clip(0*rr + float(fit.c_tail), y_floor, y_cap),
                  '-', color='0.5', lw=1.1, alpha=0.9, label='const')

    # power overlay
    if np.isfinite(getattr(fit, "etaG_mean", np.nan)) and np.isfinite(getattr(fit, "logA_power", np.nan)):
        A = float(np.exp(fit.logA_power))
        ax.loglog(rr, np.clip(A * rr**(-fit.etaG_mean), y_floor, y_cap),
                  '-', color='C2', lw=1.6, label=fr'power (η_G≈{fit.etaG_mean:.3f})')

    # exponential overlay
    if np.isfinite(getattr(fit, "alpha_exp", np.nan)) and np.isfinite(getattr(fit, "logA_exp", np.nan)):
        ax.loglog(rr, np.clip(np.exp(fit.alpha_exp * rr + fit.logA_exp), y_floor, y_cap),
                  '-', color='C1', lw=1.6, label='exp')

    ax.set_xlabel('r'); ax.set_ylabel(r'$|c(r)|$')
    ax.set_ylim(y_floor, y_cap)
    if title:
        ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=8, frameon=False)
    return ax
