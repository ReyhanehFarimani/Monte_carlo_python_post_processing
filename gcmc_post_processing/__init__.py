"""
Reyhaneh — package init
Updated: 10 Oct 2025
"""

# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------
from .data_loader import (
    get_data_files,
    load_xyz,
    load_xyz_crpt_check,
    load_txt_data,
    load_txt_data_old,
    load_dump_file,
)

# -----------------------------------------------------------------------------
# analysis (core + orientational)
# -----------------------------------------------------------------------------
from .analysis import (
    # core helpers
    compute_local_density,
    compute_rdf,
    average_rdf_over_trajectory,
    voronoi_neighborhood_list,

    # orientational order
    compute_psi,                    # back-compat: <|psi_k|> over frames
    compute_psi6_series,            # per-frame complex Psi_6 + <|Psi_6|>±stderr
    binder_and_susceptibility,      # U4^(6), chi_6

    # g6(r)
    compute_g6,                     # back-compat wrapper -> compute_g6_avg
    compute_g6_avg,                 # averaged g6(r), optional per-frame
    fit_eta6_from_g6,               # simple eta_6 fit (log–log)

    # model selection + classifiers
    fit_g6_auto,                    # simple power vs exp (tail window)
    fit_g6_models,                  # const vs power vs exp (AIC + bootstrap)
    classify_orientational_phase,   # simple classifier
    classify_orientational_phase_rules,  # robust rule-based classifier

    # scans
    scan_by_density_or_mu,
    scan_and_classify_orientational,
    scan_and_classify_orientational_robust,

    # maps
    compute_psi_density,

    # translational (stubs for now)
    compute_gG,
    sub_system_translational,

    # --- structure-factor pipelines
    compute_and_save_structure_factor,         # 1D radial S(q)
    compute_and_save_structure_factor_2d,      # 2D S(kx,ky)
)

# -----------------------------------------------------------------------------
# visualization
# -----------------------------------------------------------------------------
from .visualization import (
    plot_data,
    plot_rdf,
    plot_voronoi_with_ids,
    user_defined_voronoi_plot,
    plot_voro,

    # orientational plots
    plot_psi6_vs_density,
    plot_binder_vs_density,
    plot_chi6_vs_density,
    plot_psi6_time_series,
    plot_psi6_abs_histogram,
    plot_binder_scalar,
    plot_chi6_scalar,
    plot_g6_curve,

    # --- structure-factor plots
    plot_structure_factor,          # 1D S(q)
    plot_structure_factor_2d,       # 2D S(kx,ky)
)

# -----------------------------------------------------------------------------
# utilities (generators + io helpers)
# -----------------------------------------------------------------------------
from .utils import (
    generate_fake_crystal_structure,
    generate_fake_triangular_lattice,
    generate_fake_triangular_lattice_defected,
    generate_random_particles,

    # --- io helper used by structure-factor pipelines
    save_table_csv,
)

# -----------------------------------------------------------------------------
# statics / helpers (incl. structure factor core)
# -----------------------------------------------------------------------------
from .statics import (
    extract_mu_and_temperature,
    read_simulation_input_Old,
    read_simulation_input,
    process_simulation_data,
    bin_data_by_density,

    # --- structure-factor core API (1D + 2D)
    StructureFactorResult,          # 1D
    structure_factor,               # 1D
    structure_factor_from_gr,       # 1D (from g(r))
    StructureFactor2DResult,        # 2D
    structure_factor_2d,            # 2D
    extract_q_vectors_from_S2D,
    extract_first_shell_q_vectors,
    extract_q_vectors_from_S2D,
)

# -----------------------------------------------------------------------------
# back-compat aliases / shims
# -----------------------------------------------------------------------------
# Keep the historical misspelling importable to avoid breakage.
# Users may have: from pkg import classify_orientional_phase_rules
classify_orientional_phase_rules = classify_orientational_phase_rules  # noqa: E305

# -----------------------------------------------------------------------------
# public api
# -----------------------------------------------------------------------------
__all__ = [
    # data_loader
    "get_data_files",
    "load_xyz",
    "load_xyz_crpt_check",
    "load_txt_data",
    "load_txt_data_old",
    "load_dump_file",

    # analysis (core + orientational)
    "compute_local_density",
    "compute_rdf",
    "average_rdf_over_trajectory",
    "voronoi_neighborhood_list",
    "compute_psi",
    "compute_psi6_series",
    "binder_and_susceptibility",
    "compute_g6",
    "compute_g6_avg",
    "fit_eta6_from_g6",
    "fit_g6_auto",
    "fit_g6_models",
    "classify_orientational_phase",
    "classify_orientional_phase_rules",   # back-compat alias
    "classify_orientational_phase_rules",
    "scan_by_density_or_mu",
    "scan_and_classify_orientational",
    "scan_and_classify_orientational_robust",
    "compute_psi_density",
    "compute_gG",
    "sub_system_translational",

    # visualization
    "plot_data",
    "plot_rdf",
    "plot_voronoi_with_ids",
    "user_defined_voronoi_plot",
    "plot_voro",
    "plot_psi6_vs_density",
    "plot_binder_vs_density",
    "plot_chi6_vs_density",
    "plot_psi6_time_series",
    "plot_psi6_abs_histogram",
    "plot_binder_scalar",
    "plot_chi6_scalar",
    "plot_g6_curve",
    "plot_structure_factor",        # 1D
    "plot_structure_factor_2d",     # 2D

    # utils
    "generate_fake_crystal_structure",
    "generate_fake_triangular_lattice",
    "generate_fake_triangular_lattice_defected",
    "generate_random_particles",
    "save_table_csv",

    # statics
    "extract_mu_and_temperature",
    "read_simulation_input_Old",
    "read_simulation_input",
    "process_simulation_data",
    "bin_data_by_density",

    # structure factor core + pipelines
    "StructureFactorResult",
    "structure_factor",
    "structure_factor_from_gr",
    "StructureFactor2DResult",
    "structure_factor_2d",
    "compute_and_save_structure_factor",
    "compute_and_save_structure_factor_2d",
    "extract_q_vectors_from_S2D",
    "extract_first_shell_q_vectors",
    "extract_q_vectors_from_S2D",
]
