from loguru import logger
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    # "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "STIXGeneral"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False
})


def align_symmetry_branch(gammas: np.ndarray,
                          betas: np.ndarray,
                          period_gamma: float = np.pi,
                          period_beta: float = np.pi
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Adjusts QAOA parameters to remain in a continuous principal symmetry sector.

    Gamma is restricted to [0, pi] during optimisation (integer eigenvalues of H_C
    + even symmetry E(gamma,beta)=E(-gamma,-beta)), so period_gamma=pi.
    Beta is naturally periodic with period pi.
    """
    g = gammas % period_gamma
    b = betas % period_beta

    if np.mean(g) > (period_gamma / 2):
        g = period_gamma - g
        b = period_beta - b

    g = np.unwrap(g)
    b = np.unwrap(b * 2) / 2

    return g, b


def plot_parameter_evolution() -> None:
    """
    Loads QAOA parameters from CSV and plots their depth-wise evolution
    specifically for p=3, p=4, and p=5.
    """
    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "data" / "processed" / "params.csv"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        logger.error(f"Parameter CSV file not found: {csv_file}")
        logger.warning("Please run params_by_depths.py first to generate the data.")
        return

    logger.info(f"Loading parameters from {csv_file}")

    df = pd.read_csv(csv_file)
    results: Dict[int, np.ndarray] = {}

    for p_val, group in df.groupby("Depth (p)"):
        group = group.sort_values("Layer (i)")
        raw_g = group["Gamma (raw)"].values
        raw_b = group["Beta (raw)"].values
        results[p_val] = np.column_stack((raw_g, raw_b))

    target_depths = [3, 4, 5]
    depths = [p for p in target_depths if p in results]

    if not depths:
        logger.error(f"None of the target depths {target_depths} were found in the CSV.")
        return

    n_cols = len(depths)
    fig, axes = plt.subplots(nrows=2,
                             ncols=n_cols,
                             figsize=(max(10, 3 * n_cols), 6),
                             sharex=False, sharey="row"
                             )

    color_gamma = '#1f77b4'
    color_beta = '#ff7f0e'
    param_names = [r"$\frac{\gamma}{\pi}$", r"$\frac{\beta}{\pi}$"]

    logger.info(f"Aligning symmetries and rendering plots for depths: {depths}.")

    for col_idx, p in enumerate(depths):
        params = results[p]
        raw_gammas = params[:, 0]
        raw_betas = params[:, 1]

        g_aligned, b_aligned = align_symmetry_branch(raw_gammas, raw_betas)

        g_norm = g_aligned / np.pi
        b_norm = b_aligned / np.pi

        layers = range(1, p + 1)

        ax_gamma = axes[0, col_idx] if n_cols > 1 else axes[0]
        ax_beta = axes[1, col_idx] if n_cols > 1 else axes[1]

        face_color_gamma = mcolors.to_rgba(color_gamma, alpha=0.4)
        face_color_beta = mcolors.to_rgba(color_beta, alpha=0.4)

        ax_gamma.plot(layers, g_norm,
                      marker="o",
                      linestyle="-",
                      color=color_gamma,
                      markersize=6,
                      markerfacecolor=face_color_gamma,
                      markeredgecolor=color_gamma,
                      markeredgewidth=1.5
                      )
        ax_gamma.set_title(f"Depth $p={p}$")
        ax_gamma.set_xticks(layers)
        ax_gamma.grid(True, linestyle="--", alpha=0.5)

        ax_beta.plot(layers, b_norm,
                     marker="o",
                     linestyle="-",
                     color=color_beta,
                     markersize=6,
                     markerfacecolor=face_color_beta,
                     markeredgecolor=color_beta,
                     markeredgewidth=1.5
                     )
        ax_beta.set_xticks(layers)
        ax_beta.grid(True, linestyle="--", alpha=0.5)

        if col_idx == 0:
            ax_gamma.set_ylabel(param_names[0], fontsize=14, rotation=0, labelpad=15)
            ax_beta.set_ylabel(param_names[1], fontsize=14, rotation=0, labelpad=15)

        ax_beta.set_xlabel("Layer $i$", fontsize=10)

    plt.tight_layout()

    save_path = reports_dir / "parameter_evolution.pdf"

    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    logger.success(f"Plot successfully saved to: {save_path}")


if __name__ == "__main__":
    plot_parameter_evolution()
