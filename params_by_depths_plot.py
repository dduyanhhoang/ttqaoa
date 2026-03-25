import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from loguru import logger

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False
    }
)


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


def plot_ensemble_parameter_evolution() -> None:
    """
    1. Process and plot each instance individually (background lines)
    2. Calculate Ensemble Statistics
    3. Plot the Mean Trajectories
    """
    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "data" / "processed" / "vikstal_params_ensemble.csv"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        logger.error(f"Ensemble CSV file not found: {csv_file}")
        return

    logger.info(f"Loading ensemble parameters from {csv_file}")
    df = pd.read_csv(csv_file)

    target_depths = [3, 4, 5]
    plot_depths = sorted(df[df["Depth (p)"].isin(target_depths)]["Depth (p)"].unique())

    if not plot_depths:
        logger.error("No data found for target depths p=3, 4, or 5.")
        return

    n_cols = len(plot_depths)
    fig, axes = plt.subplots(nrows=2,
                             ncols=n_cols,
                             figsize=(max(10, 3 * n_cols), 6),
                             sharex=False,
                             sharey="row"
                             )

    color_gamma = '#1f77b4'
    color_beta = '#ff7f0e'
    face_color_gamma = mcolors.to_rgba(color_gamma, alpha=0.4)
    face_color_beta = mcolors.to_rgba(color_beta, alpha=0.4)

    param_names = [r"$\frac{\gamma}{\pi}$", r"$\frac{\beta}{\pi}$"]

    for col_idx, p in enumerate(plot_depths):
        df_p = df[df["Depth (p)"] == p]

        all_g_norm = []
        all_b_norm = []
        layers = range(1, p + 1)

        ax_gamma = axes[0, col_idx] if n_cols > 1 else axes[0]
        ax_beta = axes[1, col_idx] if n_cols > 1 else axes[1]

        for inst_name, group in df_p.groupby("Instance"):
            group = group.sort_values("Layer (i)")
            raw_g = group["Gamma (raw)"].values
            raw_b = group["Beta (raw)"].values

            g_aligned, b_aligned = align_symmetry_branch(raw_g, raw_b)
            g_norm = g_aligned / np.pi
            b_norm = b_aligned / np.pi

            all_g_norm.append(g_norm)
            all_b_norm.append(b_norm)

            ax_gamma.plot(layers, g_norm, color=color_gamma, alpha=0.15, linewidth=1)
            ax_beta.plot(layers, b_norm, color=color_beta, alpha=0.15, linewidth=1)

        g_mean = np.mean(all_g_norm, axis=0)
        g_std = np.std(all_g_norm, axis=0)
        b_mean = np.mean(all_b_norm, axis=0)
        b_std = np.std(all_b_norm, axis=0)

        ax_gamma.plot(
            layers, g_mean, marker="o", linestyle="-", color=color_gamma,
            markersize=6, markerfacecolor=face_color_gamma,
            markeredgecolor=color_gamma, markeredgewidth=1.5, linewidth=2, label="Mean"
        )
        ax_gamma.fill_between(layers, g_mean - g_std, g_mean + g_std, color=color_gamma, alpha=0.1)

        ax_beta.plot(layers,
                     b_mean,
                     marker="s",
                     linestyle="-",
                     color=color_beta,
                     markersize=6,
                     markerfacecolor=face_color_beta,
                     markeredgecolor=color_beta,
                     markeredgewidth=1.5,
                     linewidth=2,
                     label="Mean"
                     )
        ax_beta.fill_between(layers, b_mean - b_std, b_mean + b_std, color=color_beta, alpha=0.1)

        ax_gamma.set_title(f"Depth $p={p}$")
        ax_gamma.set_xticks(layers)
        ax_gamma.grid(True, linestyle="--", alpha=0.5)

        ax_beta.set_xticks(layers)
        ax_beta.grid(True, linestyle="--", alpha=0.5)
        ax_beta.set_xlabel("Layer $i$", fontsize=10)

        if col_idx == 0:
            ax_gamma.set_ylabel(param_names[0], fontsize=14, rotation=0, labelpad=15)
            ax_beta.set_ylabel(param_names[1], fontsize=14, rotation=0, labelpad=15)
            ax_gamma.legend(loc="upper left")
            ax_beta.legend(loc="upper left")

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
    save_path = reports_dir / f"({timestamp})_ensemble_parameter_evolution.pdf"

    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    logger.success(f"Ensemble plot successfully saved to: {save_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_ensemble_parameter_evolution()
