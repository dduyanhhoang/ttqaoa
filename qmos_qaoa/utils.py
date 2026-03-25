import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
from loguru import logger

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


def top_k_lists(samples: np.ndarray, k: int = 10) -> List[Tuple]:
    """
    Extracts the top-k most frequently measured bitstrings from QAOA samples.

    Args:
        samples (np.ndarray): A 2D array of shape (shots, n_qubits) containing 
                              the sampled bitstrings from the quantum device.
        k (int): The number of top results to return. Default is 10.

    Returns:
        List[Tuple[Tuple[int, ...], int, float]]: A list containing the top-k results.
            Each entry is a tuple of: (bitstring_tuple, count, appearance_rate).
    """
    logger.debug(f"Extracting top {k} bitstrings from {len(samples)} samples...")
    total_shots = len(samples)

    unique_rows, counts = np.unique(samples, axis=0, return_counts=True)
    sort_indices = np.argsort(-counts)

    top_k = []
    for idx in sort_indices[:k]:
        bitstring = tuple(unique_rows[idx].tolist())
        count = int(counts[idx])
        rate = count / total_shots
        top_k.append((bitstring, count, rate))

    if top_k:
        logger.info(f"Top bitstring appeared {top_k[0][1]} times ({top_k[0][2]:.2%}).")

    return top_k


def plot_top_k(top_k_stats: List[Tuple],
               save_path: Path | str | None = None
               ) -> None:
    """
    Visualizes the most frequent bitstrings as a horizontal bar chart.

    Args:
        top_k_stats (List[Tuple]): The output generated from `top_k_lists`.
        save_path (Path | str | None): The file path to save the generated plot. 
                                       If None, the plot is displayed interactively.
    """
    if not top_k_stats:
        logger.warning("No statistics provided to plot.")
        return

    logger.info("Generating top-k results plot.")

    labels = [str(list(tup)) for tup, _, _ in top_k_stats]
    counts = [cnt for _, cnt, _ in top_k_stats]
    rates = [rate for _, _, rate in top_k_stats]

    fig_height = max(6.0, len(top_k_stats) / 3.0)
    plt.figure(figsize=(12, fig_height))

    bars = plt.barh(range(len(labels)), rates, color='skyblue')
    max_rate = max(rates) if rates else 1.0
    plt.xlim(0, max_rate * 1.2)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Appearance Rate")
    plt.title(f"Top {len(top_k_stats)} States by Appearance Rate")

    for bar, count, rate in zip(bars, counts, rates):
        plt.text(bar.get_width(),
                 bar.get_y() + bar.get_height() / 2,
                 f" {count} ({rate:.2%})",
                 ha="left",
                 va="center",
                 fontsize=10
                 )

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        logger.success(f"Plot successfully saved to: {save_path}")
    else:
        plt.show()

    plt.close()
