from collections import Counter
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path


def top_k_lists(samples, k=10) -> List[Tuple[Tuple, int, float]]:
    """
    Finds the top k most frequent bitstrings in the samples.
    """
    list_of_lists = samples.tolist()
    total = len(list_of_lists)
    counts = Counter(map(tuple, list_of_lists))
    top_k = counts.most_common(k)
    return [(lst, cnt, cnt / total) for lst, cnt in top_k]


def plot_top_k(top_k_stats, save_path: Path = None):
    """
    Plots the top k results and optionally saves them to a path.
    """
    labels = [list(tup) for tup, _, _ in top_k_stats]
    counts = [cnt for _, cnt, _ in top_k_stats]
    rates = [rate for _, _, rate in top_k_stats]

    plt.figure(figsize=(10, max(6, len(top_k_stats) / 3)))
    bars = plt.barh(range(len(labels)), rates)

    str_labels = [str(l) for l in labels]
    plt.yticks(range(len(labels)), str_labels)

    plt.xlabel("Appearance rate")
    plt.title(f"Top {len(top_k_stats)} lists by appearance rate")

    for i, (b, cnt, rate) in enumerate(zip(bars, counts, rates)):
        plt.text(b.get_width(), b.get_y() + b.get_height() / 2,
                 f"{cnt} ({rate:.2%})",
                 ha="left", va="center")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        # Ensure it's a Path object
        save_path = Path(save_path)
        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
