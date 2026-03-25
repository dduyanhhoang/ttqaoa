from loguru import logger
from pathlib import Path
from qmos_qaoa.data import load_local_data
from qmos_qaoa.model import TimetableProblem
from qmos_qaoa.solver import solve_qaoa
from typing import Dict
import numpy as np
import pandas as pd


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "raw" / "20qubits"
    reports_dir = base_dir / "reports"
    processed_dir = base_dir / "data" / "processed"

    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting exhaustive search.")

    try:
        data = load_local_data(data_dir)
    except Exception as e:
        logger.exception("Failed to load local data.")
        raise e

    problem = TimetableProblem(data)

    depths_to_run = list(range(1, 6))
    shots = 20000
    restarts = 100
    n_jobs = -1

    results_storage: Dict[int, np.ndarray] = {}
    csv_export_data = []

    for p in depths_to_run:
        logger.info(f"Initiating exhaustive search for p={p}.")

        samples, optimal_params = solve_qaoa(problem=problem,
                                             depth=p,
                                             shots=shots,
                                             restarts=restarts,
                                             n_jobs=n_jobs
                                             )

        results_storage[p] = optimal_params

        for layer_idx, (g, b) in enumerate(optimal_params):
            csv_export_data.append({
                "Depth (p)": p,
                "Layer (i)": layer_idx + 1,
                "Gamma (raw)": g,
                "Beta (raw)": b,
                "Gamma/Pi": g / np.pi,
                "Beta/Pi": b / np.pi
            })

    csv_file = processed_dir / "params.csv"
    df_export = pd.DataFrame(csv_export_data)
    df_export.to_csv(csv_file, index=False)

    logger.success(f"All runs complete.")
    logger.info(f"Human-readable CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
