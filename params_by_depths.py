from loguru import logger
from pathlib import Path
from qmos_qaoa.data import load_local_data
from qmos_qaoa.model import TimetableProblem
from qmos_qaoa.solver import solve_qaoa
import numpy as np
import pandas as pd


def main():
    # Get all instance directories sorted (inst_01, inst_02, etc.)
    base_dir = Path(__file__).resolve().parent
    ensembles_dir = base_dir / "data" / "raw" / "ensembles"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not ensembles_dir.exists():
        logger.error(f"Ensembles directory not found: {ensembles_dir}")
        logger.warning("Please run generate_ensembles.py first.")
        return

    logger.info("Starting Exhaustive Ensemble QAOA Parameter Search.")

    depths_to_run = list(range(1, 6))
    shots = 20000
    restarts = 20
    n_jobs = -1

    csv_export_data = []

    instance_dirs = sorted([d for d in ensembles_dir.iterdir() if d.is_dir()])

    for inst_dir in instance_dirs[:10]:
        inst_name = inst_dir.name
        logger.info(f"Processing {inst_name}")

        try:
            data = load_local_data(inst_dir)
            problem = TimetableProblem(data)
        except Exception as e:
            logger.error(f"Failed to load or build problem for {inst_name}: {e}")
            continue

        for p in depths_to_run:
            logger.info(f"Optimizing {inst_name} at depth p={p}")

            _, optimal_params = solve_qaoa(problem=problem,
                                           depth=p,
                                           shots=shots,
                                           restarts=restarts,
                                           n_jobs=n_jobs
                                           )

            for layer_idx, (g, b) in enumerate(optimal_params):
                csv_export_data.append(
                    {
                        "Instance": inst_name,
                        "Depth (p)": p,
                        "Layer (i)": layer_idx + 1,
                        "Gamma (raw)": g,
                        "Beta (raw)": b,
                        "Gamma/Pi": g / np.pi,
                        "Beta/Pi": b / np.pi
                    }
                )

    csv_file = processed_dir / "vikstal_params_ensemble.csv"
    df_export = pd.DataFrame(csv_export_data)
    df_export.to_csv(csv_file, index=False)

    logger.success("All ensemble runs complete!")
    logger.info(f"Master CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
