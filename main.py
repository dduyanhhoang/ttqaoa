from pathlib import Path
from qmos_qaoa.data import load_local_data
from qmos_qaoa.model import TimetableProblem
from qmos_qaoa.solver import solve_qaoa
from qmos_qaoa.utils import top_k_lists, plot_top_k
import datetime


def main() -> None:
    """
    Run the full QAOA solving pipeline.
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "raw" / "16qubits"
    reports_dir = base_dir / "reports"

    data_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    try:
        data = load_local_data(data_dir)
    except FileNotFoundError as e:
        raise e

    problem = TimetableProblem(data)

    p_depth = 1
    shots = 20000
    samples, opt_params = solve_qaoa(problem, depth=p_depth, shots=shots)

    cnt = top_k_lists(samples, k=20)
    now = datetime.datetime.now().strftime("%Y-%b-%d_%H-%M")
    filename = f"({now})_p{p_depth}_s{shots}.pdf"
    save_path = reports_dir / filename
    plot_top_k(cnt, save_path=save_path)


if __name__ == "__main__":
    main()
