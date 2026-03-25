from pathlib import Path
from qmos_qaoa.data import load_local_data
from qmos_qaoa.model import TimetableProblem
from qmos_qaoa.solver import solve_qaoa
from qmos_qaoa.utils import top_k_lists, plot_top_k
import datetime


def main():
    base_dir = Path(__file__).resolve().parent

    data_dir = base_dir / "data" / "raw" / "16qbit_quota"
    reports_dir = base_dir / "reports" / "quota"

    try:
        data = load_local_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    print("Building Hamiltonian...")
    problem = TimetableProblem(data)
    print(f"Total variables/qubits: {problem.n_qubits}")

    p_depth = 3
    shots = 20000

    samples = solve_qaoa(problem, depth=p_depth, shots=shots)

    print("Analyzing results...")
    cnt = top_k_lists(samples, k=20)

    now = datetime.datetime.now()
    now = str(now.strftime("%Y-%b-%d_%H-%M-%S"))
    filename = f"({now})_p{p_depth}_s{shots}.png"
    save_path = reports_dir / filename

    plot_top_k(cnt, save_path=save_path)


if __name__ == "__main__":
    main()
