# QAOA University Timetabling

Applies the Quantum Approximate Optimization Algorithm (QAOA) to the university
course-scheduling problem, following the parameter evolution methodology of
Vikstål et al. (2020). The timetabling instance is encoded as a QUBO cost
Hamiltonian whose ground state encodes a valid schedule.

---

## Project Structure

```
ttqaoa/
├── data/
│   ├── raw/
│   │   ├── 16qubits/          # 16-qubit base instance (used by main.py)
│   │   ├── 20qubits/          # 20-qubit instance (used by single-instance study)
│   │   └── ensembles/         # 20 generated diverse 16-qubit instances
│   └── processed/
│       ├── params.csv                    # Output of params_by_depths_single.py
│       └── vikstal_params_ensemble.csv   # Output of params_by_depths.py
├── reports/                   # Generated PDF plots
├── output/                    # JSON results from algorithm_runner.py
├── qmos_qaoa/                 # Core library
│   ├── data.py                # CSV loading and preprocessing
│   ├── model.py               # QUBO Hamiltonian construction
│   ├── solver.py              # Multi-start QAOA optimizer (lightning.qubit)
│   └── utils.py               # Top-k bitstring analysis and bar-chart helpers
├── generate_ensembles.py      # Generate 20 diverse 16-qubit instances
├── params_by_depths.py        # Ensemble parameter evolution (10 instances, p=1..5)
├── params_by_depths_plot.py   # Plot ensemble parameter evolution
├── params_by_depths_single.py # Single-instance parameter evolution (20-qubit)
├── params_by_depths_plot_single.py  # Plot single-instance parameter evolution
└── main.py                    # Single QAOA run entry point (16-qubit)
```

---

## Setup

**With uv (recommended):**

```bash
uv sync
```

**With pip:**

```bash
pip install -r requirements.txt
```

> `requirements.txt` includes `-e .` as its first line, so pip installs both
> the external dependencies and the local `qmos_qaoa` package in one step.

---

## Parameter Evolution Study

The parameter evolution study reproduces Vikstål (2020): it runs QAOA for
circuit depths p = 1 … 5, each with multiple random restarts, and records the
optimal (γ, β) angles.  The plots show whether the angles concentrate into a
smooth, monotonically increasing trajectory as p grows.

### Single-instance study (20-qubit dataset)

**Step 1 — Optimize**

```bash
python params_by_depths_single.py
```

Runs multi-start QAOA (p = 1 … 5) on `data/raw/20qubits/`.
Result: `data/processed/params.csv`

**Step 2 — Plot**

```bash
python params_by_depths_plot_single.py
```

Result: `reports/parameter_evolution.pdf`

---

### Ensemble study (20 generated 16-qubit instances)

**Step 1 — Generate instances** *(run once)*

```bash
python generate_ensembles.py
```

Creates `data/raw/ensembles/inst_01/` … `inst_20/`.
Each instance has 8 tasks × 2 eligible teachers = 16 qubits.
A diversity report is printed to stdout.

**Step 2 — Optimize**

```bash
python params_by_depths.py
```

Runs multi-start QAOA across the first 10 instances for p = 1 … 5 in parallel.
Result: `data/processed/vikstal_params_ensemble.csv`

**Step 3 — Plot**

```bash
python params_by_depths_plot.py
```

Result: `reports/<timestamp>_ensemble_parameter_evolution.pdf`

---

## HPC Submission

Set `OMP_NUM_THREADS=1` so each joblib worker stays single-threaded and does
not compete with the other parallel workers over cores:

```bash
export OMP_NUM_THREADS=1
python params_by_depths.py
```

Recommended SLURM header for a 256-core node:

```bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=64G
```

### Runtime estimates (256-core AMD EPYC 9754, lightning.qubit)

| Restarts | Wall time (p = 1…5) |
|----------|----------------------|
| 20       | ~17 min              |
| 100      | ~17 min              |
| 1 000    | ~1.1 h               |
| 4 000    | ~4.5 h               |

---

## Configuration

Key parameters are set at the top of each runner script:

| Parameter      | File                          | Default | Notes                        |
|----------------|-------------------------------|---------|------------------------------|
| `restarts`     | `params_by_depths*.py`        | 20      | Increase to 1000–4000 for HPC |
| `n_jobs`       | `params_by_depths*.py`        | -1      | -1 = all available cores     |
| `depths_to_run`| `params_by_depths*.py`        | [1..5]  | QAOA circuit depths to sweep |
| `shots`        | `params_by_depths*.py`        | 20000   | Finite shots for final sample |

---

## QUBO Hamiltonian

Three penalty terms are summed into a single cost Hamiltonian H_C.
Qubit projectors n_i = (I − Z_i)/2 map the Pauli-Z basis to binary {0, 1}.

| Constraint              | Weight | Penalty form                              |
|-------------------------|--------|-------------------------------------------|
| One teacher per section | W = 3  | `(Σ_t x_{t,s} − 1)²` per section s       |
| No overlapping slots    | W = 2  | `(x_{t,s1} + x_{t,s2})²` per conflict pair |
| Teacher quota (min/max) | W = 1  | `(Σ_s x_{t,s} − q_min)² + (Σ_s x_{t,s} − q_max)²` per teacher |

---

## Reference

Vikstål, P., Grönkvist, M., Svensson, M., Andersson, M., Johansson, G., &
Ferrini, G. (2020). Applying the quantum approximate optimization algorithm to
the tail-assignment problem. *Physical Review Applied, 14*(3), 034009.
