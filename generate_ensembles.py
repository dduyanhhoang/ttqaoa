from collections import defaultdict
from loguru import logger
from pathlib import Path
import numpy as np
import pandas as pd
import shutil


def _diversity_report(profiler_stats: list[dict]) -> None:
    """
    Compute and print a pairwise distance analysis to assess structural diversity
    across the generated ensemble.

    Uses normalized Euclidean distance over three structural features:
      - Max Tasks/Teacher  (peak teacher load)
      - Conflict Edges     (number of active slot-conflict pairs in the Hamiltonian)
      - Load Std           (standard deviation of teacher load distribution)

    Near-duplicate instances (distance < threshold) are flagged as a warning.
    """
    feature_keys = ["Max Tasks/Teacher", "Conflict Edges", "Load Std"]
    feature_matrix = np.array([[s[k] for k in feature_keys] for s in profiler_stats], dtype=float)

    print("\n# DIVERSITY ANALYSIS")
    W = 25
    print(f"  {'Metric':<{W}} {'Min':>7} {'Max':>7} {'Mean':>7} {'Std':>7}")
    print("  " + "-" * (W + 30))
    for col, k in enumerate(feature_keys):
        vals = feature_matrix[:, col]
        print(f"  {k:<{W}} {vals.min():>7.2f} {vals.max():>7.2f} {vals.mean():>7.2f} {vals.std():>7.2f}")

    # Normalize per-feature then compute pairwise Euclidean distances
    col_std = feature_matrix.std(axis=0)
    col_std[col_std < 1e-9] = 1.0  # guard: avoid division by zero for constant features
    normed = (feature_matrix - feature_matrix.mean(axis=0)) / col_std

    n = len(normed)
    THRESHOLD = 0.5
    dists = []
    near_dupes = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(normed[i] - normed[j]))
            dists.append(d)
            if d < THRESHOLD:
                near_dupes.append((profiler_stats[i]["Instance"], profiler_stats[j]["Instance"], d))

    dists_arr = np.array(dists)
    print(
        f"\n  Pairwise distance (normalized)  "
        f"min={dists_arr.min():.3f}  mean={dists_arr.mean():.3f}  max={dists_arr.max():.3f}"
    )

    if near_dupes:
        print(f"\n  WARNING: {len(near_dupes)} near-duplicate pairs (dist < {THRESHOLD}):")
        for a, b, d in near_dupes:
            print(f"    {a} <-> {b}  dist={d:.3f}")
    else:
        print(f"\n  No near-duplicates detected (threshold = {THRESHOLD}).")


def generate_datasets(num_instances: int = 20):
    """
    Generates num_instances diverse 16-qubit timetabling instances.

    Design choices:
      - 8 tasks, each assigned to exactly 2 of 4 teachers (2 x 8 = 16 feasible vars = 16 qubits).
      - [:4] teacher pool: small enough that with 8 tasks x 2 draws, each teacher's expected
        load is 4, keeping the probability of any teacher receiving zero assignments very low.
        This keeps n_qubits reliably at 16.
      - Slot assignments are randomised per task to vary the slot-conflict graph topology.
      - Quota min=1: every eligible teacher must be assigned at least 1 class, penalising
        solutions that leave a teacher idle.
      - Quota max in {2, 3, 4}: ensures sum(max_quotas) >= 8 = n_tasks so a feasible
        solution always exists under the quota constraints.
    """
    base_dir = Path(__file__).resolve().parent
    template_dir = base_dir / "data" / "raw" / "16qubits"
    output_base = base_dir / "data" / "raw" / "ensembles"

    if not template_dir.exists():
        logger.error(f"Template directory not found: {template_dir}")
        return

    logger.info(f"Generating {num_instances} diverse 16-qubit instances (8 Tasks x 2 Teachers)...")

    df_skill_template = pd.read_csv(template_dir / "InstructorSkill.csv", index_col=0)
    df_slot_template = pd.read_csv(template_dir / "InstructorSlot.csv", index_col=0)
    df_conflict = pd.read_csv(template_dir / "SlotConflict.csv", index_col=0)

    teachers = list(df_skill_template.index)[:4]
    slots = list(df_slot_template.columns)  # [A24, A42, A25, A52]

    profiler_stats = []

    for inst_idx in range(1, num_instances + 1):
        inst_dir = output_base / f"inst_{inst_idx:02d}"
        inst_dir.mkdir(parents=True, exist_ok=True)

        n_tasks = 8
        subjects = [f"SUBJ_{k:02d}" for k in range(n_tasks)]

        df_task = pd.DataFrame({
            "Class": [f"CLASS_{k:02d}" for k in range(n_tasks)],
            "Subject": subjects,
            "Slot": np.random.choice(slots, size=n_tasks),  # randomise to vary conflict topology
        })

        df_skill = pd.DataFrame(0, index=teachers, columns=subjects)
        df_slot = pd.DataFrame(0, index=teachers, columns=slots)

        teacher_task_assignments: dict[str, list[int]] = defaultdict(list)

        for task_idx in range(n_tasks):
            subj = df_task.loc[task_idx, "Subject"]
            slot = df_task.loc[task_idx, "Slot"]

            for teacher in np.random.choice(teachers, size=2, replace=False):
                df_skill.at[teacher, subj] = 5
                df_slot.at[teacher, slot] = 5
                teacher_task_assignments[teacher].append(task_idx)

        # --- Profiler metrics ---
        loads = [len(teacher_task_assignments.get(t, [])) for t in teachers]
        max_tasks_per_teacher = int(max(loads))
        load_std = float(np.std(loads))

        active_conflict_edges = 0
        for teacher, t_tasks in teacher_task_assignments.items():
            for i1 in range(len(t_tasks)):
                slot1 = df_task.loc[t_tasks[i1], "Slot"]
                for i2 in range(i1 + 1, len(t_tasks)):
                    slot2 = df_task.loc[t_tasks[i2], "Slot"]
                    if slot1 in df_conflict.index and slot2 in df_conflict.columns:
                        if df_conflict.at[slot1, slot2] == 1:
                            active_conflict_edges += 1

        # Quota: min=1 so idle-teacher solutions are penalised;
        #        max in {2,3,4} guarantees sum(max) >= 8 (feasibility w.r.t. section count).
        df_quota = pd.DataFrame(index=teachers)
        df_quota["Min quota"] = 1
        df_quota["Max quota"] = np.random.randint(2, 5, size=len(teachers))

        profiler_stats.append({
            "Instance": f"inst_{inst_idx:02d}",
            "Max Tasks/Teacher": max_tasks_per_teacher,
            "Conflict Edges": active_conflict_edges,
            "Load Std": load_std,
        })

        df_task.to_csv(inst_dir / "Task.csv", index=False)
        df_skill.to_csv(inst_dir / "InstructorSkill.csv")
        df_slot.to_csv(inst_dir / "InstructorSlot.csv")
        df_quota.to_csv(inst_dir / "InstructorQuota.csv")
        shutil.copy(template_dir / "SlotConflict.csv", inst_dir / "SlotConflict.csv")

    logger.success(f"Successfully generated {num_instances} datasets in: {output_base}")

    print("\n# DIVERSITY PROFILER REPORT")
    W1, W2, W3, W4 = 18, 25, 25, 12
    print(f"|{'Instance':<{W1}}|{'Max tasks / teacher':<{W2}}|{'Active conflict edges':<{W3}}|{'Load Std':<{W4}}|")
    print(f"|{'-' * W1}|{'-' * W2}|{'-' * W3}|{'-' * W4}|")
    for stat in profiler_stats:
        print(f"|{stat['Instance']:<{W1}}|{stat['Max Tasks/Teacher']:<{W2}}"
              f"|{stat['Conflict Edges']:<{W3}}|{stat['Load Std']:<{W4}.2f}|")

    _diversity_report(profiler_stats)


if __name__ == "__main__":
    np.random.seed(42)
    generate_datasets(20)
