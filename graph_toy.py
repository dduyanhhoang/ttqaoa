import pandas as pd
import networkx as nx
import pickle
from pathlib import Path

DATA_DIR = Path("data/raw/toy")
FILES = {
    # The task file containing the 5 toy classes
    "task": "Task.csv",
    "slot_conflict": "SlotConflict.csv",
    "instructor_slot": "InstructorSlot.csv",
    "instructor_skill": "InstructorSkill.csv",
    "instructor_quota": "InstructorQuota.csv"
}
OUTPUT_PATH = Path("data/processed/toy_graph_dense.pickle")


def load_toy_data(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    def read_clean(key, index_col=None):
        filename = FILES.get(key)
        if not filename:
            print(f"Warning: No filename configured for {key}")
            return pd.DataFrame()

        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path, index_col=index_col)

        df = df.dropna(how='all')
        if index_col is not None:
            df = df.dropna(axis=1, how='all')
        return df

    print(f"Loading data from {data_dir}...")

    df_task = read_clean("task")
    df_slotconf = read_clean("slot_conflict", index_col=0)
    df_instructor_slot = read_clean("instructor_slot", index_col=0)
    df_instructor_skill = read_clean("instructor_skill", index_col=0)
    df_instructor_quota = read_clean("instructor_quota", index_col=0)

    df_slotconf = df_slotconf.fillna(0).astype(int)
    df_instructor_slot = df_instructor_slot.fillna(0).astype(int)
    df_instructor_skill = df_instructor_skill.fillna(0).astype(int)

    data = {
        "task_list": df_task[["Class", "Subject", "Slot"]].to_dict("records"),
        "slot_conflict": df_slotconf.to_dict(),
        "instructor_slot": df_instructor_slot.to_dict(),
        "instructor_skill": df_instructor_skill.to_dict(),
        "instructor_quota": df_instructor_quota.to_dict("index"),
    }

    return data


def build_graph_from_toy():
    data = load_toy_data(DATA_DIR)

    tasks = data["task_list"]
    conflicts = data["slot_conflict"]

    print(f"Loaded {len(tasks)} tasks.")

    G = nx.Graph()

    for i, task in enumerate(tasks):
        G.add_node(i,
                   label=task["Class"],
                   slot=task["Slot"],
                   subject=task["Subject"])

    edge_count = 0
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            slot_i = tasks[i]["Slot"]
            slot_j = tasks[j]["Slot"]

            is_conflict = False

            if slot_i in conflicts and slot_j in conflicts[slot_i]:
                if conflicts[slot_i][slot_j] == 1:
                    is_conflict = True

            if is_conflict:
                G.add_edge(i, j)
                edge_count += 1
                print(f"Edge added: {tasks[i]['Class']} <--> {tasks[j]['Class']} (Slot {slot_i} vs {slot_j})")

    return G, data


if __name__ == "__main__":
    try:
        G, raw_data = build_graph_from_toy()

        print("\n=== Graph Construction Verification ===")
        print(f"Number of Nodes: {G.number_of_nodes()}")
        print(f"Number of Edges: {G.number_of_edges()}")
        print("Nodes:", G.nodes(data=True))

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(G, f)
        print(f"\nGraph saved to {OUTPUT_PATH}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure your CSV files are in 'data/raw/toy/' and filenames match the CONFIGURATION.")
