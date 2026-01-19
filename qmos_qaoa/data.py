from pathlib import Path
import pandas as pd


def load_local_data(data_dir: Path):
    """
    Loads and cleans timetable data from local CSV files using pathlib.
    """
    # Ensure input is a Path object
    data_dir = Path(data_dir)

    files = {
        "task": "QAOA_Data_Fall_2025 - Task_easy.csv",
        "slot_conflict": "QAOA_Data_Fall_2025 - SlotConflict_easy.csv",
        "instructor_slot": "QAOA_Data_Fall_2025 - InstructorSlot_easy.csv",
        "instructor_skill": "QAOA_Data_Fall_2025 - InstructorSkill_easy.csv",
        "instructor_quota": "QAOA_Data_Fall_2025 - InstructorQuota_easy.csv"
    }

    # Helper to read and clean
    def read_clean(filename, index_col=None):
        # Use the / operator to join paths
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path, index_col=index_col)
        df = df.dropna(how='all')
        if index_col is not None:
            df = df.dropna(axis=1, how='all')
        return df

    print(f"Loading data from {data_dir}...")

    # Load DataFrames
    df_task = read_clean(files["task"])
    df_slotconf = read_clean(files["slot_conflict"], index_col=0)
    df_instructor_slot = read_clean(files["instructor_slot"], index_col=0)
    df_instructor_skill = read_clean(files["instructor_skill"], index_col=0)
    df_instructor_quota = read_clean(files["instructor_quota"], index_col=0)

    # Cleanup / Conversion
    df_instructor_slot = df_instructor_slot.fillna(0).astype(int)
    df_instructor_skill = df_instructor_skill.fillna(0).astype(int)
    df_slotconf = df_slotconf.fillna(0).astype(int)

    # Build dictionaries
    data = {
        "task_list": df_task[["Class", "Subject", "Slot"]].to_dict("records"),
        "slot_conflict": df_slotconf.to_dict(),
        "instructor_slot": df_instructor_slot.to_dict(),
        "instructor_skill": df_instructor_skill.to_dict(),
        "instructor_quota": df_instructor_quota.to_dict("index"),
    }

    return data
