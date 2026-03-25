from loguru import logger
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd


def load_local_data(data_dir: Path | str) -> Dict:
    """
    Loads, cleans, and transforms timetable scheduling data CSV files.

    It first loads and drops completely empty rows and columns.
    Then it converts relevant DataFrames to binary numpy.int8 for later processing.

    Args:
        data_dir (Path | str): The directory path containing the raw CSV files.

    Returns:
        Dict[str, Any]: A dictionary containing the processed data:
            - 'task_list' (List[Dict]): List of tasks with 'Class', 'Subject', and 'Slot'.
            - 'slot_conflict' (pd.DataFrame): Binary matrix representing slot overlap conflicts.
            - 'instructor_slot' (pd.DataFrame): Instructor availability per slot.
            - 'instructor_skill' (pd.DataFrame): Instructor skill levels per subject.
            - 'instructor_quota' (pd.DataFrame): Min/max class quota per instructor.

    Raises:
        FileNotFoundError: If any required CSV files are missing in the target directory.
    """

    # Ensure input is a Path object
    data_dir = Path(data_dir)
    logger.info(f"Loading and processing data from: {data_dir}")
    files = {
        "task": "Task.csv",
        "slot_conflict": "SlotConflict.csv",
        "instructor_slot": "InstructorSlot.csv",
        "instructor_skill": "InstructorSkill.csv",
        "instructor_quota": "InstructorQuota.csv"
    }

    def read_clean_(filename: str,
                    index_col: Optional[int] = None) -> pd.DataFrame:
        """Helper function to load and drop completely empty rows/cols."""
        path = data_dir / filename
        if not path.exists():
            logger.error(f"Missing required data file: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path, index_col=index_col)
        df = df.dropna(how='all')
        if index_col is not None:
            df = df.dropna(axis=1, how='all')

        logger.debug(f"Loaded {filename} | shape: {df.shape}")
        return df

    try:
        df_task = read_clean_(files["task"])
        df_slotconf = read_clean_(files["slot_conflict"], index_col=0)
        df_instructor_slot = read_clean_(files["instructor_slot"], index_col=0)
        df_instructor_skill = read_clean_(files["instructor_skill"], index_col=0)
        df_instructor_quota = read_clean_(files["instructor_quota"], index_col=0)
    except Exception as e:
        logger.exception("Data loading sequence failed.")
        raise e

    for col in ["Class", "Subject", "Slot"]:
        if col in df_task.columns and df_task[col].dtype == 'object':
            df_task[col] = df_task[col].str.strip()

    df_instructor_slot = df_instructor_slot.fillna(0).astype(np.int8)
    df_instructor_skill = df_instructor_skill.fillna(0).astype(np.int8)
    df_slotconf = df_slotconf.fillna(0).astype(np.int8)

    data = {
        "task_list": df_task[["Class", "Subject", "Slot"]].to_dict("records"),
        "slot_conflict": df_slotconf,
        "instructor_slot": df_instructor_slot,
        "instructor_skill": df_instructor_skill,
        "instructor_quota": df_instructor_quota,
    }

    logger.success("All data files successfully loaded and formatted.")
    return data
