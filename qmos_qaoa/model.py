from functools import reduce
from loguru import logger
from typing import Dict, Any, List, Tuple
import pandas as pd
import pennylane as qml


def qubit_proj(idx: int):
    """
    Constructs the projector P1 = |1><1| = (I - Z)/2 for a specific qubit.

    This projector maps a Pauli-Z measurement back to the binary {0, 1} domain.

    Args:
        idx (int): The qubit index (wire) to apply the projector to.

    Returns:
        qml.Hamiltonian: The PennyLane observable representing the projector.
    """
    return (1 - qml.PauliZ(idx)) / 2


class TimetableProblem:
    """Formulates a university timetabling instance as a QUBO cost Hamiltonian.

    Decision variables x_{t,s} ∈ {0,1} indicate whether teacher t is assigned
    to section s.  Three penalty terms are summed into a single cost Hamiltonian:

        H_C = W_section^2 * Σ_s (Σ_t x_{t,s} - 1)^2          (one teacher per section)
            + W_conflict^2 * Σ_{conflict(s1,s2)} (x_{t,s1} + x_{t,s2})^2  (no overlapping slots)
            + W_quota^2   * Σ_t [(Σ_s x_{t,s} - q_min)^2
                                + (Σ_s x_{t,s} - q_max)^2]    (teacher quota)

    Default weights follow Hong Anh's code: W_section=3, W_conflict=2, W_quota=1.
    Qubit projectors n_i = (I - Z_i)/2 map the Pauli-Z basis to {0,1}.
    """

    def __init__(self,
                 data: Dict[str, Any],
                 w_section: float = 3.0,
                 w_conflict: float = 2.0,
                 w_quota: float = 1.0,
                 skill_threshold: int = 5,
                 slot_threshold: int = 5) -> None:
        """
        Initializes the problem dimensions and constraint weights.

        It processes the input data to identify feasible teacher-section assignments based on
        skill and availability thresholds, then build the cost Hamiltonian.

        Args:
            data (Dict[str, Any]): The processed data dictionary from `data.py`.
            w_section (float): Penalty weight for the "One teacher per section" constraint.
            w_conflict (float): Penalty weight for the "No overlapping slots" constraint.
            w_quota (float): Penalty weight for the "Teacher quota" constraint.
            skill_threshold (int): Minimum skill level required for a teacher to be assigned a subject.
            slot_threshold (int): Minimum availability required for a teacher to teach at a slot.
        """
        logger.info("Initializing TimetableProblem formulation.")

        self.task_list: List[Dict] = data["task_list"]
        self.slot_conflict: pd.DataFrame = data["slot_conflict"]
        self.instructor_slot: pd.DataFrame = data["instructor_slot"]
        self.instructor_skill: pd.DataFrame = data["instructor_skill"]
        self.instructor_quota: pd.DataFrame = data["instructor_quota"]
        self.weights = {
            "section": w_section,
            "conflict": w_conflict,
            "quota": w_quota
        }
        self.skill_threshold = skill_threshold
        self.slot_threshold = slot_threshold
        self.teachers = list(self.instructor_skill.index)
        self.sections = list(range(len(self.task_list)))

        logger.debug(f"Problem space initialized with {len(self.teachers)} teachers "
                     f"and {len(self.sections)} sections.")

        self.instructor_slot_by_teacher = {t: {slot: self.instructor_slot[slot][t]
                                               for slot in self.instructor_slot}
                                           for t in self.teachers}
        self.instructor_skill_by_teacher = {t: {subj: self.instructor_skill[subj][t]
                                                for subj in self.instructor_skill}
                                            for t in self.teachers}

        self.feasible_vars = self._find_feasible_vars()
        self.var_to_idx = {(t, s): idx for idx, (t, s) in enumerate(self.feasible_vars)}
        self.n_qubits = len(self.feasible_vars)

        logger.info(f"Total qubits required after filtering: {self.n_qubits}")

        self.cost_h = self._build_hamiltonian()

    def _find_feasible_vars(self) -> List[Tuple]:
        """
        Filters the global decision space down to only feasible (Teacher, Section) pairs.

        Instead of creating a qubit for every possible teacher-to-section assignment
        (require |Teachers| * |Sections| qubits), this function prunes assignments
        where the teacher does not have the required skill or availability.

        Returns:
            List[Tuple[int, int]]: A list of valid (teacher_idx, section_idx) tuples.
        """
        feasible = []
        for t_idx, teacher in enumerate(self.teachers):
            for s_idx, section in enumerate(self.task_list):
                subj = section['Subject']
                slot = section['Slot']

                skill_level = 0
                if subj in self.instructor_skill.columns and teacher in self.instructor_skill.index:
                    skill_level = self.instructor_skill.at[teacher, subj]

                slot_avail = 0
                if slot in self.instructor_slot.columns and teacher in self.instructor_slot.index:
                    slot_avail = self.instructor_slot.at[teacher, slot]

                if skill_level >= self.skill_threshold and slot_avail >= self.slot_threshold:
                    feasible.append((t_idx, s_idx))

        total_possible = len(self.teachers) * len(self.sections)
        logger.debug(f"Search space reduced from {total_possible} to {len(feasible)} variables.")
        return feasible

    def _build_hamiltonian(self):
        """
        Constructs the cost Hamiltonian by converting scheduling constraints into QUBO penalties.

        Section terms:
            - Rule: Every section must be assigned exactly one teacher.
            - QUBO: min((sum(x_{t, s}) - 1)^2) for each section s.
        Conflict terms:
            - Rule: A teacher cannot be assigned to two sections that have overlapping time slots.
            - QUBO: min(sum(x_{t, s1} * x_{t, s2})) for all conflicting pairs (s1, s2)
        Quota terms:
            - Rule: Each teacher must be assigned a number of sections within their min/max quota.
            - QUBO: min((sum(x_{t, s}) - q_min)^2 + (sum(x_{t, s}) - q_max)^2) for each teacher t.

        Returns:
            qml.Hamiltonian: The final summed cost Hamiltonian to be minimized.
        """
        logger.info("Constructing cost Hamiltonian.")
        terms = []

        for s_idx in self.sections:
            relevant_vars = [self.var_to_idx[(t, s_idx)]
                             for t in range(len(self.teachers))
                             if (t, s_idx) in self.var_to_idx]
            if relevant_vars:
                expr = sum([qubit_proj(idx) for idx in relevant_vars]) - 1
                terms.append((self.weights["section"] * expr) ** 2)

        for t_idx, teacher in enumerate(self.teachers):
            teacher_sections = [s for (tt, s) in self.feasible_vars if tt == t_idx]
            for i1, s1 in enumerate(teacher_sections):
                slot1 = self.task_list[s1]['Slot']
                for s2 in teacher_sections[i1 + 1:]:
                    slot2 = self.task_list[s2]['Slot']

                    conflict = 0
                    if slot1 in self.slot_conflict.index and slot2 in self.slot_conflict.columns:
                        conflict = self.slot_conflict.at[slot1, slot2]

                    if conflict == 1:
                        idx1 = self.var_to_idx[(t_idx, s1)]
                        idx2 = self.var_to_idx[(t_idx, s2)]
                        terms.append((self.weights["conflict"] * (qubit_proj(idx1) + qubit_proj(idx2))) ** 2)

        for t_idx, teacher in enumerate(self.teachers):
            relevant_vars = [self.var_to_idx[(t_idx, s_idx)]
                             for (t_idx2, s_idx) in self.feasible_vars
                             if t_idx2 == t_idx]
            if not relevant_vars:
                continue

            total_expr = sum([qubit_proj(idx) for idx in relevant_vars])
            qmin = self.instructor_quota.at[teacher, "Min quota"]
            qmax = self.instructor_quota.at[teacher, "Max quota"]

            terms.append((self.weights["quota"] * (total_expr - qmin)) ** 2)
            terms.append((self.weights["quota"] * (total_expr - qmax)) ** 2)

        if not terms:
            logger.warning("No Hamiltonian terms were generated. Check data and thresholds.")
            return qml.Hamiltonian([], [])
        cost_h = reduce(lambda a, b: a + b, terms)
        logger.success("Hamiltonian construction complete.")

        return cost_h
