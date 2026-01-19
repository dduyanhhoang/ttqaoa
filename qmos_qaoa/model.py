import pennylane as qml
from functools import reduce


def qubit_proj(idx):
    """Projector P1 = |1><1| = (I - Z)/2"""
    return (1 - qml.PauliZ(idx)) / 2


class TimetableProblem:
    def __init__(self, data, w_section=300, w_conflict=300, w_quota=2.0):
        self.task_list = data["task_list"]
        self.slot_conflict = data["slot_conflict"]
        self.instructor_slot = data["instructor_slot"]
        self.instructor_skill = data["instructor_skill"]
        self.instructor_quota = data["instructor_quota"]

        self.weights = {
            "section": w_section,
            "conflict": w_conflict,
            "quota": w_quota
        }

        # Initialize problem definition
        self.teachers = sorted(list(self.instructor_skill[list(self.instructor_skill.keys())[0]].keys()))
        self.sections = list(range(len(self.task_list)))

        # Pre-process lookups
        self.instructor_slot_by_teacher = {t: {slot: self.instructor_slot[slot][t] for slot in self.instructor_slot} for
                                           t in self.teachers}
        self.instructor_skill_by_teacher = {t: {subj: self.instructor_skill[subj][t] for subj in self.instructor_skill}
                                            for t in self.teachers}

        # Build Graph / Hamiltonian
        self.feasible_vars = self._find_feasible_vars()
        self.var_to_idx = {(t, s): idx for idx, (t, s) in enumerate(self.feasible_vars)}
        self.n_qubits = len(self.feasible_vars)
        self.cost_h = self._build_hamiltonian()

    def _find_feasible_vars(self):
        feasible = []
        for t_idx, teacher in enumerate(self.teachers):
            for s_idx, section in enumerate(self.task_list):
                subj = section['Subject']
                slot = section['Slot']
                # Check constraints (Skill >= 5 and Slot Availability >= 5)
                if (self.instructor_skill_by_teacher[teacher].get(subj, 0) >= 5 and
                        self.instructor_slot_by_teacher[teacher].get(slot, 0) >= 5):
                    feasible.append((t_idx, s_idx))
        return feasible

    def _build_hamiltonian(self):
        terms = []

        # 1. Section Terms (Each section must have exactly one teacher)
        for s_idx in self.sections:
            relevant_vars = [self.var_to_idx[(t, s_idx)]
                             for t in range(len(self.teachers))
                             if (t, s_idx) in self.var_to_idx]
            if relevant_vars:
                expr = sum([qubit_proj(idx) for idx in relevant_vars]) - 1
                terms.append((self.weights["section"] * expr) ** 2)

        # 2. Conflict Terms (No overlapping slots for the same teacher)
        for t_idx, teacher in enumerate(self.teachers):
            teacher_sections = [s for (tt, s) in self.feasible_vars if tt == t_idx]
            for i1, s1 in enumerate(teacher_sections):
                slot1 = self.task_list[s1]['Slot']
                for s2 in teacher_sections[i1 + 1:]:
                    slot2 = self.task_list[s2]['Slot']
                    if self.slot_conflict.get(slot1, {}).get(slot2, 0) == 1:
                        idx1 = self.var_to_idx[(t_idx, s1)]
                        idx2 = self.var_to_idx[(t_idx, s2)]
                        terms.append((self.weights["conflict"] * (qubit_proj(idx1) + qubit_proj(idx2))) ** 2)

        # 3. Quota Terms (Min/Max classes per teacher)
        for t_idx, teacher in enumerate(self.teachers):
            relevant_vars = [self.var_to_idx[(t_idx, s_idx)]
                             for (t_idx2, s_idx) in self.feasible_vars
                             if t_idx2 == t_idx]
            if not relevant_vars:
                continue

            total_expr = sum([qubit_proj(idx) for idx in relevant_vars])
            qmin = self.instructor_quota[teacher]["Min quota"]
            qmax = self.instructor_quota[teacher]["Max quota"]

            terms.append((self.weights["quota"] * (total_expr - qmin)) ** 2)
            terms.append((self.weights["quota"] * (total_expr - qmax)) ** 2)

        cost_h = reduce(lambda a, b: a + b, terms)

        return cost_h
