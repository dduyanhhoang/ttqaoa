from joblib import Parallel, delayed
from loguru import logger
from pennylane import numpy as np
from typing import Tuple, List, Any
import pennylane as qml
import scipy


def _ensure_hamiltonian(op: qml.operation.Operator) -> qml.Hamiltonian:
    """
    Ensures that a PennyLane operator is strictly cast to a qml.Hamiltonian object.

    Assist joblib in handling the cost operator by converting it to a structurally
    flat Hamiltonian.

    Args:
        op (qml.operation.Operator): The cost operator to be converted.

    Returns:
        qml.Hamiltonian: A structurally flat Hamiltonian.
    """
    if isinstance(op, qml.Hamiltonian):
        return op

    try:
        return qml.pauli.pauli_sentence(op).hamiltonian()
    except Exception:
        pass

    coeffs = []
    ops = []

    def _extract_term(term):
        if isinstance(term, qml.ops.SProd):
            return term.scalar, term.base
        return 1.0, term

    if isinstance(op, qml.ops.Sum):
        for term in op.operands:
            c, o = _extract_term(term)
            coeffs.append(c)
            ops.append(o)
    else:
        c, o = _extract_term(op)
        coeffs.append(c)
        ops.append(o)

    return qml.Hamiltonian(coeffs, ops)


def _run_single_optimization(depth: int,
                             n_qubits: int,
                             cost_h_ops: List,
                             cost_h_coeffs: List,
                             seed: int
                             ) -> Tuple[float, np.ndarray]:
    """
    Runs a single COBYLA optimization trajectory for the QAOA circuit.
    This function is designed to be executed in parallel by joblib.

    Seeds are scattered uniformly across the optimization runs to ensure
    diverse starting points in the parameter space, avoiding local minima traps.

    Args:
        depth (int): The number of QAOA layers (p).
        n_qubits (int): Total number of decision variables/qubits.
        cost_h_ops (List): The list of Pauli operators from the cost Hamiltonian.
        cost_h_coeffs (List): The corresponding coefficients for the operators.
        seed (int): Random seed to determine the starting parameters.

    Returns:
        Tuple[float, np.ndarray]: The best cost found, and the flattened optimal parameters.
    """
    cost_h = qml.Hamiltonian(cost_h_coeffs, cost_h_ops)
    dev = qml.device("lightning.qubit", wires=n_qubits)

    def qaoa_layer(gamma, beta):
        qml.templates.TrotterProduct(cost_h, gamma, order=1, n=1)
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)

    @qml.qnode(dev)
    def cost_circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for gamma, beta in params:
            qaoa_layer(gamma, beta)
        return qml.expval(cost_h)

    rng = np.random.default_rng(seed)

    if seed == 42:
        init_params = 0.01 * rng.standard_normal((depth, 2))
    else:
        gamma_init = rng.uniform(0, np.pi, depth)
        beta_init = rng.uniform(0, np.pi, depth)
        init_params = np.column_stack((gamma_init, beta_init))

    opt_result = scipy.optimize.minimize(
        lambda x: cost_circuit(x.reshape((depth, 2))),
        init_params.flatten(),
        method="COBYLA",
        options={"maxiter": 2000}
    )

    return opt_result.fun, opt_result.x


def solve_qaoa(problem: Any,
               depth: int = 3,
               shots: int = 20000,
               restarts: int = 10,
               seed: int = 42,
               n_jobs: int = -1
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes a Multi-Start QAOA optimization and samples the optimal state.

    Simplify the cost_h to run on lightning.qubit, ensure it's a Hamiltonian object for joblib
    compatibility. Then run independent multi-start optimizations in parallel, each with a
    different random seed to explore the parameter space.

    Select the trajectory that yields the lowest cost and sample from the corresponding optimal
    circuit. Then perform finite-shot sampling at the optimal parameters to obtain the final
    bitstring samples.

    Args:
        problem (TimetableProblem): The formulated QUBO problem.
        depth (int): The number of QAOA layers (p). Default is 3.
        shots (int): The number of finite shots for the final sampling. Default is 20000.
        restarts (int): The number of independent optimization runs to avoid local minima.
        seed (int): Base random seed.
        n_jobs (int): Number of parallel CPU workers to use (-1 for all cores).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - samples: An array of shape (shots, n_qubits) containing the measured bitstrings.
            - final_params: An array of shape (depth, 2) containing the optimal (gamma, beta) angles.
    """
    n_qubits = problem.n_qubits

    logger.info("Simplifying and formatting Cost Hamiltonian for execution.")
    cost_h = qml.simplify(problem.cost_h)
    cost_h = _ensure_hamiltonian(cost_h)

    cost_h_ops = cost_h.ops
    cost_h_coeffs = cost_h.coeffs

    logger.info(f"Starting QAOA (p={depth}) with {restarts} restarts across {n_jobs} parallel jobs.")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_optimization)(
            depth, n_qubits, cost_h_ops, cost_h_coeffs, seed + i
        )
        for i in range(restarts)
    )

    best_cost, best_params_flat = min(results, key=lambda x: x[0])
    logger.success(f"Global Optimization Complete. Best Cost found: {best_cost:.4f}")

    final_params = best_params_flat.reshape((depth, 2))

    logger.info(f"Sampling optimal circuit with {shots} shots.")
    dev_sample = qml.device("lightning.qubit", wires=n_qubits)
    cost_h_local = qml.Hamiltonian(cost_h_coeffs, cost_h_ops)

    def qaoa_layer(gamma, beta):
        qml.templates.TrotterProduct(cost_h_local, gamma, order=1, n=1)
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)

    @qml.set_shots(shots=shots)
    @qml.qnode(dev_sample)
    def sample_circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for gamma, beta in params:
            qaoa_layer(gamma, beta)
        return qml.sample(wires=range(n_qubits))

    samples = sample_circuit(final_params)
    logger.success("Sampling complete.")

    return samples, final_params
