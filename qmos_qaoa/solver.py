import pennylane as qml
from pennylane import numpy as np
import scipy


def solve_qaoa(problem, depth=3, shots=20000, seed=42):
    """
    Runs the QAOA optimization using exact simulation on lightning.qubit,
    then samples the result.
    """
    n_qubits = problem.n_qubits
    cost_h = problem.cost_h
    cost_h = qml.simplify(cost_h)

    device_name = "lightning.qubit"

    # ---------------------------------------------------------
    # 1. Optimization Step (Use Exact/Analytic Expectation)
    # ---------------------------------------------------------
    dev_exact = qml.device(device_name, wires=n_qubits)

    def qaoa_layer(gamma, beta):
        qml.templates.TrotterProduct(cost_h, gamma, order=1, n=1)
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)

    @qml.qnode(dev_exact)
    def cost_circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for gamma, beta in params:
            qaoa_layer(gamma, beta)
        return qml.expval(cost_h)

    # Initialize parameters
    np.random.seed(seed)
    init_params = 0.01 * np.random.randn(depth, 2)

    print(f"Starting optimization with depth p={depth} on {device_name}...")
    opt_result = scipy.optimize.minimize(
        lambda x: cost_circuit(x.reshape((depth, 2))),
        init_params.flatten(),
        method="COBYLA"
    )

    print(f"Optimal Cost: {opt_result.fun}")

    # ---------------------------------------------------------
    # 2. Sampling Step (Use Finite Shots)
    # ---------------------------------------------------------
    final_params = opt_result.x.reshape((depth, 2))
    dev_sample = qml.device(device_name, wires=n_qubits)

    @qml.set_shots(shots=shots)
    @qml.qnode(dev_sample)
    def sample_circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for gamma, beta in params:
            qaoa_layer(gamma, beta)
        return qml.sample(wires=range(n_qubits))

    print(f"Sampling {shots} shots...")
    samples = sample_circuit(final_params)

    return samples
