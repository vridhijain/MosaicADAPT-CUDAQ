"""Regression tests for the full variational gradient implementation."""

import numpy as np

from ansatz_executor import evaluate_ansatz
from hamiltonian import build_cost_hamiltonian, extract_hamiltonian_terms
from operator_pool import build_operator_pool
from optimization import pack_parameters, unpack_parameters
from variational_gradients import (
    apply_pauli_word,
    compute_variational_gradient,
    get_ansatz_statevector,
    simulate_layers_numpy,
)


NUM_QUBITS = 3
FINITE_DIFFERENCE_EPSILON = 1e-6

FORMULA = [
    [(0, False), (1, False), (2, False)],
    [(0, True), (1, False), (2, False)],
]

COST_HAMILTONIAN = build_cost_hamiltonian(FORMULA)
COST_COEFFICIENTS, COST_PAULI_WORDS = extract_hamiltonian_terms(
    COST_HAMILTONIAN,
    NUM_QUBITS,
)

OPERATOR_POOL = build_operator_pool(NUM_QUBITS)
OPERATORS = {entry["name"]: entry for entry in OPERATOR_POOL}

LAYERS = [
    {
        "gamma": 0.11,
        "operators": [
            OPERATORS["X0Y2"],
            OPERATORS["Y1"],
        ],
        "betas": [0.21, -0.17],
    },
    {
        "gamma": -0.07,
        "operators": [
            OPERATORS["SUM_X"],
        ],
        "betas": [0.13],
    },
]


def energy_from_parameters(parameters):
    """Evaluate the ansatz energy for a packed parameter vector."""

    layers = unpack_parameters(parameters, LAYERS)

    return evaluate_ansatz(
        layers=layers,
        cost_hamiltonian=COST_HAMILTONIAN,
        n_qubits=NUM_QUBITS,
        cost_coefficients=COST_COEFFICIENTS,
        cost_pauli_words=COST_PAULI_WORDS,
    )


def finite_difference_gradient(parameters):
    """Compute a central finite-difference gradient for validation."""

    gradient = []

    for index in range(len(parameters)):
        plus = list(parameters)
        minus = list(parameters)

        plus[index] += FINITE_DIFFERENCE_EPSILON
        minus[index] -= FINITE_DIFFERENCE_EPSILON

        derivative = (
            energy_from_parameters(plus)
            - energy_from_parameters(minus)
        ) / (2.0 * FINITE_DIFFERENCE_EPSILON)

        gradient.append(derivative)

    return np.asarray(gradient, dtype=float)


def test_pauli_word_matches_cudaq_bit_order():
    """Qubit 0 corresponds to the least-significant statevector bit."""

    basis_000 = np.zeros(2**NUM_QUBITS, dtype=np.complex128)
    basis_000[0] = 1.0

    x0_state = apply_pauli_word(basis_000, "XII", NUM_QUBITS)
    x1_state = apply_pauli_word(basis_000, "IXI", NUM_QUBITS)
    x2_state = apply_pauli_word(basis_000, "IIX", NUM_QUBITS)

    assert abs(x0_state[1] - 1.0) < 1e-12
    assert abs(x1_state[2] - 1.0) < 1e-12
    assert abs(x2_state[4] - 1.0) < 1e-12


def test_cudaq_and_numpy_states_agree_up_to_global_phase():
    cudaq_state = get_ansatz_statevector(
        layers=LAYERS,
        n_qubits=NUM_QUBITS,
        cost_coefficients=COST_COEFFICIENTS,
        cost_pauli_words=COST_PAULI_WORDS,
    )

    numpy_state = simulate_layers_numpy(
        layers=LAYERS,
        n_qubits=NUM_QUBITS,
        cost_coefficients=COST_COEFFICIENTS,
        cost_pauli_words=COST_PAULI_WORDS,
    )

    # CUDA-Q may omit the global phase introduced by identity terms in
    # the cost Hamiltonian, so physical equivalence is tested after
    # phase alignment rather than by direct amplitude comparison.
    overlap = np.vdot(numpy_state, cudaq_state)
    overlap_magnitude = abs(overlap)

    assert abs(overlap_magnitude - 1.0) < 1e-10

    phase = overlap / overlap_magnitude
    aligned_numpy_state = phase * numpy_state

    assert np.max(np.abs(cudaq_state - aligned_numpy_state)) < 1e-10


def test_parameter_packing_uses_reference_order():
    parameters = pack_parameters(LAYERS)

    expected = np.asarray(
        [
            0.11,
            -0.07,
            0.21,
            -0.17,
            0.13,
        ]
    )

    assert len(parameters) == 5
    assert np.allclose(parameters, expected, atol=1e-12, rtol=0.0)


def test_variational_gradient_matches_finite_difference():
    analytic = compute_variational_gradient(
        layers=LAYERS,
        n_qubits=NUM_QUBITS,
        cost_coefficients=COST_COEFFICIENTS,
        cost_pauli_words=COST_PAULI_WORDS,
    )

    parameters = pack_parameters(LAYERS)
    numerical = finite_difference_gradient(parameters)

    assert len(analytic) == len(parameters)
    assert np.max(np.abs(analytic - numerical)) < 1e-6
