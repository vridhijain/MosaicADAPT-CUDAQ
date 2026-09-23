"""Regression tests for first-layer BFGS optimization.

The reference values come from the Julia MosaicADAPT-QAOA
implementation using Optim.jl. SciPy and Optim.jl are not expected
to follow identical optimization trajectories, so the tests compare
the converged physical solution rather than iteration counts.
"""

import math

import numpy as np
import pytest

from ansatz_executor import evaluate_ansatz
from bfgs_optimizer import optimize_ansatz_bfgs
from hamiltonian import build_maxcut_hamiltonian, extract_hamiltonian_terms
from operator_pool import build_qaoa_double_pool
from optimization import pack_parameters
from variational_gradients import compute_variational_gradient


NUM_QUBITS = 6
GAMMA_0 = 0.001
GRADIENT_TOLERANCE = 1e-6

REFERENCE_EDGES = [
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 5),
    (3, 5),
]

# Julia uses one-based pool indices.
JULIA_SELECTED_INDICES = [23, 31, 62]
JULIA_SELECTED_NAMES = ["Z0Y4", "Z1Y2", "Y3Z5"]

JULIA_FINAL_PARAMETERS = np.array(
    [
        1.3825275090112525e-11,
        0.7853981973282732,
        0.7853980952523312,
        0.7853981973282728,
    ],
    dtype=float,
)

JULIA_FINAL_ENERGY = -3.999999999999994


COST_HAMILTONIAN = build_maxcut_hamiltonian(
    NUM_QUBITS,
    REFERENCE_EDGES,
)

COST_COEFFICIENTS, COST_PAULI_WORDS = extract_hamiltonian_terms(
    COST_HAMILTONIAN,
    NUM_QUBITS,
)

OPERATOR_POOL = build_qaoa_double_pool(NUM_QUBITS)
OPERATORS = {entry["name"]: entry for entry in OPERATOR_POOL}

INITIAL_LAYERS = [
    {
        "gamma": GAMMA_0,
        "operators": [
            OPERATORS["Z0Y4"],
            OPERATORS["Z1Y2"],
            OPERATORS["Y3Z5"],
        ],
        "betas": [0.0, 0.0, 0.0],
    }
]


@pytest.fixture(scope="module")
def optimized_result():
    """Run the first-layer optimization once for this test module."""

    return optimize_ansatz_bfgs(
        layers=INITIAL_LAYERS,
        cost_hamiltonian=COST_HAMILTONIAN,
        n_qubits=NUM_QUBITS,
        cost_coefficients=COST_COEFFICIENTS,
        cost_pauli_words=COST_PAULI_WORDS,
        gtol=GRADIENT_TOLERANCE,
        maxiter=None,
        verbose=False,
    )


def test_reference_pool_indices():
    assert len(OPERATOR_POOL) == 67

    selected_names = [
        OPERATOR_POOL[index - 1]["name"]
        for index in JULIA_SELECTED_INDICES
    ]

    assert selected_names == JULIA_SELECTED_NAMES


def test_initial_layer_matches_reference():
    parameters = np.asarray(
        pack_parameters(INITIAL_LAYERS),
        dtype=float,
    )

    expected_parameters = np.array(
        [GAMMA_0, 0.0, 0.0, 0.0],
        dtype=float,
    )

    initial_energy = evaluate_ansatz(
        layers=INITIAL_LAYERS,
        cost_hamiltonian=COST_HAMILTONIAN,
        n_qubits=NUM_QUBITS,
        cost_coefficients=COST_COEFFICIENTS,
        cost_pauli_words=COST_PAULI_WORDS,
    )

    assert np.array_equal(parameters, expected_parameters)
    assert abs(initial_energy + 2.5) < 1e-12


def test_scipy_bfgs_converges(optimized_result):
    (
        optimized_layers,
        optimized_energy,
        result,
        _objective_history,
        _gradient_history,
    ) = optimized_result

    final_gradient = np.asarray(
        compute_variational_gradient(
            layers=optimized_layers,
            n_qubits=NUM_QUBITS,
            cost_coefficients=COST_COEFFICIENTS,
            cost_pauli_words=COST_PAULI_WORDS,
        ),
        dtype=float,
    )

    assert result.success
    assert np.linalg.norm(final_gradient, ord=np.inf) < GRADIENT_TOLERANCE
    assert abs(optimized_energy - JULIA_FINAL_ENERGY) < 1e-10


def test_optimized_parameters_match_physical_solution(optimized_result):
    optimized_layers, _, _, _, _ = optimized_result

    parameters = np.asarray(
        pack_parameters(optimized_layers),
        dtype=float,
    )

    pi_over_4 = math.pi / 4.0

    assert abs(parameters[0]) < 1e-5
    assert np.allclose(
        parameters[1:],
        pi_over_4,
        atol=1e-5,
        rtol=0.0,
    )


def test_scipy_and_julia_endpoints_are_close(optimized_result):
    optimized_layers, _, _, _, _ = optimized_result

    parameters = np.asarray(
        pack_parameters(optimized_layers),
        dtype=float,
    )

    # The optimizers need not return identical floating-point endpoints.
    # This tolerance checks agreement without requiring identical trajectories.
    assert np.allclose(
        parameters,
        JULIA_FINAL_PARAMETERS,
        atol=1e-5,
        rtol=0.0,
    )
