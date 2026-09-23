"""Integration tests for the KaMIS maximum-weight independent-set backend."""

from pathlib import Path

import pytest

from hamiltonian import build_cost_hamiltonian, extract_hamiltonian_terms
from kamis_mwis import (
    resolve_kamis_binary,
    scale_weights_for_kamis,
    solve_mwis_kamis,
)
from mosaic_graph import (
    build_incompatibility_graph,
    is_independent_set,
    solve_mwis_bruteforce,
)
from operator_pool import build_operator_pool
from tetris_qaoa_scoring import compute_tetris_qaoa_pool_scores


NUM_QUBITS = 3
GAMMA_0 = 0.01
GRADIENT_THRESHOLD = 1e-3
KAMIS_SEED = 42

FORMULA = [
    [(0, False), (1, False), (2, False)],
    [(0, True), (1, False), (2, False)],
]


@pytest.fixture(scope="module")
def mwis_problem():
    cost_hamiltonian = build_cost_hamiltonian(FORMULA)
    coefficients, pauli_words = extract_hamiltonian_terms(
        cost_hamiltonian,
        NUM_QUBITS,
    )

    operator_pool = build_operator_pool(NUM_QUBITS)

    scores = compute_tetris_qaoa_pool_scores(
        layers=[],
        gamma0=GAMMA_0,
        operator_pool=operator_pool,
        n_qubits=NUM_QUBITS,
        cost_coefficients=coefficients,
        cost_pauli_words=pauli_words,
    )

    scores_by_name = {
        entry["name"]: entry
        for entry in scores
    }

    # Preserve pool order so graph node ordering is deterministic.
    eligible = [
        scores_by_name[entry["name"]]
        for entry in operator_pool
        if scores_by_name[entry["name"]]["abs_gradient"]
        >= GRADIENT_THRESHOLD
    ]

    graph = build_incompatibility_graph(eligible)

    return eligible, graph


def test_kamis_binary_is_available():
    binary = Path(resolve_kamis_binary())

    assert binary.is_file()
    assert binary.stat().st_mode & 0o111


def test_reference_problem_has_expected_candidate_count(mwis_problem):
    eligible, _ = mwis_problem

    assert len(eligible) == 17


def test_weight_conversion_produces_valid_integer_weights(mwis_problem):
    eligible, _ = mwis_problem

    integer_weights, metadata = scale_weights_for_kamis(eligible)

    assert set(integer_weights) == {
        entry["name"]
        for entry in eligible
    }

    assert all(
        isinstance(weight, int) and weight > 0
        for weight in integer_weights.values()
    )

    assert metadata["total_integer_weight"] == sum(
        integer_weights.values()
    )


def test_kamis_matches_exact_integer_objective(mwis_problem):
    eligible, graph = mwis_problem

    integer_weights, _ = scale_weights_for_kamis(eligible)

    integer_problem = []

    for entry in eligible:
        weighted_entry = dict(entry)
        weighted_entry["abs_gradient"] = integer_weights[entry["name"]]
        integer_problem.append(weighted_entry)

    _, exact_integer_weight = solve_mwis_bruteforce(
        integer_problem,
        graph,
    )

    selected_names, _ = solve_mwis_kamis(
        operator_results=eligible,
        graph=graph,
        seed=KAMIS_SEED,
    )

    assert is_independent_set(selected_names, graph)

    kamis_integer_weight = sum(
        integer_weights[name]
        for name in selected_names
    )

    assert kamis_integer_weight == exact_integer_weight


def test_kamis_reports_selected_float_weight(mwis_problem):
    eligible, graph = mwis_problem

    selected_names, reported_weight = solve_mwis_kamis(
        operator_results=eligible,
        graph=graph,
        seed=KAMIS_SEED,
    )

    scores_by_name = {
        entry["name"]: entry["abs_gradient"]
        for entry in eligible
    }

    expected_weight = sum(
        scores_by_name[name]
        for name in selected_names
    )

    assert reported_weight == pytest.approx(
        expected_weight,
        abs=1e-12,
        rel=0.0,
    )
