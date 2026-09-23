"""Analytic TetrisQAOA candidate scoring."""

import numpy as np

from variational_gradients import compute_variational_gradient


def _build_pool_scoring_layers(layers, gamma0, operator_pool):
    """Append a temporary scoring layer with all candidate betas set to zero."""
    scoring_layers = list(layers)

    scoring_layers.append(
        {
            "gamma": float(gamma0),
            "operators": list(operator_pool),
            "betas": [0.0] * len(operator_pool),
        }
    )

    return scoring_layers


def compute_tetris_qaoa_pool_scores(
    layers,
    gamma0,
    operator_pool,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
):
    """Compute analytic TetrisQAOA scores for the complete operator pool.

    The current optimized ansatz is followed by a provisional cost
    evolution with angle gamma0 and one zero-angle beta parameter for
    every candidate mixer. Because the candidate unitaries are identity
    at beta = 0, all candidate derivatives are evaluated at the same
    provisional state.

    The variational-gradient backend orders parameters as all gammas
    followed by all betas, so the final len(operator_pool) entries are
    the candidate mixer gradients.
    """
    if not operator_pool:
        return []

    scoring_layers = _build_pool_scoring_layers(
        layers,
        gamma0,
        operator_pool,
    )

    full_gradient = np.asarray(
        compute_variational_gradient(
            layers=scoring_layers,
            n_qubits=n_qubits,
            cost_coefficients=cost_coefficients,
            cost_pauli_words=cost_pauli_words,
        ),
        dtype=float,
    )

    number_of_candidates = len(operator_pool)

    if len(full_gradient) < number_of_candidates:
        raise RuntimeError(
            "Gradient vector is shorter than the operator pool."
        )

    candidate_gradients = full_gradient[-number_of_candidates:]

    results = []

    for entry, gradient in zip(
        operator_pool,
        candidate_gradients,
    ):
        gradient = float(gradient)

        results.append(
            {
                "name": entry["name"],
                "operator": entry["operator"],
                "pauli_word": entry["pauli_word"],
                "support": entry["support"],
                "is_global_mixer": entry["is_global_mixer"],
                "gradient": gradient,
                "abs_gradient": abs(gradient),
            }
        )

    # Adaptive selection consumes candidates in descending score order.
    # Canonical pool order is restored later before KaMIS graph construction.
    results.sort(
        key=lambda result: result["abs_gradient"],
        reverse=True,
    )

    return results
