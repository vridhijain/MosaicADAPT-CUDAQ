"""End-to-end MosaicADAPT-QAOA driver.

Each adaptation performs:

    analytic TetrisQAOA candidate scoring
    -> gradient thresholding
    -> incompatibility-graph construction
    -> MWIS operator selection
    -> layer acceptance
    -> full BFGS re-optimization

The brute-force MWIS backend is intended for small-instance validation.
KaMIS provides the scalable selection backend used by the reference
implementation.
"""

import numpy as np

from ansatz_executor import evaluate_ansatz
from bfgs_optimizer import optimize_ansatz_bfgs
from kamis_mwis import solve_mwis_kamis
from mosaic_graph import (
    build_incompatibility_graph,
    get_selected_operator_entries,
    is_independent_set,
    solve_mwis_bruteforce,
)
from tetris_qaoa_scoring import compute_tetris_qaoa_pool_scores
from variational_gradients import compute_variational_gradient


def _maximum_pool_gradient(gradient_results):
    """Return the maximum absolute candidate gradient."""
    if not gradient_results:
        return 0.0

    return max(
        entry["abs_gradient"]
        for entry in gradient_results
    )


def _filter_by_gradient_threshold(
    gradient_results,
    gradient_threshold,
):
    """Retain candidates satisfying the reference score >= threshold rule."""
    return [
        entry
        for entry in gradient_results
        if entry["abs_gradient"] >= gradient_threshold
    ]


def _restore_operator_pool_order(eligible, operator_pool):
    """Restore canonical pool order before MWIS graph construction.

    Candidate scoring is sorted by decreasing absolute gradient for
    diagnostics. The Julia reference instead builds the KaMIS graph in
    original operator-pool order, so that ordering is restored here.
    """
    eligible_by_name = {
        entry["name"]: entry
        for entry in eligible
    }

    restored = [
        eligible_by_name[pool_entry["name"]]
        for pool_entry in operator_pool
        if pool_entry["name"] in eligible_by_name
    ]

    if len(restored) != len(eligible):
        raise RuntimeError(
            "Failed to restore canonical operator-pool order."
        )

    return restored


def _append_mosaic_layer(
    layers,
    selected_entries,
    gamma0,
):
    """Return a new ansatz with one zero-beta Mosaic layer appended."""
    new_layers = list(layers)

    new_layers.append(
        {
            "gamma": float(gamma0),
            "operators": list(selected_entries),
            "betas": [0.0] * len(selected_entries),
        }
    )

    return new_layers


def _select_mwis(
    eligible,
    graph,
    selection_backend,
    kamis_path=None,
    kamis_seed=42,
    kamis_time_limit=0.0,
):
    """Run the requested MWIS backend."""
    if selection_backend == "bruteforce":
        return solve_mwis_bruteforce(
            eligible,
            graph,
        )

    if selection_backend == "kamis":
        return solve_mwis_kamis(
            operator_results=eligible,
            graph=graph,
            kamis_path=kamis_path,
            seed=kamis_seed,
            time_limit=kamis_time_limit,
            config="mmwis",
        )

    raise ValueError(
        "Unknown selection_backend "
        f"{selection_backend!r}. "
        "Expected 'bruteforce' or 'kamis'."
    )


def _stop_record(
    adaptation_index,
    layers,
    max_pool_gradient,
    eligible_count,
    selected_names,
    selected_weight,
    selection_backend,
    stop_reason,
):
    """Construct one stopping-event history entry."""
    return {
        "adaptation_attempt": adaptation_index,
        "layers_before": len(layers),
        "max_pool_gradient": float(max_pool_gradient),
        "eligible_count": eligible_count,
        "selected_names": list(selected_names),
        "selected_weight": float(selected_weight),
        "selection_backend": selection_backend,
        "stop_reason": stop_reason,
    }


def run_mosaic_adapt_qaoa(
    operator_pool,
    cost_hamiltonian,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
    gamma0,
    gradient_threshold=1e-3,
    bfgs_gtol=1e-6,
    max_layers=10,
    selection_backend="bruteforce",
    kamis_path=None,
    kamis_seed=42,
    kamis_time_limit=0.0,
    verbose=True,
):
    """Run MosaicADAPT-QAOA until a reference stopping condition fires.

    gamma0 is the provisional cost angle for candidate scoring and the
    initial gamma assigned to each accepted layer. Newly selected mixer
    angles begin at zero.

    ScoreStopper uses a strict max_score < gradient_threshold condition.
    LayerStopper is evaluated before another layer is accepted.
    """
    if selection_backend not in {"bruteforce", "kamis"}:
        raise ValueError(
            "selection_backend must be 'bruteforce' or 'kamis'."
        )

    layers = []
    history = []

    machine_epsilon = np.finfo(float).eps
    adaptation_index = 0
    stop_reason = None

    initial_energy = float(
        evaluate_ansatz(
            layers=layers,
            cost_hamiltonian=cost_hamiltonian,
            n_qubits=n_qubits,
            cost_coefficients=cost_coefficients,
            cost_pauli_words=cost_pauli_words,
        )
    )

    if verbose:
        print("\nMosaicADAPT-QAOA")
        print(f"Initial energy = {initial_energy}")
        print(f"gamma0 = {gamma0}")
        print(f"gradient threshold = {gradient_threshold}")
        print(f"BFGS gtol = {bfgs_gtol}")
        print(f"LayerStopper = {max_layers}")
        print(f"MWIS backend = {selection_backend}")

        if selection_backend == "kamis":
            print(f"KaMIS seed = {kamis_seed}")

    while True:
        adaptation_index += 1

        if verbose:
            print(
                f"\nAdaptation attempt {adaptation_index} "
                f"(accepted layers: {len(layers)})"
            )

        # Score every candidate after a provisional gamma0 cost layer.
        gradient_results = compute_tetris_qaoa_pool_scores(
            layers=layers,
            gamma0=gamma0,
            operator_pool=operator_pool,
            n_qubits=n_qubits,
            cost_coefficients=cost_coefficients,
            cost_pauli_words=cost_pauli_words,
        )

        max_pool_gradient = _maximum_pool_gradient(
            gradient_results
        )

        if verbose:
            print(
                f"Maximum pool gradient = {max_pool_gradient}"
            )

        # Internal TetrisQAOA convergence check.
        all_machine_zero = all(
            entry["abs_gradient"] < machine_epsilon
            for entry in gradient_results
        )

        if all_machine_zero:
            stop_reason = "AllGradientsHitMachineEpsilon"

            history.append(
                _stop_record(
                    adaptation_index=adaptation_index,
                    layers=layers,
                    max_pool_gradient=max_pool_gradient,
                    eligible_count=0,
                    selected_names=(),
                    selected_weight=0.0,
                    selection_backend=selection_backend,
                    stop_reason=stop_reason,
                )
            )

            if verbose:
                print(
                    "Stopping: all pool gradients are below "
                    "machine epsilon."
                )

            break

        eligible = _filter_by_gradient_threshold(
            gradient_results,
            gradient_threshold,
        )

        # KaMIS node order follows the canonical operator-pool order,
        # not the diagnostic score ordering.
        eligible = _restore_operator_pool_order(
            eligible,
            operator_pool,
        )

        if verbose:
            print(
                f"Eligible operators = "
                f"{len(eligible)} / {len(gradient_results)}"
            )

        graph = build_incompatibility_graph(eligible)

        selected_names, selected_weight = _select_mwis(
            eligible=eligible,
            graph=graph,
            selection_backend=selection_backend,
            kamis_path=kamis_path,
            kamis_seed=kamis_seed,
            kamis_time_limit=kamis_time_limit,
        )

        selected_entries = get_selected_operator_entries(
            selected_names,
            eligible,
        )

        if not is_independent_set(
            selected_names,
            graph,
        ):
            raise RuntimeError(
                "MWIS backend returned an incompatible operator set."
            )

        if verbose:
            print(f"MWIS selected = {list(selected_names)}")
            print(f"MWIS total weight = {selected_weight}")

        # Reference callbacks are evaluated after selection and before
        # modifying the ansatz.
        if max_pool_gradient < gradient_threshold:
            stop_reason = "ScoreStopper"

            history.append(
                _stop_record(
                    adaptation_index=adaptation_index,
                    layers=layers,
                    max_pool_gradient=max_pool_gradient,
                    eligible_count=len(eligible),
                    selected_names=selected_names,
                    selected_weight=selected_weight,
                    selection_backend=selection_backend,
                    stop_reason=stop_reason,
                )
            )

            if verbose:
                print("Stopping: ScoreStopper.")

            break

        if len(layers) >= max_layers:
            stop_reason = "LayerStopper"

            history.append(
                _stop_record(
                    adaptation_index=adaptation_index,
                    layers=layers,
                    max_pool_gradient=max_pool_gradient,
                    eligible_count=len(eligible),
                    selected_names=selected_names,
                    selected_weight=selected_weight,
                    selection_backend=selection_backend,
                    stop_reason=stop_reason,
                )
            )

            if verbose:
                print("Stopping: LayerStopper.")

            break

        if not selected_entries:
            raise RuntimeError(
                "MOSAIC selection returned no operators even though "
                "ScoreStopper did not converge."
            )

        layers_before_count = len(layers)

        layers = _append_mosaic_layer(
            layers=layers,
            selected_entries=selected_entries,
            gamma0=gamma0,
        )

        if verbose:
            print(f"Accepted QAOA layer {len(layers)}")
            print(f"Initial gamma = {gamma0}")
            print(f"Initial betas = {layers[-1]['betas']}")

        (
            optimized_layers,
            optimized_energy,
            optimizer_result,
            _objective_history,
            _gradient_history,
        ) = optimize_ansatz_bfgs(
            layers=layers,
            cost_hamiltonian=cost_hamiltonian,
            n_qubits=n_qubits,
            cost_coefficients=cost_coefficients,
            cost_pauli_words=cost_pauli_words,
            gtol=bfgs_gtol,
            maxiter=None,
            verbose=False,
        )

        layers = optimized_layers

        final_parameter_gradient = compute_variational_gradient(
            layers=layers,
            n_qubits=n_qubits,
            cost_coefficients=cost_coefficients,
            cost_pauli_words=cost_pauli_words,
        )

        final_parameter_gradient_inf_norm = float(
            np.linalg.norm(
                final_parameter_gradient,
                ord=np.inf,
            )
        )

        history.append(
            {
                "adaptation_attempt": adaptation_index,
                "layers_before": layers_before_count,
                "layers_after": len(layers),
                "max_pool_gradient": float(max_pool_gradient),
                "eligible_count": len(eligible),
                "selected_names": list(selected_names),
                "selected_weight": float(selected_weight),
                "selection_backend": selection_backend,
                "optimized_energy": float(optimized_energy),
                "optimizer_success": bool(
                    optimizer_result.success
                ),
                "optimizer_message": str(
                    optimizer_result.message
                ),
                "optimizer_iterations": int(
                    optimizer_result.nit
                ),
                "function_evaluations": int(
                    optimizer_result.nfev
                ),
                "gradient_evaluations": int(
                    optimizer_result.njev
                ),
                "final_parameter_gradient_inf_norm": (
                    final_parameter_gradient_inf_norm
                ),
                "stop_reason": None,
            }
        )

        if verbose:
            print(f"Optimized energy = {optimized_energy}")
            print(f"BFGS success = {optimizer_result.success}")
            print(f"BFGS iterations = {optimizer_result.nit}")
            print(
                "Final parameter ||gradient||_inf = "
                f"{final_parameter_gradient_inf_norm}"
            )

        if not optimizer_result.success:
            raise RuntimeError(
                "BFGS failed during MosaicADAPT-QAOA: "
                f"{optimizer_result.message}"
            )

    final_energy = float(
        evaluate_ansatz(
            layers=layers,
            cost_hamiltonian=cost_hamiltonian,
            n_qubits=n_qubits,
            cost_coefficients=cost_coefficients,
            cost_pauli_words=cost_pauli_words,
        )
    )

    return {
        "layers": layers,
        "number_of_layers": len(layers),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "stop_reason": stop_reason,
        "adaptation_attempts": adaptation_index,
        "selection_backend": selection_backend,
        "history": history,
    }
