"""BFGS optimization for MosaicADAPT-QAOA.

The Julia reference uses BFGS with an explicitly supplied variational
gradient. This implementation uses SciPy BFGS with the same parameter
ordering and an analytic Jacobian.

SciPy and Optim.jl are different BFGS implementations, so their exact
iteration trajectories are not expected to match.
"""

import numpy as np
from scipy.optimize import minimize

from ansatz_executor import evaluate_ansatz
from optimization import pack_parameters, unpack_parameters
from variational_gradients import compute_variational_gradient


def make_bfgs_functions(
    template_layers,
    cost_hamiltonian,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
    objective_history=None,
    gradient_history=None,
    verbose=False,
):
    """Construct the energy objective and analytic gradient for BFGS."""

    def objective(parameters):
        layers = unpack_parameters(
            parameters,
            template_layers,
        )

        energy = float(
            evaluate_ansatz(
                layers=layers,
                cost_hamiltonian=cost_hamiltonian,
                n_qubits=n_qubits,
                cost_coefficients=cost_coefficients,
                cost_pauli_words=cost_pauli_words,
            )
        )

        if objective_history is not None:
            objective_history.append(
                {
                    "parameters": [
                        float(value)
                        for value in parameters
                    ],
                    "energy": energy,
                }
            )

        if verbose:
            parameter_values = [
                round(float(value), 6)
                for value in parameters
            ]
            print(
                f"f(x): {parameter_values} -> {energy:.12f}"
            )

        return energy

    def gradient(parameters):
        layers = unpack_parameters(
            parameters,
            template_layers,
        )

        gradient_vector = np.asarray(
            compute_variational_gradient(
                layers=layers,
                n_qubits=n_qubits,
                cost_coefficients=cost_coefficients,
                cost_pauli_words=cost_pauli_words,
            ),
            dtype=float,
        )

        if gradient_history is not None:
            gradient_history.append(
                {
                    "parameters": [
                        float(value)
                        for value in parameters
                    ],
                    "gradient": gradient_vector.copy(),
                    "gradient_inf_norm": float(
                        np.linalg.norm(
                            gradient_vector,
                            ord=np.inf,
                        )
                    ),
                }
            )

        if verbose:
            gradient_values = [
                round(float(value), 8)
                for value in gradient_vector
            ]
            print(f"g(x): {gradient_values}")

        return gradient_vector

    return objective, gradient


def optimize_ansatz_bfgs(
    layers,
    cost_hamiltonian,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
    gtol=1e-6,
    maxiter=None,
    verbose=False,
):
    """Optimize the complete ansatz with BFGS and an analytic gradient."""
    initial_parameters = np.asarray(
        pack_parameters(layers),
        dtype=float,
    )

    if len(initial_parameters) == 0:
        raise ValueError(
            "Cannot optimize an ansatz with no variational parameters."
        )

    objective_history = []
    gradient_history = []

    objective, gradient = make_bfgs_functions(
        template_layers=layers,
        cost_hamiltonian=cost_hamiltonian,
        n_qubits=n_qubits,
        cost_coefficients=cost_coefficients,
        cost_pauli_words=cost_pauli_words,
        objective_history=objective_history,
        gradient_history=gradient_history,
        verbose=verbose,
    )

    options = {
        "gtol": gtol,
    }

    if maxiter is not None:
        options["maxiter"] = int(maxiter)

    result = minimize(
        objective,
        initial_parameters,
        method="BFGS",
        jac=gradient,
        options=options,
    )

    optimized_layers = unpack_parameters(
        result.x,
        layers,
    )

    optimized_energy = float(result.fun)

    return (
        optimized_layers,
        optimized_energy,
        result,
        objective_history,
        gradient_history,
    )
