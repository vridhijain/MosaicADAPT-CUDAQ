"""Analytic variational gradients for MosaicADAPT-QAOA.

Parameters are ordered as

    [gamma_1, ..., gamma_L, beta_1, ..., beta_B]

to match the Julia TetrisQAOA ansatz.

The gradient is evaluated with a reverse state/costate sweep. Finite
differences are used only in regression tests, not in production.
"""

import cudaq
import numpy as np

from ansatz_executor import flatten_layers, mosaic_ansatz_kernel


def _as_complex_vector(state):
    """Return a copied NumPy complex statevector."""
    return np.asarray(state, dtype=np.complex128).copy()


def _pauli_word_string(pauli_word, n_qubits):
    """Normalize a Pauli word to the project's qubit-index convention."""
    word = (
        pauli_word
        if isinstance(pauli_word, str)
        else str(pauli_word)
    )

    word = word.strip().replace(" ", "")

    if len(word) != n_qubits:
        raise ValueError(
            "Pauli-word length mismatch: "
            f"expected {n_qubits}, received '{word}'."
        )

    if any(symbol not in "IXYZ" for symbol in word):
        raise ValueError(f"Unsupported Pauli word '{word}'.")

    return word


def apply_pauli_word(state, pauli_word, n_qubits):
    """Apply a Pauli word to a statevector.

    CUDA-Q uses qubit 0 as the least-significant statevector bit.
    """
    state = _as_complex_vector(state)

    if len(state) != 1 << n_qubits:
        raise ValueError(
            f"Statevector dimension does not match {n_qubits} qubits."
        )

    word = _pauli_word_string(pauli_word, n_qubits)
    result = np.zeros_like(state)

    for basis_index, amplitude in enumerate(state):
        target_index = basis_index
        phase = 1.0 + 0.0j

        for qubit_index, symbol in enumerate(word):
            if symbol == "I":
                continue

            bit = (basis_index >> qubit_index) & 1

            if symbol == "X":
                target_index ^= 1 << qubit_index

            elif symbol == "Y":
                target_index ^= 1 << qubit_index
                phase *= 1.0j if bit == 0 else -1.0j

            elif symbol == "Z" and bit == 1:
                phase *= -1.0

        result[target_index] += phase * amplitude

    return result


def evolve_pauli_word(state, pauli_word, theta, n_qubits):
    """Apply exp(-i theta P) for a Pauli word satisfying P^2 = I."""
    state = _as_complex_vector(state)
    pauli_state = apply_pauli_word(
        state,
        pauli_word,
        n_qubits,
    )

    return (
        np.cos(theta) * state
        - 1.0j * np.sin(theta) * pauli_state
    )


def _single_x_word(n_qubits, qubit_index):
    """Return the Pauli word for X acting on one qubit."""
    symbols = ["I"] * n_qubits
    symbols[qubit_index] = "X"
    return "".join(symbols)


def apply_sum_x_generator(state, n_qubits):
    """Apply SUM_X = X_0 + ... + X_(n-1) to a statevector."""
    state = _as_complex_vector(state)
    result = np.zeros_like(state)

    for qubit_index in range(n_qubits):
        result += apply_pauli_word(
            state,
            _single_x_word(n_qubits, qubit_index),
            n_qubits,
        )

    return result


def evolve_sum_x(state, theta, n_qubits):
    """Apply exp(-i theta SUM_X).

    The single-qubit X terms commute, so the evolution factorizes into
    one exp(-i theta X_i) operation per qubit.
    """
    result = _as_complex_vector(state)

    for qubit_index in range(n_qubits):
        result = evolve_pauli_word(
            result,
            _single_x_word(n_qubits, qubit_index),
            theta,
            n_qubits,
        )

    return result


def apply_cost_generator(
    state,
    cost_coefficients,
    cost_pauli_words,
    n_qubits,
):
    """Apply H_C to a statevector using its Pauli decomposition."""
    state = _as_complex_vector(state)
    result = np.zeros_like(state)

    for coefficient, pauli_word in zip(
        cost_coefficients,
        cost_pauli_words,
    ):
        result += float(coefficient) * apply_pauli_word(
            state,
            pauli_word,
            n_qubits,
        )

    return result


def evolve_cost(
    state,
    theta,
    cost_coefficients,
    cost_pauli_words,
    n_qubits,
):
    """Apply exp(-i theta H_C) for a diagonal QAOA cost Hamiltonian."""
    result = _as_complex_vector(state)

    for coefficient, pauli_word in zip(
        cost_coefficients,
        cost_pauli_words,
    ):
        word = _pauli_word_string(
            pauli_word,
            n_qubits,
        )

        if any(symbol not in "IZ" for symbol in word):
            raise ValueError(
                "QAOA cost evolution requires diagonal I/Z terms; "
                f"received '{word}'."
            )

        result = evolve_pauli_word(
            result,
            word,
            theta * float(coefficient),
            n_qubits,
        )

    return result


def _is_sum_x(operator_entry):
    """Return whether an operator-pool entry represents SUM_X."""
    return bool(
        operator_entry.get(
            "is_global_mixer",
            False,
        )
    )


def apply_mixer_generator(state, operator_entry, n_qubits):
    """Apply the generator associated with one mixer."""
    if _is_sum_x(operator_entry):
        return apply_sum_x_generator(
            state,
            n_qubits,
        )

    return apply_pauli_word(
        state,
        operator_entry["pauli_word"],
        n_qubits,
    )


def evolve_mixer(state, operator_entry, theta, n_qubits):
    """Apply exp(-i theta G) for one mixer generator."""
    if _is_sum_x(operator_entry):
        return evolve_sum_x(
            state,
            theta,
            n_qubits,
        )

    return evolve_pauli_word(
        state,
        operator_entry["pauli_word"],
        theta,
        n_qubits,
    )


def get_ansatz_statevector(
    layers,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
):
    """Return the final CUDA-Q statevector for the current ansatz."""
    (
        gamma_values,
        beta_values,
        layer_sizes,
        mixer_kinds,
        mixer_words,
    ) = flatten_layers(
        layers,
        n_qubits,
    )

    state = cudaq.get_state(
        mosaic_ansatz_kernel,
        n_qubits,
        gamma_values,
        beta_values,
        layer_sizes,
        cost_coefficients,
        cost_pauli_words,
        mixer_kinds,
        mixer_words,
    )

    return _as_complex_vector(state)


def simulate_layers_numpy(
    layers,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
):
    """Execute the ansatz with an independent NumPy statevector model."""
    dimension = 1 << n_qubits

    state = np.ones(
        dimension,
        dtype=np.complex128,
    ) / np.sqrt(dimension)

    for layer in layers:
        state = evolve_cost(
            state,
            float(layer["gamma"]),
            cost_coefficients,
            cost_pauli_words,
            n_qubits,
        )

        for operator_entry, beta in zip(
            layer["operators"],
            layer["betas"],
        ):
            state = evolve_mixer(
                state,
                operator_entry,
                float(beta),
                n_qubits,
            )

    return state


def compute_variational_gradient(
    layers,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
):
    """Compute the full analytic gradient of the variational ansatz.

    For an operation U(theta) = exp(-i theta G), the reverse sweep
    evaluates

        |sigma> = U(theta) (-i G) |psi_before>

    and

        dE/dtheta = 2 Re <sigma | lambda>,

    where |lambda> is the costate propagated backward from
    H_C |psi_final>.
    """
    number_of_layers = len(layers)
    number_of_betas = sum(
        len(layer["betas"])
        for layer in layers
    )
    number_of_parameters = number_of_layers + number_of_betas

    if number_of_parameters == 0:
        return np.zeros(0, dtype=float)

    psi = get_ansatz_statevector(
        layers=layers,
        n_qubits=n_qubits,
        cost_coefficients=cost_coefficients,
        cost_pauli_words=cost_pauli_words,
    )

    lambda_state = apply_cost_generator(
        psi,
        cost_coefficients,
        cost_pauli_words,
        n_qubits,
    )

    gradients = np.zeros(
        number_of_parameters,
        dtype=float,
    )

    # Store operations in forward circuit order while assigning
    # gradient indices in [all gammas, then all betas] order.
    operations = []
    beta_counter = 0

    for layer_index, layer in enumerate(layers):
        operations.append(
            {
                "kind": "cost",
                "theta": float(layer["gamma"]),
                "gradient_index": layer_index,
                "operator": None,
            }
        )

        for operator_entry, beta in zip(
            layer["operators"],
            layer["betas"],
        ):
            operations.append(
                {
                    "kind": "mixer",
                    "theta": float(beta),
                    "gradient_index": (
                        number_of_layers + beta_counter
                    ),
                    "operator": operator_entry,
                }
            )
            beta_counter += 1

    for operation in reversed(operations):
        theta = operation["theta"]
        gradient_index = operation["gradient_index"]

        if operation["kind"] == "cost":
            # Undo the gate so psi is the state immediately before it.
            psi = evolve_cost(
                psi,
                -theta,
                cost_coefficients,
                cost_pauli_words,
                n_qubits,
            )

            generator_state = apply_cost_generator(
                psi,
                cost_coefficients,
                cost_pauli_words,
                n_qubits,
            )

            sigma = evolve_cost(
                -1.0j * generator_state,
                theta,
                cost_coefficients,
                cost_pauli_words,
                n_qubits,
            )

            gradients[gradient_index] = 2.0 * np.real(
                np.vdot(
                    sigma,
                    lambda_state,
                )
            )

            lambda_state = evolve_cost(
                lambda_state,
                -theta,
                cost_coefficients,
                cost_pauli_words,
                n_qubits,
            )

        else:
            operator_entry = operation["operator"]

            psi = evolve_mixer(
                psi,
                operator_entry,
                -theta,
                n_qubits,
            )

            generator_state = apply_mixer_generator(
                psi,
                operator_entry,
                n_qubits,
            )

            sigma = evolve_mixer(
                -1.0j * generator_state,
                operator_entry,
                theta,
                n_qubits,
            )

            gradients[gradient_index] = 2.0 * np.real(
                np.vdot(
                    sigma,
                    lambda_state,
                )
            )

            lambda_state = evolve_mixer(
                lambda_state,
                operator_entry,
                -theta,
                n_qubits,
            )

    return gradients
