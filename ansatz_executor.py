"""CUDA-Q execution utilities for MosaicADAPT-QAOA ansätze."""

import cudaq


PAULI_GENERATOR = 0
GLOBAL_SUM_X_GENERATOR = 1


@cudaq.kernel
def mosaic_ansatz_kernel(
    n_qubits: int,
    gamma_values: list[float],
    beta_values: list[float],
    layer_sizes: list[int],
    coefficients: list[float],
    cost_words: list[cudaq.pauli_word],
    mixer_kinds: list[int],
    mixer_words: list[cudaq.pauli_word],
):
    q = cudaq.qvector(n_qubits)

    for i in range(n_qubits):
        h(q[i])

    mixer_index = 0

    for layer_index in range(len(gamma_values)):
        # CUDA-Q's exp_pauli convention requires the negative angle
        # to implement exp(-i gamma H).
        for term_index in range(len(coefficients)):
            exp_pauli(
                -gamma_values[layer_index] * coefficients[term_index],
                q,
                cost_words[term_index],
            )

        for _ in range(layer_sizes[layer_index]):
            if mixer_kinds[mixer_index] == PAULI_GENERATOR:
                exp_pauli(
                    -beta_values[mixer_index],
                    q,
                    mixer_words[mixer_index],
                )
            else:
                # exp(-i beta SUM_i X_i) factorizes because the X_i commute.
                for qubit_index in range(n_qubits):
                    rx(
                        2.0 * beta_values[mixer_index],
                        q[qubit_index],
                    )

            mixer_index += 1


@cudaq.kernel
def mosaic_candidate_kernel(
    n_qubits: int,
    gamma_values: list[float],
    beta_values: list[float],
    layer_sizes: list[int],
    coefficients: list[float],
    cost_words: list[cudaq.pauli_word],
    mixer_kinds: list[int],
    mixer_words: list[cudaq.pauli_word],
    candidate_kind: int,
    candidate_word: cudaq.pauli_word,
    candidate_alpha: float,
):
    q = cudaq.qvector(n_qubits)

    for i in range(n_qubits):
        h(q[i])

    mixer_index = 0

    for layer_index in range(len(gamma_values)):
        for term_index in range(len(coefficients)):
            exp_pauli(
                -gamma_values[layer_index] * coefficients[term_index],
                q,
                cost_words[term_index],
            )

        for _ in range(layer_sizes[layer_index]):
            if mixer_kinds[mixer_index] == PAULI_GENERATOR:
                exp_pauli(
                    -beta_values[mixer_index],
                    q,
                    mixer_words[mixer_index],
                )
            else:
                for qubit_index in range(n_qubits):
                    rx(
                        2.0 * beta_values[mixer_index],
                        q[qubit_index],
                    )

            mixer_index += 1

    # The candidate is temporary: it is evaluated but not added to layers.
    if candidate_kind == PAULI_GENERATOR:
        exp_pauli(
            -candidate_alpha,
            q,
            candidate_word,
        )
    else:
        for qubit_index in range(n_qubits):
            rx(
                2.0 * candidate_alpha,
                q[qubit_index],
            )


def operator_to_executor_metadata(entry, n_qubits):
    """Convert an operator-pool entry to CUDA-Q kernel metadata.

    SUM_X uses an identity Pauli word as a placeholder so that
    mixer_words remains aligned with mixer_kinds. The placeholder is
    never applied; SUM_X is executed using RX rotations.
    """
    if entry["is_global_mixer"]:
        return (
            GLOBAL_SUM_X_GENERATOR,
            cudaq.pauli_word("I" * n_qubits),
        )

    pauli_word = entry["pauli_word"]

    if pauli_word is None:
        raise ValueError(
            f"Operator {entry['name']} is not marked as SUM_X "
            "but has no Pauli-word representation."
        )

    if len(str(pauli_word)) != n_qubits:
        raise ValueError(
            f"Operator {entry['name']} has Pauli word {pauli_word}, "
            f"which does not match the {n_qubits}-qubit register."
        )

    return (
        PAULI_GENERATOR,
        cudaq.pauli_word(pauli_word),
    )


def flatten_layers(layers, n_qubits):
    """Flatten structured adaptive layers into CUDA-Q kernel arguments."""
    gamma_values = []
    beta_values = []
    layer_sizes = []
    mixer_kinds = []
    mixer_words = []

    for layer_index, layer in enumerate(layers):
        if "gamma" not in layer:
            raise ValueError(f"Layer {layer_index} has no gamma value.")

        if "operators" not in layer:
            raise ValueError(f"Layer {layer_index} has no operators.")

        if "betas" not in layer:
            raise ValueError(f"Layer {layer_index} has no beta values.")

        operators = layer["operators"]
        layer_betas = layer["betas"]

        if len(operators) != len(layer_betas):
            raise ValueError(
                f"Layer {layer_index} contains {len(operators)} operators "
                f"but {len(layer_betas)} beta parameters."
            )

        gamma_values.append(float(layer["gamma"]))
        layer_sizes.append(len(operators))

        for operator_entry, beta in zip(operators, layer_betas):
            kind, word = operator_to_executor_metadata(
                operator_entry,
                n_qubits,
            )

            beta_values.append(float(beta))
            mixer_kinds.append(kind)
            mixer_words.append(word)

    return (
        gamma_values,
        beta_values,
        layer_sizes,
        mixer_kinds,
        mixer_words,
    )


def evaluate_ansatz(
    layers,
    cost_hamiltonian,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
):
    """Return <H_C> for an arbitrary-depth MosaicADAPT-QAOA ansatz."""
    (
        gamma_values,
        beta_values,
        layer_sizes,
        mixer_kinds,
        mixer_words,
    ) = flatten_layers(layers, n_qubits)

    return cudaq.observe(
        mosaic_ansatz_kernel,
        cost_hamiltonian,
        n_qubits,
        gamma_values,
        beta_values,
        layer_sizes,
        cost_coefficients,
        cost_pauli_words,
        mixer_kinds,
        mixer_words,
    ).expectation()


def evaluate_with_candidate(
    layers,
    candidate,
    candidate_alpha,
    cost_hamiltonian,
    n_qubits,
    cost_coefficients,
    cost_pauli_words,
):
    """Evaluate <H_C> after temporarily applying one candidate mixer."""
    (
        gamma_values,
        beta_values,
        layer_sizes,
        mixer_kinds,
        mixer_words,
    ) = flatten_layers(layers, n_qubits)

    candidate_kind, candidate_word = operator_to_executor_metadata(
        candidate,
        n_qubits,
    )

    return cudaq.observe(
        mosaic_candidate_kernel,
        cost_hamiltonian,
        n_qubits,
        gamma_values,
        beta_values,
        layer_sizes,
        cost_coefficients,
        cost_pauli_words,
        mixer_kinds,
        mixer_words,
        candidate_kind,
        candidate_word,
        float(candidate_alpha),
    ).expectation()
