"""Cost-Hamiltonian utilities for Max-E3-SAT and MaxCut."""

from cudaq import spin


def literal_penalty(var_index, is_negated):
    """Return the projector that is 1 when a Boolean literal is false.

    A positive literal x_i uses (I + Z_i) / 2, while a negated literal
    uses (I - Z_i) / 2.
    """
    identity = spin.i(var_index)
    z_operator = spin.z(var_index)

    if is_negated:
        return 0.5 * (identity - z_operator)

    return 0.5 * (identity + z_operator)


def build_clause_hamiltonian(clause):
    """Construct the penalty Hamiltonian for one OR clause."""
    clause_hamiltonian = None

    for var_index, is_negated in clause:
        penalty = literal_penalty(var_index, is_negated)

        if clause_hamiltonian is None:
            clause_hamiltonian = penalty
        else:
            clause_hamiltonian *= penalty

    return clause_hamiltonian


def build_cost_hamiltonian(formula):
    """Construct a Max-E3-SAT Hamiltonian counting violated clauses.

    For a computational-basis assignment |x>, the expectation
    <x|H_C|x> equals the number of violated clauses.
    """
    cost_hamiltonian = None

    for clause in formula:
        clause_hamiltonian = build_clause_hamiltonian(clause)

        if cost_hamiltonian is None:
            cost_hamiltonian = clause_hamiltonian
        else:
            cost_hamiltonian += clause_hamiltonian

    return cost_hamiltonian


def count_violated_clauses(bits, formula):
    """Count violated clauses for a classical Boolean assignment."""
    violations = 0

    for clause in formula:
        clause_satisfied = False

        for var_index, is_negated in clause:
            value = bits[var_index]
            literal_value = 1 - value if is_negated else value

            if literal_value == 1:
                clause_satisfied = True
                break

        if not clause_satisfied:
            violations += 1

    return violations


def extract_hamiltonian_terms(hamiltonian, n_qubits, tolerance=1e-12):
    """Extract nonzero coefficients and Pauli words from a spin Hamiltonian."""
    coefficients = []
    pauli_words = []

    for term in hamiltonian:
        coefficient = term.evaluate_coefficient()
        pauli_word = term.get_pauli_word(n_qubits)

        if abs(coefficient.real) > tolerance:
            coefficients.append(coefficient.real)
            pauli_words.append(pauli_word)

    return coefficients, pauli_words


def _parse_edge(edge):
    """Return (i, j, weight) for an unweighted or weighted MaxCut edge."""
    if len(edge) == 2:
        i, j = edge
        return i, j, 1.0

    if len(edge) == 3:
        i, j, weight = edge
        return i, j, float(weight)

    raise ValueError(
        "Each MaxCut edge must be (i, j) or (i, j, weight)."
    )


def build_maxcut_hamiltonian(n_qubits, edges):
    """Construct the MaxCut Hamiltonian used by the Julia reference.

    Each weighted edge contributes

        (w_ij / 2) * (Z_i Z_j - I),

    so a cut edge contributes -w_ij and an uncut edge contributes 0.
    Minimizing the Hamiltonian therefore maximizes the cut weight.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")

    cost_hamiltonian = None

    for edge in edges:
        i, j, weight = _parse_edge(edge)

        if i < 0 or j < 0 or i >= n_qubits or j >= n_qubits:
            raise ValueError(
                f"MaxCut edge contains an out-of-range vertex: {edge}"
            )

        if i == j:
            raise ValueError(
                f"MaxCut self-loops are not supported: {edge}"
            )

        # spin.i(0) represents the global identity term.
        edge_hamiltonian = (
            0.5 * weight * spin.z(i) * spin.z(j)
            - 0.5 * weight * spin.i(0)
        )

        if cost_hamiltonian is None:
            cost_hamiltonian = edge_hamiltonian
        else:
            cost_hamiltonian += edge_hamiltonian

    if cost_hamiltonian is None:
        # Preserve a valid SpinOperator for an empty graph.
        cost_hamiltonian = 0.0 * spin.i(0)

    return cost_hamiltonian


def maxcut_size(bits, edges):
    """Return the weighted cut size of a classical bit assignment."""
    total = 0.0

    for edge in edges:
        i, j, weight = _parse_edge(edge)

        if bits[i] != bits[j]:
            total += weight

    return total


def maxcut_energy(bits, edges):
    """Return the classical energy corresponding to the MaxCut Hamiltonian."""
    return -maxcut_size(bits, edges)
