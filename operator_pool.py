"""Operator-pool construction for MosaicADAPT-QAOA."""

from cudaq import spin


_PAULI_BUILDERS = {
    "X": spin.x,
    "Y": spin.y,
    "Z": spin.z,
}

_NONDAGONAL_PAIR_TYPES = [
    ("X", "X"),
    ("Y", "Y"),
    ("Y", "Z"),
    ("Z", "Y"),
    ("X", "Y"),
    ("Y", "X"),
    ("X", "Z"),
    ("Z", "X"),
]

_QAOA_DOUBLE_PAIR_TYPES = [
    ("X", "X"),
    ("Y", "Y"),
    ("Y", "Z"),
    ("Z", "Y"),
]


def make_pauli_word(n_qubits, pauli_map):
    """Construct an n-qubit Pauli word from a qubit-to-Pauli mapping."""
    word = ["I"] * n_qubits

    for qubit, pauli in pauli_map.items():
        word[qubit] = pauli

    return "".join(word)


def _global_x_mixer(n_qubits):
    """Construct SUM_X = X_0 + ... + X_(n-1)."""
    mixer = None

    for qubit in range(n_qubits):
        x_operator = spin.x(qubit)
        mixer = x_operator if mixer is None else mixer + x_operator

    return {
        "name": "SUM_X",
        "operator": mixer,
        "pauli_word": None,
        "support": set(range(n_qubits)),
        "is_global_mixer": True,
    }


def _single_pauli_entry(n_qubits, qubit, pauli):
    """Build metadata for one single-qubit Pauli generator."""
    return {
        "name": f"{pauli}{qubit}",
        "operator": _PAULI_BUILDERS[pauli](qubit),
        "pauli_word": make_pauli_word(
            n_qubits,
            {qubit: pauli},
        ),
        "support": {qubit},
        "is_global_mixer": False,
    }


def _two_qubit_pauli_entry(n_qubits, i, j, pauli_i, pauli_j):
    """Build metadata for one two-qubit Pauli-product generator."""
    return {
        "name": f"{pauli_i}{i}{pauli_j}{j}",
        "operator": (
            _PAULI_BUILDERS[pauli_i](i)
            * _PAULI_BUILDERS[pauli_j](j)
        ),
        "pauli_word": make_pauli_word(
            n_qubits,
            {
                i: pauli_i,
                j: pauli_j,
            },
        ),
        "support": {i, j},
        "is_global_mixer": False,
    }


def build_operator_pool(n_qubits):
    """Construct the non-diagonal pool used for the Max-E3-SAT tests.

    The pool contains the global QAOA mixer, single-qubit X and Y
    generators, and eight two-qubit generators for every pair i < j.
    """
    pool = [_global_x_mixer(n_qubits)]

    for qubit in range(n_qubits):
        pool.append(
            _single_pauli_entry(
                n_qubits,
                qubit,
                "X",
            )
        )
        pool.append(
            _single_pauli_entry(
                n_qubits,
                qubit,
                "Y",
            )
        )

    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            for pauli_i, pauli_j in _NONDAGONAL_PAIR_TYPES:
                pool.append(
                    _two_qubit_pauli_entry(
                        n_qubits,
                        i,
                        j,
                        pauli_i,
                        pauli_j,
                    )
                )

    return pool


def expected_pool_size(n_qubits):
    """Return the size of the non-diagonal operator pool."""
    return 4 * n_qubits**2 - 2 * n_qubits + 1


def build_qaoa_double_pool(n_qubits):
    """Construct the qaoa_double_pool used by the Julia reference.

    Ordering is preserved exactly:

    1. X_0, X_1, ..., X_(n-1)
    2. SUM_X
    3. XX, YY, YZ, ZY for each pair i < j

    The ordering is significant because it determines the graph-node
    ordering passed to KaMIS during adaptive selection.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")

    pool = [
        _single_pauli_entry(
            n_qubits,
            qubit,
            "X",
        )
        for qubit in range(n_qubits)
    ]

    pool.append(_global_x_mixer(n_qubits))

    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            for pauli_i, pauli_j in _QAOA_DOUBLE_PAIR_TYPES:
                pool.append(
                    _two_qubit_pauli_entry(
                        n_qubits,
                        i,
                        j,
                        pauli_i,
                        pauli_j,
                    )
                )

    return pool
