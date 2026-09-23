"""End-to-end scaling benchmark for MosaicADAPT-QAOA.

This benchmark measures the complete adaptive algorithm:

    candidate scoring
    -> thresholding
    -> incompatibility graph
    -> KaMIS MMWIS selection
    -> layer acceptance
    -> BFGS optimization
    -> repeat until stopping

In addition to total runtime, the benchmark records adaptive-layer and
optimizer work. Local macOS measurements characterize the CPU simulator
path rather than NVIDIA GPU performance.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hamiltonian import (
    build_maxcut_hamiltonian,
    extract_hamiltonian_terms,
)
from mosaic_adapt import run_mosaic_adapt_qaoa
from operator_pool import build_qaoa_double_pool


DEFAULT_SIZES = [6, 8, 10, 12]

EDGE_PROBABILITY = 0.5
GRAPH_SEED = 0

GAMMA0 = 0.001
GRADIENT_THRESHOLD = 1e-3

BFGS_GTOL = 1e-6
MAX_LAYERS = 10

KAMIS_SEED = 42

OUTPUT_PATH = Path(
    "results/scaling_full_mosaic.csv"
)

FIELDNAMES = [
    "n_qubits",
    "statevector_dimension",
    "edge_count",
    "pool_size",
    "accepted_layers",
    "adaptation_attempts",
    "stop_reason",
    "initial_energy",
    "final_energy",
    "total_bfgs_iterations",
    "total_function_evaluations",
    "total_gradient_evaluations",
    "max_final_gradient_inf_norm",
    "total_seconds",
]


def generate_graph_edges(
    n_qubits,
    probability,
    seed,
):
    """Generate the deterministic graph family used by both benchmarks."""
    rng = random.Random(seed)
    edges = []

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if rng.random() < probability:
                edges.append((i, j))

    return edges


def benchmark_one_size(n_qubits):
    """Run one complete MosaicADAPT-QAOA benchmark instance."""
    print(f"\n{'=' * 72}")
    print(f"FULL RUN: n = {n_qubits}")
    print("=" * 72)

    edges = generate_graph_edges(
        n_qubits=n_qubits,
        probability=EDGE_PROBABILITY,
        seed=GRAPH_SEED,
    )

    cost_hamiltonian = build_maxcut_hamiltonian(
        n_qubits,
        edges,
    )

    (
        cost_coefficients,
        cost_pauli_words,
    ) = extract_hamiltonian_terms(
        cost_hamiltonian,
        n_qubits,
    )

    operator_pool = build_qaoa_double_pool(
        n_qubits
    )

    print(f"edges = {len(edges)}")
    print(f"pool size = {len(operator_pool)}")

    start = time.perf_counter()

    result = run_mosaic_adapt_qaoa(
        operator_pool=operator_pool,
        cost_hamiltonian=cost_hamiltonian,
        n_qubits=n_qubits,
        cost_coefficients=cost_coefficients,
        cost_pauli_words=cost_pauli_words,
        gamma0=GAMMA0,
        gradient_threshold=GRADIENT_THRESHOLD,
        bfgs_gtol=BFGS_GTOL,
        max_layers=MAX_LAYERS,
        selection_backend="kamis",
        kamis_seed=KAMIS_SEED,
        verbose=False,
    )

    total_seconds = time.perf_counter() - start

    accepted_records = [
        record
        for record in result["history"]
        if record.get("layers_after") is not None
    ]

    total_bfgs_iterations = sum(
        record["optimizer_iterations"]
        for record in accepted_records
    )

    total_function_evaluations = sum(
        record["function_evaluations"]
        for record in accepted_records
    )

    total_gradient_evaluations = sum(
        record["gradient_evaluations"]
        for record in accepted_records
    )

    max_optimizer_gradient = max(
        (
            record[
                "final_parameter_gradient_inf_norm"
            ]
            for record in accepted_records
        ),
        default=0.0,
    )

    print(
        f"accepted layers = {result['number_of_layers']}"
    )
    print(
        f"adaptation attempts = "
        f"{result['adaptation_attempts']}"
    )
    print(
        f"stop reason = {result['stop_reason']}"
    )
    print(
        f"initial energy = "
        f"{result['initial_energy']:.12f}"
    )
    print(
        f"final energy = "
        f"{result['final_energy']:.12f}"
    )
    print(
        f"total BFGS iterations = "
        f"{total_bfgs_iterations}"
    )
    print(
        f"total runtime = {total_seconds:.6f} s"
    )

    return {
        "n_qubits": n_qubits,
        "statevector_dimension": 1 << n_qubits,
        "edge_count": len(edges),
        "pool_size": len(operator_pool),
        "accepted_layers": result["number_of_layers"],
        "adaptation_attempts": result[
            "adaptation_attempts"
        ],
        "stop_reason": result["stop_reason"],
        "initial_energy": result["initial_energy"],
        "final_energy": result["final_energy"],
        "total_bfgs_iterations": total_bfgs_iterations,
        "total_function_evaluations": (
            total_function_evaluations
        ),
        "total_gradient_evaluations": (
            total_gradient_evaluations
        ),
        "max_final_gradient_inf_norm": (
            max_optimizer_gradient
        ),
        "total_seconds": total_seconds,
    }


def write_results(rows, output_path):
    """Write benchmark rows to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark complete "
            "MosaicADAPT-QAOA scaling."
        )
    )

    parser.add_argument(
        "--sizes",
        default=",".join(
            str(value)
            for value in DEFAULT_SIZES
        ),
        help="Comma-separated qubit counts. Default: 6,8,10,12",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV output path.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    sizes = [
        int(value.strip())
        for value in args.sizes.split(",")
        if value.strip()
    ]

    print("\nFull MosaicADAPT-QAOA Scaling")
    print("==============================")
    print(f"sizes = {sizes}")
    print(f"gamma0 = {GAMMA0}")
    print(
        f"gradient threshold = {GRADIENT_THRESHOLD}"
    )
    print(f"BFGS gtol = {BFGS_GTOL}")
    print(f"maximum layers = {MAX_LAYERS}")
    print(f"graph probability = {EDGE_PROBABILITY}")
    print(f"graph seed = {GRAPH_SEED}")
    print(f"KaMIS seed = {KAMIS_SEED}")

    rows = []

    for n_qubits in sizes:
        rows.append(
            benchmark_one_size(n_qubits)
        )

        # Preserve completed runs if a later instance is interrupted.
        write_results(
            rows=rows,
            output_path=args.output,
        )

    print("\nFull scaling benchmark complete.")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
