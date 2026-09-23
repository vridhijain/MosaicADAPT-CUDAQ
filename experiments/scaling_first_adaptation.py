"""First-adaptation scaling benchmark for MosaicADAPT-QAOA.

For each problem size, this benchmark measures the adaptive-selection
pipeline:

    MaxCut construction
    -> operator-pool construction
    -> analytic TetrisQAOA scoring
    -> gradient thresholding
    -> incompatibility-graph construction
    -> KaMIS MMWIS selection

BFGS optimization is intentionally excluded so that this experiment
isolates the cost of one adaptive-selection step.

On macOS, CUDA-Q uses the local CPU simulator path; these results should
therefore not be interpreted as GPU scaling measurements.
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
from kamis_mwis import solve_mwis_kamis
from mosaic_graph import build_incompatibility_graph
from operator_pool import build_qaoa_double_pool
from tetris_qaoa_scoring import compute_tetris_qaoa_pool_scores


DEFAULT_SIZES = [4, 6, 8, 10, 12, 14]

EDGE_PROBABILITY = 0.5
GRAPH_SEED = 0

GAMMA0 = 0.001
GRADIENT_THRESHOLD = 1e-3

KAMIS_SEED = 42

OUTPUT_PATH = Path(
    "results/scaling_first_adaptation.csv"
)

FIELDNAMES = [
    "n_qubits",
    "statevector_dimension",
    "edge_count",
    "pool_size",
    "eligible_count",
    "selected_count",
    "max_gradient",
    "selected_weight",
    "hamiltonian_seconds",
    "pool_seconds",
    "scoring_seconds",
    "threshold_seconds",
    "graph_seconds",
    "kamis_seconds",
    "total_seconds",
]


def generate_graph_edges(
    n_qubits,
    probability,
    seed,
):
    """Generate a deterministic Erdos-Renyi-style graph."""
    rng = random.Random(seed)
    edges = []

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if rng.random() < probability:
                edges.append((i, j))

    return edges


def threshold_in_pool_order(
    scoring_results,
    operator_pool,
    threshold,
):
    """Apply score >= threshold while preserving canonical pool order."""
    results_by_name = {
        entry["name"]: entry
        for entry in scoring_results
    }

    return [
        results_by_name[pool_entry["name"]]
        for pool_entry in operator_pool
        if (
            results_by_name[pool_entry["name"]]["abs_gradient"]
            >= threshold
        )
    ]


def benchmark_one_size(n_qubits):
    """Benchmark one first-adaptation instance."""
    print(f"\n{'=' * 72}")
    print(f"n = {n_qubits}")
    print("=" * 72)

    edges = generate_graph_edges(
        n_qubits=n_qubits,
        probability=EDGE_PROBABILITY,
        seed=GRAPH_SEED,
    )

    print(f"edges = {len(edges)}")

    start = time.perf_counter()

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

    hamiltonian_seconds = time.perf_counter() - start

    start = time.perf_counter()

    operator_pool = build_qaoa_double_pool(
        n_qubits
    )

    pool_seconds = time.perf_counter() - start
    pool_size = len(operator_pool)

    expected_pool_size = (
        2 * n_qubits**2
        - n_qubits
        + 1
    )

    if pool_size != expected_pool_size:
        raise RuntimeError(
            "Unexpected qaoa_double_pool size: "
            f"expected {expected_pool_size}, got {pool_size}."
        )

    print(f"pool size = {pool_size}")

    start = time.perf_counter()

    scoring_results = compute_tetris_qaoa_pool_scores(
        layers=[],
        gamma0=GAMMA0,
        operator_pool=operator_pool,
        n_qubits=n_qubits,
        cost_coefficients=cost_coefficients,
        cost_pauli_words=cost_pauli_words,
    )

    scoring_seconds = time.perf_counter() - start

    max_gradient = max(
        (
            entry["abs_gradient"]
            for entry in scoring_results
        ),
        default=0.0,
    )

    start = time.perf_counter()

    eligible = threshold_in_pool_order(
        scoring_results=scoring_results,
        operator_pool=operator_pool,
        threshold=GRADIENT_THRESHOLD,
    )

    threshold_seconds = time.perf_counter() - start
    eligible_count = len(eligible)

    start = time.perf_counter()

    incompatibility_graph = build_incompatibility_graph(
        eligible
    )

    graph_seconds = time.perf_counter() - start

    start = time.perf_counter()

    selected_names, selected_weight = solve_mwis_kamis(
        operator_results=eligible,
        graph=incompatibility_graph,
        seed=KAMIS_SEED,
        config="mmwis",
    )

    kamis_seconds = time.perf_counter() - start

    total_seconds = (
        hamiltonian_seconds
        + pool_seconds
        + scoring_seconds
        + threshold_seconds
        + graph_seconds
        + kamis_seconds
    )

    print(
        f"eligible = {eligible_count} / {pool_size}"
    )
    print(
        f"selected mixers = {len(selected_names)}"
    )
    print(
        f"maximum gradient = {max_gradient:.6e}"
    )

    print("\nTiming")
    print("------")
    print(
        f"Hamiltonian        {hamiltonian_seconds:10.6f} s"
    )
    print(
        f"Operator pool      {pool_seconds:10.6f} s"
    )
    print(
        f"Analytic scoring   {scoring_seconds:10.6f} s"
    )
    print(
        f"Threshold          {threshold_seconds:10.6f} s"
    )
    print(
        f"Graph build        {graph_seconds:10.6f} s"
    )
    print(
        f"KaMIS              {kamis_seconds:10.6f} s"
    )
    print(
        f"TOTAL              {total_seconds:10.6f} s"
    )

    return {
        "n_qubits": n_qubits,
        "statevector_dimension": 1 << n_qubits,
        "edge_count": len(edges),
        "pool_size": pool_size,
        "eligible_count": eligible_count,
        "selected_count": len(selected_names),
        "max_gradient": max_gradient,
        "selected_weight": selected_weight,
        "hamiltonian_seconds": hamiltonian_seconds,
        "pool_seconds": pool_seconds,
        "scoring_seconds": scoring_seconds,
        "threshold_seconds": threshold_seconds,
        "graph_seconds": graph_seconds,
        "kamis_seconds": kamis_seconds,
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
            "Benchmark first-adaptation "
            "MosaicADAPT-QAOA scaling."
        )
    )

    parser.add_argument(
        "--sizes",
        default=",".join(
            str(value)
            for value in DEFAULT_SIZES
        ),
        help=(
            "Comma-separated qubit counts. "
            "Default: 4,6,8,10,12,14"
        ),
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

    print("\nCUDA-Q First-Adaptation Scaling")
    print("================================")
    print(f"sizes = {sizes}")
    print(f"gamma0 = {GAMMA0}")
    print(
        f"gradient threshold = {GRADIENT_THRESHOLD}"
    )
    print(
        f"graph probability = {EDGE_PROBABILITY}"
    )
    print(f"graph seed = {GRAPH_SEED}")
    print(f"KaMIS seed = {KAMIS_SEED}")

    rows = []

    for n_qubits in sizes:
        rows.append(
            benchmark_one_size(n_qubits)
        )

        # Preserve completed runs if a larger instance is interrupted.
        write_results(
            rows=rows,
            output_path=args.output,
        )

    print("\nScaling benchmark complete.")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
