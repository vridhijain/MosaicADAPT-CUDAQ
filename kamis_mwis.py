"""KaMIS maximum-weight independent-set backend for MosaicADAPT-QAOA.

Eligible operators are graph vertices weighted by their absolute
candidate gradients. Operators with overlapping support are adjacent.

KaMIS requires integer vertex weights. The conversion follows the Julia
reference: scores are multiplied by 1e14, proportionally downscaled if
their total exceeds 2e9, rounded to integers, and positive scores that
round to zero are assigned weight 1.

The order of operator_results defines the METIS node ordering and must be
preserved.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path


SCALE_FACTOR = 1.0e14
MAX_SAFE_SUM = 2_000_000_000.0


def resolve_kamis_binary(kamis_path=None):
    """Locate the KaMIS mmwis executable.

    Search order is an explicit path, KAMIS_MMWIS, then the project's
    external/KaMIS/mmwis/deploy/mmwis binary.
    """
    candidates = []

    if kamis_path is not None:
        candidates.append(Path(kamis_path))

    environment_path = os.environ.get("KAMIS_MMWIS")
    if environment_path:
        candidates.append(Path(environment_path))

    project_root = Path(__file__).resolve().parent
    candidates.append(
        project_root
        / "external"
        / "KaMIS"
        / "mmwis"
        / "deploy"
        / "mmwis"
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()

        if not candidate.is_file():
            continue

        if not os.access(candidate, os.X_OK):
            raise PermissionError(
                f"KaMIS mmwis exists but is not executable: {candidate}"
            )

        return candidate

    searched = "\n".join(
        f"  - {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not locate KaMIS mmwis.\n"
        "Set KAMIS_MMWIS or supply kamis_path explicitly.\n"
        "Searched:\n"
        f"{searched}"
    )


def scale_weights_for_kamis(operator_results):
    """Convert floating-point candidate scores to KaMIS integer weights."""
    original_weights = [
        float(entry["abs_gradient"])
        for entry in operator_results
    ]

    scaled_weights = [
        weight * SCALE_FACTOR
        for weight in original_weights
    ]

    total_before_downscale = float(sum(scaled_weights))
    scale_down_factor = 1.0

    if total_before_downscale > MAX_SAFE_SUM:
        scale_down_factor = (
            MAX_SAFE_SUM / total_before_downscale
        )
        scaled_weights = [
            weight * scale_down_factor
            for weight in scaled_weights
        ]

    integer_weights = []

    for original_weight, scaled_weight in zip(
        original_weights,
        scaled_weights,
    ):
        integer_weight = int(round(scaled_weight))

        if integer_weight == 0 and original_weight > 0.0:
            integer_weight = 1

        integer_weights.append(integer_weight)

    weights_by_name = {
        entry["name"]: integer_weight
        for entry, integer_weight in zip(
            operator_results,
            integer_weights,
        )
    }

    metadata = {
        "scale_factor": SCALE_FACTOR,
        "max_safe_sum": MAX_SAFE_SUM,
        "total_before_downscale": total_before_downscale,
        "scale_down_factor": scale_down_factor,
        "total_integer_weight": int(sum(integer_weights)),
    }

    return weights_by_name, metadata


def write_metis_graph(
    operator_results,
    graph,
    integer_weights,
    output_file,
):
    """Write a node-weighted incompatibility graph in METIS format.

    operator_results order determines the 1-indexed METIS node IDs.
    """
    output_file = Path(output_file)

    names = [
        entry["name"]
        for entry in operator_results
    ]

    if len(set(names)) != len(names):
        raise ValueError("Operator names must be unique.")

    node_id = {
        name: index + 1
        for index, name in enumerate(names)
    }

    expected_names = set(names)
    graph_names = set(graph)

    if graph_names != expected_names:
        raise ValueError(
            "Graph/operator mismatch.\n"
            f"operators={sorted(expected_names)}\n"
            f"graph={sorted(graph_names)}"
        )

    for name in names:
        if name in graph[name]:
            raise ValueError(
                f"Self-loop found for operator {name}."
            )

        for neighbor in graph[name]:
            if neighbor not in graph:
                raise ValueError(
                    f"Unknown graph neighbor: {neighbor}"
                )

            if name not in graph[neighbor]:
                raise ValueError(
                    "Incompatibility graph is not symmetric: "
                    f"{name} -> {neighbor}"
                )

    total_adjacency_entries = sum(
        len(graph[name])
        for name in names
    )

    if total_adjacency_entries % 2 != 0:
        raise ValueError(
            "Graph contains an odd number of adjacency entries."
        )

    n_nodes = len(names)
    n_edges = total_adjacency_entries // 2

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(f"{n_nodes} {n_edges} 10\n")

        for name in names:
            if name not in integer_weights:
                raise ValueError(
                    f"Missing KaMIS weight for {name}."
                )

            weight = int(integer_weights[name])

            neighbor_ids = sorted(
                node_id[neighbor]
                for neighbor in graph[name]
            )

            if neighbor_ids:
                neighbors = " ".join(
                    str(index)
                    for index in neighbor_ids
                )
                handle.write(
                    f"{weight} {neighbors}\n"
                )
            else:
                handle.write(f"{weight}\n")

    return node_id


def run_kamis_mmwis(
    graph_file,
    output_file,
    kamis_path=None,
    seed=0,
    time_limit=0.0,
    config="mmwis",
):
    """Run the external KaMIS mmwis executable."""
    binary = resolve_kamis_binary(kamis_path)

    graph_file = Path(graph_file).resolve()
    output_file = Path(output_file).resolve()

    command = [
        str(binary),
        str(graph_file),
        f"--output={output_file}",
        f"--config={config}",
    ]

    if seed > 0:
        command.append(f"--seed={int(seed)}")

    if time_limit > 0:
        command.append(
            f"--time_limit={float(time_limit)}"
        )

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(
            "Failed to launch KaMIS mmwis."
        ) from error

    if completed.returncode != 0:
        raise RuntimeError(
            "KaMIS mmwis failed with exit code "
            f"{completed.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not output_file.is_file():
        raise RuntimeError(
            "KaMIS exited successfully but did not produce "
            f"the expected solution file:\n{output_file}"
        )

    return output_file


def parse_kamis_solution(solution_file, n_nodes):
    """Return zero-based node indices selected by KaMIS."""
    solution_file = Path(solution_file)

    if not solution_file.is_file():
        raise FileNotFoundError(
            f"KaMIS solution file not found: {solution_file}"
        )

    raw_lines = [
        line.strip()
        for line in solution_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(raw_lines) != n_nodes:
        raise ValueError(
            "KaMIS solution length mismatch: "
            f"got {len(raw_lines)} lines, expected {n_nodes}."
        )

    selected_indices = []

    for index, line in enumerate(raw_lines):
        try:
            value = int(line)
        except ValueError as error:
            raise ValueError(
                "Invalid KaMIS solution entry "
                f"at node {index + 1}: {line!r}"
            ) from error

        if value not in (0, 1):
            raise ValueError(
                "KaMIS solution entries must be 0 or 1; "
                f"got {value} at node {index + 1}."
            )

        if value == 1:
            selected_indices.append(index)

    return tuple(selected_indices)


def solve_mwis_kamis(
    operator_results,
    graph,
    kamis_path=None,
    seed=42,
    time_limit=0.0,
    config="mmwis",
    temp_dir=None,
    keep_files=False,
):
    """Solve the Mosaic incompatibility graph with KaMIS MMWIS.

    Returns selected operator names and their total original
    floating-point gradient weight.
    """
    if not operator_results:
        return (), 0.0

    if len(operator_results) == 1:
        only_entry = operator_results[0]

        return (
            (only_entry["name"],),
            float(only_entry["abs_gradient"]),
        )

    integer_weights, _ = scale_weights_for_kamis(
        operator_results
    )

    if temp_dir is not None:
        work_directory = Path(temp_dir)
        work_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return _solve_mwis_kamis_in_directory(
            operator_results=operator_results,
            graph=graph,
            integer_weights=integer_weights,
            work_directory=work_directory,
            kamis_path=kamis_path,
            seed=seed,
            time_limit=time_limit,
            config=config,
        )

    if keep_files:
        work_directory = Path(
            tempfile.mkdtemp(
                prefix="mosaic_kamis_"
            )
        )

        return _solve_mwis_kamis_in_directory(
            operator_results=operator_results,
            graph=graph,
            integer_weights=integer_weights,
            work_directory=work_directory,
            kamis_path=kamis_path,
            seed=seed,
            time_limit=time_limit,
            config=config,
        )

    with tempfile.TemporaryDirectory(
        prefix="mosaic_kamis_"
    ) as directory:
        return _solve_mwis_kamis_in_directory(
            operator_results=operator_results,
            graph=graph,
            integer_weights=integer_weights,
            work_directory=Path(directory),
            kamis_path=kamis_path,
            seed=seed,
            time_limit=time_limit,
            config=config,
        )


def _solve_mwis_kamis_in_directory(
    operator_results,
    graph,
    integer_weights,
    work_directory,
    kamis_path,
    seed,
    time_limit,
    config,
):
    """Run one KaMIS solve inside an existing working directory."""
    unique_id = uuid.uuid4().hex

    graph_file = (
        work_directory
        / f"operators_{unique_id}.metis"
    )
    solution_file = (
        work_directory
        / f"operators_{unique_id}_solution.txt"
    )

    write_metis_graph(
        operator_results=operator_results,
        graph=graph,
        integer_weights=integer_weights,
        output_file=graph_file,
    )

    run_kamis_mmwis(
        graph_file=graph_file,
        output_file=solution_file,
        kamis_path=kamis_path,
        seed=seed,
        time_limit=time_limit,
        config=config,
    )

    selected_indices = parse_kamis_solution(
        solution_file=solution_file,
        n_nodes=len(operator_results),
    )

    selected_names = tuple(
        operator_results[index]["name"]
        for index in selected_indices
    )

    selected_weight = float(
        sum(
            operator_results[index]["abs_gradient"]
            for index in selected_indices
        )
    )

    return selected_names, selected_weight
