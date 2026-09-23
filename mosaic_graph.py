"""Mosaic incompatibility-graph and exact MWIS utilities."""

from itertools import combinations


def build_incompatibility_graph(operator_results):
    """Build the graph connecting candidates with overlapping support."""
    graph = {
        entry["name"]: set()
        for entry in operator_results
    }

    for operator_i, operator_j in combinations(
        operator_results,
        2,
    ):
        if operator_i["support"].intersection(operator_j["support"]):
            name_i = operator_i["name"]
            name_j = operator_j["name"]

            graph[name_i].add(name_j)
            graph[name_j].add(name_i)

    return graph


def is_independent_set(candidate_set, graph):
    """Return whether the selected operators contain no graph edge."""
    for operator_i, operator_j in combinations(
        candidate_set,
        2,
    ):
        if operator_j in graph[operator_i]:
            return False

    return True


def solve_mwis_bruteforce(operator_results, graph):
    """Solve MWIS exactly by enumerating all candidate subsets.

    This implementation is intended as a correctness oracle for small
    instances, not as the scalable MosaicADAPT-QAOA selection backend.
    """
    weights = {
        entry["name"]: entry["abs_gradient"]
        for entry in operator_results
    }
    operator_names = list(weights)

    best_set = ()
    best_weight = 0.0

    for subset_size in range(len(operator_names) + 1):
        for candidate_set in combinations(
            operator_names,
            subset_size,
        ):
            if not is_independent_set(
                candidate_set,
                graph,
            ):
                continue

            total_weight = sum(
                weights[name]
                for name in candidate_set
            )

            if total_weight > best_weight:
                best_set = candidate_set
                best_weight = total_weight

    return best_set, best_weight


def get_selected_operator_entries(
    selected_names,
    operator_results,
):
    """Return complete metadata for the selected operator names."""
    entries_by_name = {
        entry["name"]: entry
        for entry in operator_results
    }

    return [
        entries_by_name[name]
        for name in selected_names
    ]
