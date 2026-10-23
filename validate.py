"""Exhaustive, independently checked validation of the research prototype."""

import json
import platform
import random
import time
from pathlib import Path

import networkx as nx
from matching_profiles import solve_profile, brute_force_profile
from odd_components import solve_odd_profile


def cactus_forest(g):
    for vertices in nx.biconnected_components(g):
        h = g.subgraph(vertices)
        if len(h) > 2 and (h.number_of_edges() != len(h) or
                           any(deg != 2 for _, deg in h.degree)):
            return False
    return True


def verify(g, result):
    expected = brute_force_profile(g, len(result.profile) - 1)
    assert result.profile == expected, (list(g.edges), result.profile, expected)
    for b, target in enumerate(expected):
        deleted = result.witness(b)
        assert len(deleted) <= b
        assert len(set(map(frozenset, deleted))) == len(deleted)
        assert all(g.has_edge(*e) for e in deleted)
        remaining = g.copy()
        remaining.remove_edges_from(deleted)
        actual = len(nx.max_weight_matching(remaining, maxcardinality=True))
        assert actual == target, (b, target, deleted, actual)
        if hasattr(result, "certificate"):
            barrier = result.certificate(b)["barrier"]
            assert len(set(barrier)) == len(barrier)
            residual = remaining.copy()
            residual.remove_nodes_from(barrier)
            odd = sum(len(c) % 2 for c in nx.connected_components(residual))
            assert len(g) + len(barrier) - odd == 2 * target
    return len(expected)


def run():
    start = time.perf_counter()
    counts = {"atlas_graphs_checked": 0, "atlas_cactus_forests": 0,
              "atlas_cacti_connected": 0, "atlas_graphs_rejected_width": 0,
              "budget_values_and_witnesses_checked": 0,
              "random_width2_graphs_checked": 0,
              "explicit_order_graphs_checked": 0,
              "truncated_profiles_checked": 0}
    max_states = 0
    for g in nx.graph_atlas_g():
        try:
            result = solve_profile(g)
        except ValueError as error:
            assert "partial 2-tree" in str(error)
            counts["atlas_graphs_rejected_width"] += 1
            continue
        counts["budget_values_and_witnesses_checked"] += verify(g, result)
        verify(g, solve_odd_profile(g))
        counts["atlas_graphs_checked"] += 1
        if cactus_forest(g):
            counts["atlas_cactus_forests"] += 1
            counts["atlas_cacti_connected"] += bool(len(g) and nx.is_connected(g))
        max_states = max(max_states, result.stats["max_signatures_per_budget"])

    rng = random.Random(20261019)
    for trial in range(80):
        n = rng.randrange(3, 12)
        g = nx.Graph([(0, 1)])
        for v in range(2, n):
            u, w = rng.choice(list(g.edges))
            g.add_edges_from([(u, v), (w, v)])
        g.remove_edges_from([e for e in list(g.edges) if rng.random() < .22])
        if g.number_of_edges() > 18:
            g.remove_edge(*rng.choice(list(g.edges)))
        g = nx.relabel_nodes(g, {v: ("vertex", v) for v in g})
        result = solve_profile(g)
        counts["budget_values_and_witnesses_checked"] += verify(g, result)
        verify(g, solve_odd_profile(g))
        counts["random_width2_graphs_checked"] += 1
        b = rng.randrange(g.number_of_edges() + 1)
        truncated = solve_profile(g, b)
        assert truncated.profile == result.profile[:b + 1]
        counts["budget_values_and_witnesses_checked"] += verify(g, truncated)
        verify(g, solve_odd_profile(g, b))
        counts["truncated_profiles_checked"] += 1
        max_states = max(max_states, result.stats["max_signatures_per_budget"])

    maximum_width_tested = 0
    for trial in range(50):
        n = rng.randrange(3, 8)
        g = nx.gnp_random_graph(n, rng.uniform(.25, .75), seed=rng)
        if g.number_of_edges() > 18:
            continue
        order = list(g)
        rng.shuffle(order)
        result = solve_profile(g, elimination_order=order)
        counts["budget_values_and_witnesses_checked"] += verify(g, result)
        verify(g, solve_odd_profile(g, elimination_order=order))
        counts["explicit_order_graphs_checked"] += 1
        maximum_width_tested = max(maximum_width_tested, result.width)

    invalid = [nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1)]),
               nx.Graph([(0, 0)]), nx.complete_graph(4)]
    for g in invalid:
        try:
            solve_profile(g)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid input was accepted")

    counts.update({"all_checks_passed": True,
                   "implementations_checked": ["normalized_profiles", "odd_components"],
                   "counts_apply_to_each_implementation": True,
                   "maximum_width_explicit_order": maximum_width_tested,
                   "max_observed_states_per_budget_width2": max_states,
                   "seed": 20261019,
                   "python": platform.python_version(),
                   "networkx": nx.__version__,
                   "elapsed_seconds": round(time.perf_counter() - start, 3)})
    path = Path(__file__).with_name("validation_results.json")
    path.write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    run()
