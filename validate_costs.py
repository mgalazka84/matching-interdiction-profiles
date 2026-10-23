"""Exhaustive oracle checks of the arbitrary-cost inverse frontier."""

from array import array
import json
from pathlib import Path
import platform
import random
import time
import networkx as nx
from matching_profiles import _decomposition
from cost_frontier import solve_cost_frontier


def oracle(graph):
    edges = list(graph.edges)
    m = len(edges)
    assert m <= 18
    incident = {v: 0 for v in graph}
    for i, (u, v) in enumerate(edges):
        incident[u] |= 1 << i
        incident[v] |= 1 << i
    matching = array("B", [0]) * (1 << m)
    costs = [0] * (1 << m)
    for mask in range(1, 1 << m):
        bit = mask & -mask
        i = bit.bit_length() - 1
        u, v = edges[i]
        matching[mask] = max(matching[mask ^ bit],
                             1 + matching[mask & ~(incident[u] | incident[v])])
        costs[mask] = costs[mask ^ bit] + graph[u][v].get("cost", 1)
    full = (1 << m) - 1
    frontier = [costs[full]] * (len(graph) // 2 + 1)
    for deleted in range(1 << m):
        r = matching[full ^ deleted]
        frontier[r] = min(frontier[r], costs[deleted])
    for r in range(1, len(frontier)):
        frontier[r] = min(frontier[r], frontier[r - 1])
    return frontier


def verify(graph, result):
    expected = oracle(graph)
    assert result.costs == expected, (list(graph.edges(data=True)), result.costs, expected)
    for r, cost in enumerate(expected):
        certificate = result.certificate(r)
        deleted, barrier = certificate["deleted"], certificate["barrier"]
        assert len(set(map(frozenset, deleted))) == len(deleted)
        assert all(graph.has_edge(*e) for e in deleted)
        assert sum(graph[u][v].get("cost", 1) for u, v in deleted) == cost
        h = graph.copy()
        h.remove_edges_from(deleted)
        actual = len(nx.max_weight_matching(h, maxcardinality=True))
        assert actual <= r
        assert len(set(barrier)) == len(barrier) and set(barrier) <= set(graph)
        h.remove_nodes_from(barrier)
        odd = sum(len(c) % 2 for c in nx.connected_components(h))
        bound = (len(graph) + len(barrier) - odd) // 2
        assert actual <= bound <= r
    budgets = {0}
    for cost in expected:
        budgets.update([cost, cost + 1])
        if cost:
            budgets.add(cost - 1)
    for budget in budgets:
        assert result.value_for_budget(budget) == next(r for r, c in enumerate(expected) if c <= budget)
    return len(expected), len(budgets)


def run():
    start = time.perf_counter()
    rng = random.Random(20261020)
    counts = {"atlas_graphs": 0, "cost_assignments_per_atlas_graph": 3,
              "instances": 0, "targets_and_witnesses": 0,
              "budget_queries": 0, "random_explicit_order_graphs": 0,
              "maximum_width": 0, "maximum_input_cost_bits": 0}

    def check(g, order=None):
        result = solve_cost_frontier(g, elimination_order=order)
        targets, queries = verify(g, result)
        counts["instances"] += 1
        counts["targets_and_witnesses"] += targets
        counts["budget_queries"] += queries
        counts["maximum_width"] = max(counts["maximum_width"], result.width)
        counts["maximum_input_cost_bits"] = max(counts["maximum_input_cost_bits"],
                                                  result.stats["max_cost_bits"])

    for g in nx.graph_atlas_g():
        try:
            _decomposition(g, None)
        except ValueError:
            continue
        counts["atlas_graphs"] += 1
        check(g)
        for u, v in g.edges:
            g[u][v]["cost"] = rng.randrange(21)
        check(g)
        for u, v in g.edges:
            g[u][v]["cost"] = rng.choice([0, 1, (1 << rng.randrange(80, 129)) + rng.randrange(100)])
        check(g)
    for trial in range(60):
        n = rng.randrange(3, 9)
        g = nx.gnp_random_graph(n, rng.uniform(.2, .65), seed=rng)
        while g.number_of_edges() > 18:
            g.remove_edge(*rng.choice(list(g.edges)))
        g = nx.relabel_nodes(g, {v: ("node", v) for v in g})
        for u, v in g.edges:
            g[u][v]["cost"] = rng.choice([0, rng.randrange(25), 10**30 + rng.randrange(25)])
        order = list(g)
        rng.shuffle(order)
        check(g, order)
        counts["random_explicit_order_graphs"] += 1

    counts.update({"all_checks_passed": True, "seed": 20261020,
                   "python": platform.python_version(), "networkx": nx.__version__,
                   "elapsed_seconds": round(time.perf_counter() - start, 3)})
    Path(__file__).with_name("validation_costs.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    run()
