"""Minimum deletion cost for every target cardinality matching number.

Matching weights are one. Deletion costs are arbitrary nonnegative integers,
stored in the edge attribute `cost` by default (missing attributes mean one).
The DP indexes exact Tutte--Berge scores and stores minimum costs. It never
allocates an array indexed by a monetary budget or a sum of deletion costs.
"""

from dataclasses import dataclass
import networkx as nx
from matching_profiles import Entry, _decomposition, _put
from odd_components import _combine_states


@dataclass
class CostFactor:
    boundary: tuple
    interior: int
    table: dict


@dataclass
class CostFrontierResult:
    costs: list
    width: int
    edges: list
    labels: list
    chosen: list
    stats: dict

    def certificate(self, target):
        if not isinstance(target, int) or not 0 <= target < len(self.costs):
            raise ValueError("Target outside computed range")
        stack, deleted, barrier = [self.chosen[target]], [], []
        while stack:
            trace = stack.pop()
            if trace is None:
                continue
            if trace[0] == "delete":
                deleted.append(self.edges[trace[1]])
                stack.append(trace[2])
            elif trace[0] == "barrier":
                barrier.append(self.labels[trace[1]])
                stack.append(trace[2])
            else:
                stack.extend(trace[1:])
        return {"deleted": deleted, "barrier": barrier}

    def witness(self, target):
        return self.certificate(target)["deleted"]

    def value_for_budget(self, budget):
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("Budget must be a nonnegative integer")
        # The inverse frontier is nonincreasing in the target.
        lo, hi = 0, len(self.costs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.costs[mid] <= budget:
                hi = mid
            else:
                lo = mid + 1
        return lo


def _zero(boundary):
    table = {}
    for selected in range(1 << len(boundary)):
        blocks = tuple((1 << i, 1) for i in range(len(boundary))
                       if not selected & (1 << i))
        table[((selected, blocks), 0)] = Entry(0)
    return CostFactor(boundary, 0, table)


def _lift(factor, boundary):
    if factor.boundary == boundary:
        return factor
    assert set(factor.boundary) <= set(boundary)
    positions = {v: i for i, v in enumerate(boundary)}
    masks = [sum(1 << positions[v] for j, v in enumerate(factor.boundary)
                 if mask & (1 << j))
             for mask in range(1 << len(factor.boundary))]
    missing = [1 << positions[v] for v in boundary if v not in factor.boundary]
    target = {}
    for ((selected, blocks), q), entry in factor.table.items():
        for choice in range(1 << len(missing)):
            new_selected = masks[selected]
            new_blocks = [(masks[mask], p) for mask, p in blocks]
            for j, bit in enumerate(missing):
                if choice & (1 << j):
                    new_selected |= bit
                else:
                    new_blocks.append((bit, 1))
            state = new_selected, tuple(sorted(new_blocks))
            _put(target, (state, q), entry.base, entry.trace)
    return CostFactor(boundary, factor.interior, target)


def _join(left, right, stats):
    assert left.boundary == right.boundary
    target = {}
    for (ls, a), le in left.table.items():
        for (rs, b), re in right.table.items():
            state = _combine_states(ls, rs)
            if state is None:
                continue
            stats["join_candidates"] += 1
            trace = (re.trace if le.trace is None else le.trace
                     if re.trace is None else ("join", le.trace, re.trace))
            _put(target, (state, a + b), le.base + re.base, trace)
    return CostFactor(left.boundary, left.interior + right.interior, target)


def _edge(factor, u, v, edge_id, cost):
    endpoints = (1 << factor.boundary.index(u)) | (1 << factor.boundary.index(v))
    target = {}
    for (state, q), entry in factor.table.items():
        selected, blocks = state
        kept = state
        if not selected & endpoints:
            touched = [(mask, p) for mask, p in blocks if mask & endpoints]
            if len(touched) == 2:
                merged = (touched[0][0] | touched[1][0],
                          touched[0][1] ^ touched[1][1])
                kept = selected, tuple(sorted(
                    [block for block in blocks if block not in touched] + [merged]))
        _put(target, (kept, q), entry.base, entry.trace)
        _put(target, (state, q), entry.base + cost,
             ("delete", edge_id, entry.trace))
    return CostFactor(factor.boundary, factor.interior, target)


def _forget(factor, v):
    pos = factor.boundary.index(v)
    bit = 1 << pos
    lower = bit - 1

    def project(mask):
        return (mask & lower) | ((mask >> (pos + 1)) << pos)

    target = {}
    for ((selected, blocks), q), entry in factor.table.items():
        trace = entry.trace
        if selected & bit:
            q += 1
            trace = ("barrier", v, trace)
        new_blocks = []
        for mask, parity in blocks:
            remaining = project(mask)
            if remaining:
                new_blocks.append((remaining, parity))
            else:
                q -= parity
        state = project(selected), tuple(sorted(new_blocks))
        _put(target, (state, q), entry.base, trace)
    return CostFactor(tuple(u for u in factor.boundary if u != v),
                      factor.interior + 1, target)


def solve_cost_frontier(graph, cost_attribute="cost", elimination_order=None):
    """Return K[r] = min{cost(F): nu(G-F) <= r}, 0 <= r <= floor(n/2)."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Use a simple undirected graph without self-loops")
    labels = list(graph)
    numbers = {v: i for i, v in enumerate(labels)}
    g = nx.relabel_nodes(graph, numbers, copy=True)
    edges = [tuple(sorted(e)) for e in g.edges]
    costs = [g[u][v].get(cost_attribute, 1) for u, v in edges]
    if any(type(c) is not int or c < 0 for c in costs):
        raise ValueError("Deletion costs must be nonnegative integers")
    supplied = None if elimination_order is None else [numbers[v] for v in elimination_order]
    order, later, children, roots = _decomposition(g, supplied)
    rank = {v: i for i, v in enumerate(order)}
    assigned = {v: [] for v in order}
    for i, (u, v) in enumerate(edges):
        assigned[u if rank[u] < rank[v] else v].append((u, v, i))
    factors = {}
    stats = {"join_candidates": 0, "max_entries": 0,
             "max_cost_bits": max(costs, default=0).bit_length()}

    def inspect(factor):
        assert all(-factor.interior <= q <= factor.interior
                   for state, q in factor.table)
        stats["max_entries"] = max(stats["max_entries"], len(factor.table))

    for v in order:
        bag = tuple(sorted((v,) + later[v]))
        factor = _zero(bag)
        for child in children[v]:
            factor = _join(factor, _lift(factors.pop(child), bag), stats)
            inspect(factor)
        for u, w, i in assigned[v]:
            factor = _edge(factor, u, w, i, costs[i])
            inspect(factor)
        factors[v] = _forget(factor, v)
        inspect(factors[v])
    root = _zero(())
    for v in roots:
        root = _join(root, factors.pop(v), stats)
    ordered = sorted((q, entry) for (state, q), entry in root.table.items())
    frontier, chosen, best, j = [], [], None, 0
    for r in range(len(g) // 2 + 1):
        while j < len(ordered) and ordered[j][0] <= 2 * r - len(g):
            entry = ordered[j][1]
            if best is None or entry.base < best.base:
                best = entry
            j += 1
        assert best is not None
        frontier.append(best.base)
        chosen.append(best.trace)
    return CostFrontierResult(frontier, max(map(len, later.values()), default=0),
                              [(labels[u], labels[v]) for u, v in edges],
                              labels, chosen, stats)
