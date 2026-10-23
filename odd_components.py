"""Independent DP using Tutte--Berge and parity-labelled connectivity states.

Minimizes |U| - odd((G-F)-U) for every deletion budget. A boundary vertex in U
is charged only when forgotten, and a component only when its boundary closes.
This is a second research implementation, not a production matching library.
"""

from dataclasses import dataclass
from functools import lru_cache
import networkx as nx
from matching_profiles import Entry, Factor, _decomposition, _put


@dataclass
class OddProfileResult:
    profile: list
    width: int
    edges: list
    labels: list
    chosen: list
    stats: dict

    def certificate(self, budget):
        if not 0 <= budget < len(self.profile):
            raise ValueError("Budget outside computed range")
        stack, deleted, barrier = [self.chosen[budget]], [], []
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

    def witness(self, budget):
        return self.certificate(budget)["deleted"]


def _zero(boundary):
    table = {}
    for selected in range(1 << len(boundary)):
        blocks = tuple((1 << i, 1) for i in range(len(boundary))
                       if not selected & (1 << i))
        table[selected, blocks] = Entry(0)
    return Factor(boundary, 0, [table])


def _lift(factor, boundary):
    if factor.boundary == boundary:
        return factor
    positions = {v: i for i, v in enumerate(boundary)}
    masks = [sum(1 << positions[v] for j, v in enumerate(factor.boundary)
                 if mask & (1 << j))
             for mask in range(1 << len(factor.boundary))]
    missing = [1 << positions[v] for v in boundary if v not in factor.boundary]
    tables = []
    for table in factor.tables:
        target = {}
        for (selected, blocks), entry in table.items():
            for choice in range(1 << len(missing)):
                new_selected = masks[selected]
                new_blocks = [(masks[mask], parity) for mask, parity in blocks]
                for j, bit in enumerate(missing):
                    if choice & (1 << j):
                        new_selected |= bit
                    else:
                        new_blocks.append((bit, 1))
                target[new_selected, tuple(sorted(new_blocks))] = entry
        tables.append(target)
    return Factor(boundary, factor.edge_count, tables)


@lru_cache(maxsize=None)
def _combine_states(left, right):
    selected, blocks1 = left
    selected2, blocks2 = right
    if selected != selected2:
        return None
    groups = [mask for mask, _ in blocks1]
    for mask, _ in blocks2:
        merged = mask
        remaining = []
        for group in groups:
            if group & mask:
                merged |= group
            else:
                remaining.append(group)
        groups = remaining + [merged]
    result = []
    for group in groups:
        parity = group.bit_count() & 1
        for mask, p in blocks1 + blocks2:
            if mask & group:
                parity ^= p
        result.append((group, parity))
    return selected, tuple(sorted(result))


def _join(left, right, budget, stats):
    assert left.boundary == right.boundary
    total = left.edge_count + right.edge_count
    tables = [{} for _ in range(min(budget, total) + 1)]
    for a, ltab in enumerate(left.tables):
        for b in range(min(budget - a, len(right.tables) - 1) + 1):
            target = tables[a + b]
            for ls, le in ltab.items():
                for rs, re in right.tables[b].items():
                    state = _combine_states(ls, rs)
                    if state is None:
                        continue
                    stats["join_candidates"] += 1
                    value = le.base + re.base
                    old = target.get(state)
                    if old is None or value < old.base:
                        trace = (re.trace if le.trace is None else le.trace
                                 if re.trace is None else ("join", le.trace, re.trace))
                        target[state] = Entry(value, trace)
    return Factor(left.boundary, total, tables)


def _edge(factor, u, v, edge_id, budget):
    endpoints = (1 << factor.boundary.index(u)) | (1 << factor.boundary.index(v))
    tables = [{} for _ in range(min(budget, factor.edge_count + 1) + 1)]
    for b, table in enumerate(factor.tables):
        for state, entry in table.items():
            selected, blocks = state
            kept = state
            if not selected & endpoints:
                touched = [(mask, p) for mask, p in blocks if mask & endpoints]
                if len(touched) == 2:
                    merged = (touched[0][0] | touched[1][0],
                              touched[0][1] ^ touched[1][1])
                    kept = selected, tuple(sorted(
                        [block for block in blocks if block not in touched] + [merged]))
            _put(tables[b], kept, entry.base, entry.trace)
            if b < budget:
                _put(tables[b + 1], state, entry.base,
                     ("delete", edge_id, entry.trace))
    return Factor(factor.boundary, factor.edge_count + 1, tables)


def _forget(factor, v):
    pos = factor.boundary.index(v)
    bit = 1 << pos
    lower = bit - 1

    def project(mask):
        return (mask & lower) | ((mask >> (pos + 1)) << pos)

    tables = []
    for table in factor.tables:
        target = {}
        for (selected, blocks), entry in table.items():
            value, trace = entry.base, entry.trace
            new_blocks = []
            if selected & bit:
                value += 1
                trace = ("barrier", v, trace)
            for mask, parity in blocks:
                remaining = project(mask)
                if remaining:
                    new_blocks.append((remaining, parity))
                else:
                    value -= parity
            state = project(selected), tuple(sorted(new_blocks))
            _put(target, state, value, trace)
        tables.append(target)
    return Factor(tuple(u for u in factor.boundary if u != v), factor.edge_count, tables)


def solve_odd_profile(graph, budget=None, elimination_order=None):
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Use a simple undirected graph without self-loops")
    labels = list(graph)
    numbers = {v: i for i, v in enumerate(labels)}
    g = nx.relabel_nodes(graph, numbers, copy=True)
    edges = [tuple(sorted(e)) for e in g.edges]
    m = len(edges)
    budget = m if budget is None else budget
    if not isinstance(budget, int) or not 0 <= budget <= m:
        raise ValueError("Budget must be an integer in [0, |E|]")
    supplied = None if elimination_order is None else [numbers[v] for v in elimination_order]
    order, later, children, roots = _decomposition(g, supplied)
    rank = {v: i for i, v in enumerate(order)}
    assigned = {v: [] for v in order}
    for i, (u, v) in enumerate(edges):
        assigned[u if rank[u] < rank[v] else v].append((u, v, i))
    factors = {}
    stats = {"join_candidates": 0, "max_states_per_budget": 0}

    def inspect(factor):
        stats["max_states_per_budget"] = max(
            stats["max_states_per_budget"], max(map(len, factor.tables), default=0))

    for v in order:
        bag = tuple(sorted((v,) + later[v]))
        factor = _zero(bag)
        for child in children[v]:
            factor = _join(factor, _lift(factors.pop(child), bag), budget, stats)
            inspect(factor)
        for u, w, i in assigned[v]:
            factor = _edge(factor, u, w, i, budget)
            inspect(factor)
        factors[v] = _forget(factor, v)
        inspect(factors[v])
    root = _zero(())
    for v in roots:
        root = _join(root, factors.pop(v), budget, stats)
    profile, chosen, best = [], [], None
    for table in root.tables:
        entry = table[0, ()]
        if best is None or entry.base < best.base:
            best = entry
        assert (len(g) + best.base) % 2 == 0
        profile.append((len(g) + best.base) // 2)
        chosen.append(best.trace)
    return OddProfileResult(profile, max(map(len, later.values()), default=0),
                            [(labels[u], labels[v]) for u, v in edges],
                            labels, chosen, stats)
