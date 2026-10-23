"""Exact cardinality matching interdiction by normalized boundary profiles.

Research prototype, 2026-10-19. Unit matching weights and unit deletion costs.
The default decomposition recognizes partial 2-trees. For other graphs, supply
an elimination order. Fill edges are used ONLY to form bags, never as input edges.
Python integers, dictionaries, and witness traces add implementation overhead
beyond the arithmetic-operation bound in the accompanying manuscript.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import networkx as nx


@dataclass(frozen=True)
class Entry:
    base: int
    trace: object = None


@dataclass
class Factor:
    boundary: tuple
    edge_count: int
    tables: list


@dataclass
class ProfileResult:
    profile: list
    width: int
    edges: list
    chosen: list
    stats: dict

    def witness(self, budget):
        """Return one optimal edge-deletion set for the given at-most budget."""
        if not 0 <= budget < len(self.profile):
            raise ValueError("Budget outside the computed range")
        stack = [self.chosen[budget]]
        deleted = []
        while stack:
            trace = stack.pop()
            if trace is None:
                continue
            if trace[0] == "delete":
                deleted.append(self.edges[trace[1]])
                stack.append(trace[2])
            else:
                stack.extend(trace[1:])
        return deleted


def _decomposition(graph, order):
    n = len(graph)
    adjacency = {v: set(graph.neighbors(v)) for v in graph}
    active = set(graph)
    later = {}
    if order is None:
        queue = deque(v for v in graph if len(adjacency[v]) <= 2)
        result = []
        while queue:
            v = queue.popleft()
            if v not in active or len(adjacency[v]) > 2:
                continue
            neighbors = tuple(sorted(adjacency[v]))
            later[v] = neighbors
            result.append(v)
            active.remove(v)
            if len(neighbors) == 2:
                a, b = neighbors
                adjacency[a].add(b)
                adjacency[b].add(a)
            for u in neighbors:
                adjacency[u].remove(v)
                if len(adjacency[u]) <= 2:
                    queue.append(u)
            adjacency[v].clear()
        if active:
            raise ValueError("Graph is not a partial 2-tree; supply an elimination order")
        order = result
    else:
        order = list(order)
        if len(order) != n or set(order) != active:
            raise ValueError("Elimination order must contain each vertex exactly once")
        for v in order:
            neighbors = tuple(sorted(adjacency[v]))
            later[v] = neighbors
            for j, u in enumerate(neighbors):
                for z in neighbors[j + 1:]:
                    adjacency[u].add(z)
                    adjacency[z].add(u)
            for u in neighbors:
                adjacency[u].remove(v)
            adjacency[v].clear()
    rank = {v: i for i, v in enumerate(order)}
    children = {v: [] for v in order}
    roots = []
    for v in order:
        if later[v]:
            parent = min(later[v], key=rank.__getitem__)
            children[parent].append(v)
            assert set(later[v]) <= set(later[parent]) | {parent}
        else:
            roots.append(v)
    return order, later, children, roots


def _zero(boundary):
    return Factor(boundary, 0, [{(0,) * (1 << len(boundary)): Entry(0)}])


def _put(table, signature, base, trace):
    old = table.get(signature)
    if old is None or base < old.base:
        table[signature] = Entry(base, trace)


def _lift(factor, boundary):
    if boundary == factor.boundary:
        return factor
    assert set(factor.boundary) <= set(boundary)
    positions = {v: i for i, v in enumerate(boundary)}
    projection = [sum(1 << j for j, v in enumerate(factor.boundary)
                      if mask & (1 << positions[v]))
                  for mask in range(1 << len(boundary))]
    return Factor(boundary, factor.edge_count,
                  [{tuple(sig[m] for m in projection): entry
                    for sig, entry in table.items()} for table in factor.tables])


@lru_cache(maxsize=None)
def _join_signature(left, right):
    full = len(left) - 1
    result = []
    for forbidden in range(full + 1):
        available = full ^ forbidden
        part = available
        best = 0
        while True:
            best = max(best, left[forbidden | part] + right[full ^ part])
            if part == 0:
                break
            part = (part - 1) & available
        result.append(best)
    return tuple(result)


def _join(left, right, budget, stats):
    assert left.boundary == right.boundary
    total = left.edge_count + right.edge_count
    tables = [{} for _ in range(min(budget, total) + 1)]
    for a, ltab in enumerate(left.tables):
        for b in range(min(len(right.tables) - 1, budget - a) + 1):
            target = tables[a + b]
            for lsig, le in ltab.items():
                for rsig, re in right.tables[b].items():
                    stats["join_candidates"] += 1
                    sig = _join_signature(lsig, rsig)
                    base = le.base + re.base
                    old = target.get(sig)
                    if old is None or base < old.base:
                        trace = (re.trace if le.trace is None else le.trace
                                 if re.trace is None else ("join", le.trace, re.trace))
                        target[sig] = Entry(base, trace)
    return Factor(left.boundary, total, tables)


def _edge(factor, u, v, edge_id, budget):
    boundary = factor.boundary
    endpoint_mask = (1 << boundary.index(u)) | (1 << boundary.index(v))
    tables = [{} for _ in range(min(budget, factor.edge_count + 1) + 1)]
    for b, table in enumerate(factor.tables):
        for sig, entry in table.items():
            kept = tuple(value if mask & endpoint_mask else
                         max(value, 1 + sig[mask | endpoint_mask])
                         for mask, value in enumerate(sig))
            _put(tables[b], kept, entry.base, entry.trace)
            if b < budget:
                _put(tables[b + 1], sig, entry.base,
                     ("delete", edge_id, entry.trace))
    return Factor(boundary, factor.edge_count + 1, tables)


def _forget(factor, v):
    old_boundary = factor.boundary
    boundary = tuple(u for u in old_boundary if u != v)
    old_positions = [old_boundary.index(u) for u in boundary]
    projection = [sum(1 << old_positions[j] for j in range(len(boundary))
                      if mask & (1 << j))
                  for mask in range(1 << len(boundary))]
    base_index = projection[-1]
    tables = []
    for table in factor.tables:
        new_table = {}
        for sig, entry in table.items():
            shift = sig[base_index]
            normalized = tuple(sig[m] - shift for m in projection)
            _put(new_table, normalized, entry.base + shift, entry.trace)
        tables.append(new_table)
    return Factor(boundary, factor.edge_count, tables)


def solve_profile(graph, budget=None, elimination_order=None):
    """Compute R_G(0), ..., R_G(B), for an unweighted simple undirected graph.

    budget defaults to |E| and must be between 0 and |E|. Node labels may be any
    hashable objects. An optional elimination_order uses the original labels.
    Without it, graphs of treewidth at most two are recognized by elimination.
    """
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Use a simple undirected graph without self-loops")
    labels = list(graph.nodes)
    number = {v: i for i, v in enumerate(labels)}
    g = nx.relabel_nodes(graph, number, copy=True)
    edges = [tuple(sorted(e)) for e in g.edges]
    m = len(edges)
    budget = m if budget is None else budget
    if not isinstance(budget, int) or not 0 <= budget <= m:
        raise ValueError("Budget must be an integer in [0, |E|]")
    order_arg = None if elimination_order is None else [number[v] for v in elimination_order]
    order, later, children, roots = _decomposition(g, order_arg)
    rank = {v: i for i, v in enumerate(order)}
    original_edges = {v: [] for v in order}
    for edge_id, (u, v) in enumerate(edges):
        first = u if rank[u] < rank[v] else v
        original_edges[first].append((u, v, edge_id))
    factors = {}
    stats = {"join_candidates": 0, "retained_entries": 0,
             "max_signatures_per_budget": 0, "graphs_vertices": len(g),
             "graphs_edges": m, "budget": budget}

    def inspect(factor):
        stats["retained_entries"] += sum(len(t) for t in factor.tables)
        stats["max_signatures_per_budget"] = max(
            stats["max_signatures_per_budget"], max(map(len, factor.tables), default=0))

    for v in order:
        bag = tuple(sorted((v,) + later[v]))
        factor = _zero(bag)
        for child in children[v]:
            factor = _join(factor, _lift(factors.pop(child), bag), budget, stats)
            inspect(factor)
        for u, z, edge_id in original_edges[v]:
            factor = _edge(factor, u, z, edge_id, budget)
            inspect(factor)
        factors[v] = _forget(factor, v)
        inspect(factors[v])
    root = _zero(())
    for v in roots:
        root = _join(root, factors.pop(v), budget, stats)
    profile, chosen = [], []
    best = None
    for table in root.tables:
        entry = table.get((0,))
        if entry is not None and (best is None or entry.base < best.base):
            best = entry
        assert best is not None
        profile.append(best.base)
        chosen.append(best.trace)
    output_edges = [(labels[u], labels[v]) for u, v in edges]
    return ProfileResult(profile, max((len(later[v]) for v in later), default=0),
                         output_edges, chosen, stats)


def brute_force_profile(graph, budget=None):
    """Independent exact oracle: enumerate every remaining edge set (exponential)."""
    edges = list(graph.edges)
    m = len(edges)
    if m > 23:
        raise ValueError("Brute-force oracle is limited to 23 edges")
    budget = m if budget is None else budget
    incident = {v: 0 for v in graph}
    for i, (u, v) in enumerate(edges):
        incident[u] |= 1 << i
        incident[v] |= 1 << i
    from array import array
    values = array("B", [0]) * (1 << m)
    for mask in range(1, 1 << m):
        bit = mask & -mask
        i = bit.bit_length() - 1
        u, v = edges[i]
        values[mask] = max(values[mask ^ bit],
                           1 + values[mask & ~(incident[u] | incident[v])])
    exact = [len(graph) + 1] * (budget + 1)
    full = (1 << m) - 1
    for deleted in range(1 << m):
        b = deleted.bit_count()
        if b <= budget:
            exact[b] = min(exact[b], values[full ^ deleted])
    for b in range(1, budget + 1):
        exact[b] = min(exact[b], exact[b - 1])
    return exact


if __name__ == "__main__":
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3),
                      (3, 4), (4, 5), (5, 3)])
    result = solve_profile(graph)
    print("Two triangles joined by a bridge")
    print("Profile:", result.profile)
    for b in range(len(result.profile)):
        print(f"B={b}: optimum={result.profile[b]}, delete={result.witness(b)}")
