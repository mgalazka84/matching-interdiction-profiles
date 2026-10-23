# Matching interdiction profiles

Companion research code for **Matching under Edge Failures: Complete Profiles
at Bounded Treewidth**, by **Marek Gałązka and Hanna Wdowicka**.

This repository contains algorithms, reproduction instructions, and recorded
small-graph validation results. It does not contain the manuscript or private
editorial documents.

## Model and solvers

All matching edges have weight one. Graphs are finite, simple, and undirected.

| Module | Output |
| --- | --- |
| `odd_components.py` | Complete unit-deletion-cost profile using parity and connectivity states |
| `matching_profiles.py` | Independent normalized boundary-profile formulation |
| `cost_frontier.py` | Minimum deletion cost for every target matching size, with arbitrary nonnegative integer deletion costs |

The cost solver uses an exact Tutte–Berge score index; it does not allocate
arrays indexed by monetary budgets. The results do not cover arbitrary
matching weights.

## Reproduction

The archived records were produced with Python 3.12.14 and NetworkX 3.7.
Python 3.10 or later is required.

```bash
python -m pip install -r requirements.txt
python validate.py
python validate_costs.py
```

The validation scripts overwrite `validation_results.json` and
`validation_costs.json` with fresh records. Elapsed times vary with the
environment and are not performance benchmarks. To check the published file
contents before rerunning the scripts:

```bash
sha256sum -c SHA256SUMS.txt
```

## Small example

```python
import networkx as nx
from odd_components import solve_odd_profile
from cost_frontier import solve_cost_frontier

g = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3),
              (3, 4), (4, 5), (5, 3)])

unit = solve_odd_profile(g, budget=4)
assert unit.profile == [3, 2, 2, 2, 1]
deleted = unit.witness(4)
certificate = unit.certificate(4)  # deleted edges and barrier vertices

inverse = solve_cost_frontier(g)  # a missing edge cost means one
assert inverse.costs == [7, 4, 1, 0]
assert inverse.value_for_budget(4) == 1
deleted = inverse.witness(1)  # target matching number, not a budget
```

Without an elimination order, the solvers recognize partial 2-trees using
degree-at-most-two elimination with fill edges. For other graphs, supply
`elimination_order` as a list of the original vertex labels. The reported
width is the induced width of that order, not necessarily minimum treewidth.
Fill edges construct the decomposition and are never eligible deletion edges.

`solve_profile` and `solve_odd_profile` compute every budget through `budget`,
which defaults to the number of edges. `solve_cost_frontier` accepts
`cost_attribute`, defaulting to `"cost"`. Costs are nonnegative Python
integers; floats are rejected. Vertex labels may be arbitrary hashable objects.

## Recorded checks

- Unit costs: 497 graph-atlas partial 2-trees, 80 additional random partial
  2-trees, 50 supplied-order graphs, and 80 truncated profiles; 5339 budget
  values and witnesses per implementation. Seed: 20261019.
- Arbitrary costs: 1551 instances, 6007 target values and witnesses, and
  12951 budget queries, including zero costs and integers up to 129 bits.
  Seed: 20261020.

All archived checks passed. The exhaustive oracle branches on original
edges and does not use a tree decomposition or the Tutte–Berge formula.
Witnesses also undergo independent matching and odd-component checks.

The implementations share decomposition code, and the two parity solvers
share state-combination code. These are research prototypes. Dictionaries,
arbitrary-precision integers, and witness traces add costs beyond the
arithmetic-operation model. The recorded tests do not replace mathematical
proofs or establish practical superiority over earlier algorithms.

## Version and citation

Version 0.2.0 accompanies the journal revision. It publishes the existing
implementations and recorded results, with refreshed documentation and
citation metadata. The solver and validation files are unchanged from the
version used for the reported experiments. See `CITATION.cff`; cite the
exact release or commit when reproducing results.

OpenAI ChatGPT/Codex assisted in writing implementation and validation code.
No archival DOI has been assigned to this repository.
