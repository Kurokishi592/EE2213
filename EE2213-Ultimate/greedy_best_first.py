"""Greedy Best-First Search (GBFS) utilities.

Greedy best-first expands the node that appears closest to the goal according
only to a heuristic h(n). It does NOT guarantee shortest path; it is not
optimal. For optimal paths prefer A* (not implemented here yet).

`coords` mapping modes (node -> value):
1. Coordinate sequences (list/tuple) of length >=1 (1D, 2D, nD). Distance to goal
     computed with L1 or L2 based on `heuristic` argument.
2. Direct heuristic values (float/int). If ALL values are numeric scalars we treat
     them as precomputed h(n) and use them directly (ignoring geometric distance).
     Ensure h(goal)=0 if you want admissible heuristics. Not enforced.

Functions:
    greedy_best_first(graph, start, goal, coords, heuristic='l2', verbose=False,
                                        max_expansions=None)
        - graph: dict[node] -> list of neighbors (unweighted)
        - coords: dict[node] -> sequence OR scalar
        - heuristic: 'l2' or 'l1' (label only for direct mode)
        Returns path list start->goal or None. Verbose prints:
            Heuristic mode & values (unsorted + ascending)
            Greedy path, hops, nodes expanded
            Warning if expansion cap exceeded
    print_greedy_best_first(...) convenience wrapper with verbose output.
    greedy_best_first_raw(graph, start, goal, heuristics, **kwargs)  # classic GBFS with direct h(n)
    greedy_best_first_l1(graph, start, goal, coords, **kwargs)       # classic GBFS using L1 distance
    greedy_best_first_l2(graph, start, goal, coords, **kwargs)       # classic GBFS using L2 distance
"""
from typing import Dict, List, Sequence, Any, Optional, Tuple, Union
import heapq


def _distance(a: Union[Sequence[float], float, int],
              b: Union[Sequence[float], float, int], mode: str) -> float:
    # Coerce scalars to 1D sequences for uniform handling
    if not isinstance(a, (list, tuple)):
        a = (a,)
    if not isinstance(b, (list, tuple)):
        b = (b,)
    if mode == 'l1':
        return sum(abs(x - y) for x, y in zip(a, b))
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def greedy_best_first(
    graph: Dict[Any, List[Any]],
    start: Any,
    goal: Any,
    coords: Dict[Any, Sequence[float]],
    heuristic: str = 'l2',
    verbose: bool = False,
    max_expansions: Optional[int] = None,
    tree_search: bool = False,
    max_revisits: int = 1000,
) -> Optional[List[Any]]:
    # Basic guards
    if start not in graph or goal not in graph or start not in coords or goal not in coords:
        if verbose:
            print("No route found.")
        return None

    # Detect direct heuristic mode (all scalars)
    direct_mode = all(isinstance(v, (int, float)) for v in coords.values())

    if verbose:
        if direct_mode:
            all_h = {n: float(coords[n]) for n in graph.keys() if n in coords}
            mode_label = "direct"
        else:
            all_h = {n: _distance(coords[n], coords[goal], heuristic)
                     for n in graph.keys() if n in coords and goal in coords}
            mode_label = f"geometric-{heuristic}"
        print("Heuristic mode:", mode_label)
        print("Heuristic values (unsorted):", [(n, round(h, 4)) for n, h in all_h.items()])
        print("Heuristic values (ascending):", [(n, round(all_h[n], 4)) for n in sorted(all_h, key=all_h.get)])

    # Priority queue holds (heuristic_value, node)
    pq: List[Tuple[float, Any]] = []
    h_start = (float(coords[start]) if direct_mode
               else _distance(coords[start], coords[goal], heuristic))
    heapq.heappush(pq, (h_start, start))

    predecessor: Dict[Any, Optional[Any]] = {start: None}
    visited = set()  # used only in graph-search mode
    order: List[Any] = []
    revisit_count: Dict[Any, int] = {}

    # Dynamic default cap: 10x number of nodes (very generous) to flag unusual looping
    if max_expansions is None:
        max_expansions = max(10 * len(graph), 1)
    expansions = 0

    while pq:
        h_val, node = heapq.heappop(pq)
        # Graph-search: skip already expanded nodes
        if not tree_search and node in visited:
            continue
        # Tree-search: count revisits (node may appear multiple times)
        if tree_search:
            revisit_count[node] = revisit_count.get(node, 0) + 1
            if revisit_count[node] > max_revisits:
                if verbose:
                    print(f"Warning: node '{node}' revisited > {max_revisits} times (possible loop). Aborting.")
                    print("Nodes expanded (partial):", order)
                return None
        else:
            visited.add(node)
        order.append(node)
        expansions += 1
        if expansions > max_expansions:
            if verbose:
                print("Warning: expansion limit exceeded (possible infinite loop). Stopping search.")
                print("Nodes expanded (partial):", order)
            return None
        if node == goal:
            # Reconstruct greedy path
            path: List[Any] = []
            while node is not None:
                path.append(node)
                node = predecessor[node]
            path.reverse()
            # Compute path cost: sum of weights if weighted adjacency (dict-of-dicts), else hop count
            cost = 0
            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    edges = graph.get(u, {})
                    if isinstance(edges, dict):
                        cost += edges.get(v, 1)
                    else:
                        cost += 1
            if verbose:
                print("Greedy path from", start, "to", goal, ":", path)
                print("Greedy hops:", len(path) - 1)
                print("Total cost:", cost)
                print("Nodes expanded (in order):", order)
                print("Heuristic used:", "direct" if direct_mode else heuristic)
            return path
        # Obtain neighbors: allow list OR dict-of-weights; ignore weights for greedy
        raw_neighbors = graph.get(node, [])
        if isinstance(raw_neighbors, dict):
            iterable_neighbors = raw_neighbors.keys()
        else:
            iterable_neighbors = raw_neighbors
        for nbr in iterable_neighbors:
            if (tree_search or nbr not in visited) and nbr in coords:
                h_nbr = (float(coords[nbr]) if direct_mode
                         else _distance(coords[nbr], coords[goal], heuristic))
                heapq.heappush(pq, (h_nbr, nbr))
                if nbr not in predecessor:
                    predecessor[nbr] = node
    if verbose:
        print("Nodes expanded (in order):", order)
        if tree_search and revisit_count:
            print("Revisit counts:", {k: v for k, v in revisit_count.items() if v > 1})
        print("No route found.")
    return None


def print_greedy_best_first(graph, start, goal, coords, heuristic='l2', **kwargs):
    return greedy_best_first(graph, start, goal, coords, heuristic=heuristic, verbose=True, **kwargs)


# --- Convenience wrappers for "classic" greedy best-first (tree search by default) ---
def greedy_best_first_raw(graph, start, goal, heuristics, verbose=False, **kwargs):
    """Classic greedy best-first using precomputed scalar heuristic values h(n).
    By default uses tree_search (no visited set) to mirror textbook GBFS.
    """
    return greedy_best_first(
        graph, start, goal, heuristics,
        heuristic='l2',  # label ignored in direct mode
        verbose=verbose,
        tree_search=kwargs.get('tree_search', True),
        max_expansions=kwargs.get('max_expansions'),
        max_revisits=kwargs.get('max_revisits', 1000)
    )


def greedy_best_first_l1(graph, start, goal, coords, verbose=False, **kwargs):
    """Classic greedy best-first using L1 (Manhattan) distance as heuristic."""
    return greedy_best_first(
        graph, start, goal, coords,
        heuristic='l1',
        verbose=verbose,
        tree_search=kwargs.get('tree_search', True),
        max_expansions=kwargs.get('max_expansions'),
        max_revisits=kwargs.get('max_revisits', 1000)
    )


def greedy_best_first_l2(graph, start, goal, coords, verbose=False, **kwargs):
    """Classic greedy best-first using L2 (Euclidean) distance as heuristic."""
    return greedy_best_first(
        graph, start, goal, coords,
        heuristic='l2',
        verbose=verbose,
        tree_search=kwargs.get('tree_search', True),
        max_expansions=kwargs.get('max_expansions'),
        max_revisits=kwargs.get('max_revisits', 1000)
    )


if __name__ == "__main__":
    # Simple demo
    graph_demo = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': ['G'],
        'E': ['G'],
        'F': ['H'],
        'G': ['Z'],
        'H': ['Z'],
        'Z': []
    }
    coords_demo = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (1, -1),
        'D': (2, 2),
        'E': (2, 0.5),
        'F': (2, -1.5),
        'G': (3, 1.5),
        'H': (3, -1),
        'Z': (4, 0),
    }
    print_greedy_best_first(graph_demo, 'A', 'Z', coords_demo, heuristic='l2')
