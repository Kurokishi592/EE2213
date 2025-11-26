"""A* Search utilities.

A* expands the node with the smallest f(n) = g(n) + h(n), combining the
actual cost from start (g) and a heuristic estimate to goal (h).
It is optimal if the heuristic is admissible (never overestimates true remaining cost)
and consistent (monotone). This implementation mirrors the output style of
`greedy_best_first` so they can be compared directly.

`coords` mapping modes (node -> value):
1. Coordinate sequences (list/tuple) of length >=1 (1D, 2D, nD). Heuristic distance
   computed with L1 or L2 based on `heuristic` argument.
2. Direct heuristic values (float/int). If ALL values are numeric scalars we treat
   them as precomputed h(n) and use them directly. Ensure h(goal)=0 for optimality.

Functions:
    a_star(graph, start, goal, coords, heuristic='l2', verbose=False,
           max_expansions=None, tree_search=False, check_admissible=True,
           reopen_inconsistent=True)
        - graph: dict[node] -> list of neighbors OR dict[node] -> dict[neighbor] -> weight
        - coords: dict[node] -> sequence OR scalar
        - heuristic: 'l2' or 'l1' (label only for direct mode)
        Returns path list start->goal or None. Verbose prints:
            Heuristic mode & values (unsorted + ascending)
            A* path, hops, total cost, nodes expanded
            Warning if expansion cap exceeded

    a_star_raw(graph, start, goal, heuristics, **kwargs)  # direct h(n)
    a_star_l1(graph, start, goal, coords, **kwargs)       # L1 heuristic
    a_star_l2(graph, start, goal, coords, **kwargs)       # L2 heuristic

"""
from typing import Dict, List, Sequence, Any, Optional, Tuple, Union
import heapq


def _distance(a: Union[Sequence[float], float, int],
              b: Union[Sequence[float], float, int], mode: str) -> float:
    if not isinstance(a, (list, tuple)):
        a = (a,)
    if not isinstance(b, (list, tuple)):
        b = (b,)
    if mode == 'l1':
        return sum(abs(x - y) for x, y in zip(a, b))
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _reverse_graph(graph: Dict[Any, Any]) -> Dict[Any, Dict[Any, float]]:
    rev: Dict[Any, Dict[Any, float]] = {}
    for u, nbrs in graph.items():
        if isinstance(nbrs, dict):
            for v, w in nbrs.items():
                rev.setdefault(v, {})[u] = w if isinstance(w, (int, float)) else 1
        else:  # list neighbors, unit weight
            for v in nbrs:
                rev.setdefault(v, {})[u] = 1
        rev.setdefault(u, {})  # ensure node exists even if no incoming
    return rev

def _dijkstra_from_goal(graph: Dict[Any, Any], goal: Any) -> Dict[Any, float]:
    # Compute true minimal cost from each node to goal by reversing edges
    rev = _reverse_graph(graph)
    dist: Dict[Any, float] = {goal: 0.0}
    pq: List[Tuple[float, Any]] = [(0.0, goal)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        for nbr, w in rev.get(node, {}).items():
            nd = d + (w if isinstance(w, (int, float)) else 1)
            if nd < dist.get(nbr, float('inf')):
                dist[nbr] = nd
                heapq.heappush(pq, (nd, nbr))
    return dist

def a_star(
    graph: Dict[Any, Any],
    start: Any,
    goal: Any,
    coords: Dict[Any, Sequence[float]],
    heuristic: str = 'l2',
    verbose: bool = False,
    max_expansions: Optional[int] = None,
    tree_search: bool = False,
    check_admissible: bool = True,
    reopen_inconsistent: bool = False,
) -> Optional[List[Any]]:
    # Basic guards
    if start not in graph or goal not in graph or start not in coords or goal not in coords:
        if verbose:
            print("No route found.")
        return None

    direct_mode = all(isinstance(v, (int, float)) for v in coords.values())

    if verbose:
        print("Search mode:", "tree-search" if tree_search else "graph-search")
    if verbose or check_admissible:
        if direct_mode:
            all_h = {n: float(coords[n]) for n in graph.keys() if n in coords}
            mode_label = "direct"
        else:
            all_h = {n: _distance(coords[n], coords[goal], heuristic)
                     for n in graph.keys() if n in coords and goal in coords}
            mode_label = f"geometric-{heuristic}"
        # Compute true minimal costs from each node to goal (reachability & admissibility assessment)
        true_costs = _dijkstra_from_goal(graph, goal)
        if verbose:
            print("Heuristic mode:", mode_label)
            unsorted_pairs = [
                (n, round(all_h[n], 4), None if n not in true_costs else round(true_costs[n], 4))
                for n in all_h
            ]
            ascending_pairs = [
                (n, round(all_h[n], 4), None if n not in true_costs else round(true_costs[n], 4))
                for n in sorted(all_h, key=all_h.get)
            ]
            print("Heuristic vs true cost (unsorted):", unsorted_pairs)
            print("Heuristic vs true cost (ascending by h):", ascending_pairs)
        if check_admissible:
            # Admissibility: h(n) <= true_cost(n)
            adm_violations = []
            for n, h in all_h.items():
                if n in true_costs:
                    true = true_costs[n]
                    if h > true + 1e-12:
                        adm_violations.append((n, h, true))
            if not adm_violations:
                print("All heuristics are admissible.")
            else:
                print("Non-admissible heuristics detected (h(n) > true cost to goal):")
                for n, h, true in adm_violations:
                    print(f"  Node {n}: h={h:.4f}, true={true:.4f}, excess={h-true:.4f}")

            # Consistency (monotonicity): for every edge (u,v): h(u) <= cost(u,v) + h(v)
            cons_violations = []
            for u, nbrs in graph.items():
                if u not in all_h:
                    continue
                if isinstance(nbrs, dict):
                    items = nbrs.items()
                else:
                    items = [(v, 1) for v in nbrs]
                for v, w in items:
                    if v not in all_h:
                        continue
                    cost = w if isinstance(w, (int, float)) else 1
                    if all_h[u] > cost + all_h[v] + 1e-12:
                        cons_violations.append((u, v, all_h[u], all_h[v], cost, all_h[u] - (cost + all_h[v])))
            if not cons_violations:
                print("All heuristics are consistent.")
            else:
                print("Inconsistent heuristics detected (h(u) > c(u,v) + h(v)):")
                for u, v, hu, hv, c, excess in cons_violations:
                    print(f"  Edge {u}->{v}: h({u})={hu:.4f}, c={c:.4f}, h({v})={hv:.4f}, excess={excess:.4f}")

    # g(n): cost from start; initialize
    g_cost: Dict[Any, float] = {start: 0.0}
    predecessor: Dict[Any, Optional[Any]] = {start: None}
    open_heap: List[Tuple[float, float, Any]] = []  # (f, h, node)

    h_start = (float(coords[start]) if direct_mode
               else _distance(coords[start], coords[goal], heuristic))
    heapq.heappush(open_heap, (h_start, h_start, start))

    closed = set()  # used only in graph search mode
    order: List[Any] = []
    expansions = 0
    if max_expansions is None:
        max_expansions = max(10 * len(graph), 1)

    while open_heap:
        f_val, h_val, node = heapq.heappop(open_heap)
        if not tree_search:
            if node in closed:
                continue
            closed.add(node)
        order.append(node)
        expansions += 1
        if expansions > max_expansions:
            if verbose:
                print("Warning: expansion limit exceeded (possible infinite loop). Stopping search.")
                print("Nodes expanded (partial):", order)
            return None

        if node == goal:
            # Reconstruct path
            path: List[Any] = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = predecessor[cur]
            path.reverse()
            total_cost = g_cost[goal]
            if verbose:
                print("A* path from", start, "to", goal, ":", path)
                print("A* hops:", len(path) - 1)
                print("Total cost:", total_cost)
                print("Nodes expanded (in order):", order)
                print("Heuristic used:", "direct" if direct_mode else heuristic)
            return path

        # Get neighbors (list OR dict-of-weight)
        raw_neighbors = graph.get(node, [])
        if isinstance(raw_neighbors, dict):
            iterable_neighbors = raw_neighbors.items()  # (nbr, weight)
        else:
            iterable_neighbors = [(nbr, 1) for nbr in raw_neighbors]

        for nbr, w in iterable_neighbors:
            step_cost = w if isinstance(w, (int, float)) else 1
            tentative_g = g_cost[node] + step_cost
            improved = nbr not in g_cost or tentative_g < g_cost[nbr]
            if improved:
                # If using closed-set graph search without reopening, ignore improvements to closed nodes
                if not tree_search and nbr in closed and not reopen_inconsistent:
                    continue
                g_cost[nbr] = tentative_g
                predecessor[nbr] = node
                if not tree_search and nbr in closed and reopen_inconsistent:
                    # Reopen the node: remove from closed to allow re-expansion with better g
                    closed.remove(nbr)
                if nbr in coords:
                    h_nbr = (float(coords[nbr]) if direct_mode
                             else _distance(coords[nbr], coords[goal], heuristic))
                    f_nbr = tentative_g + h_nbr
                    heapq.heappush(open_heap, (f_nbr, h_nbr, nbr))
            elif tree_search:
                # In pure tree search duplicates always allowed: push even if not improved
                if nbr in coords:
                    h_nbr = (float(coords[nbr]) if direct_mode
                             else _distance(coords[nbr], coords[goal], heuristic))
                    f_nbr = tentative_g + h_nbr
                    heapq.heappush(open_heap, (f_nbr, h_nbr, nbr))

    if verbose:
        print("Nodes expanded (in order):", order)
        print("No route found.")
    return None


def print_a_star(graph, start, goal, coords, heuristic='l2', **kwargs):
    return a_star(graph, start, goal, coords, heuristic=heuristic, verbose=True, **kwargs)


# Convenience wrappers mirroring greedy_best_first variants

def a_star_raw(graph, start, goal, heuristics, verbose=False, **kwargs):
    return a_star(
        graph, start, goal, heuristics,
        heuristic='l2', verbose=verbose,
        tree_search=kwargs.get('tree_search', False),
        reopen_inconsistent=kwargs.get('reopen_inconsistent', False),
        max_expansions=kwargs.get('max_expansions')
    )

def a_star_tree_raw(graph, start, goal, heuristics, verbose=False, **kwargs):
    return a_star(graph, start, goal, heuristics, heuristic='l2', verbose=verbose, tree_search=True, **kwargs)


def a_star_l1(graph, start, goal, coords, verbose=False, **kwargs):
    # Force graph-search semantics by default
    return a_star(
        graph, start, goal, coords,
        heuristic='l1', verbose=verbose,
        tree_search=False,
        reopen_inconsistent=False,
        max_expansions=kwargs.get('max_expansions')
    )

def a_star_tree_l1(graph, start, goal, coords, verbose=False, **kwargs):
    return a_star(graph, start, goal, coords, heuristic='l1', verbose=verbose, tree_search=True, **kwargs)


def a_star_l2(graph, start, goal, coords, verbose=False, **kwargs):
    # Force graph-search semantics by default
    return a_star(
        graph, start, goal, coords,
        heuristic='l2', verbose=verbose,
        tree_search=False,
        reopen_inconsistent=False,
        max_expansions=kwargs.get('max_expansions')
    )

def a_star_tree_l2(graph, start, goal, coords, verbose=False, **kwargs):
    return a_star(graph, start, goal, coords, heuristic='l2', verbose=verbose, tree_search=True, **kwargs)


if __name__ == "__main__":
    demo_graph = {
        'S': {'A': 2, 'B': 5},
        'A': {'B': 1},
        'B': {'C': 4, 'D': 1},
        'C': {},
        'D': {'G': 2},
        'G': {}
    }
    demo_coords = {
        'S': (0, 0), 'A': (1, 1), 'B': (1, -1), 'C': (2, 1.5), 'D': (2, 0.2), 'G': (4, 0)
    }
    demo_h = {'S': 9.0, 'A': 6.5, 'B': 6.0, 'C': 3.0, 'D': 2.5, 'G': 0.0}
    print_a_star(demo_graph, 'S', 'G', demo_coords, heuristic='l2')
    print_a_star(demo_graph, 'S', 'G', demo_h, heuristic='l2')
