"""BFS utilities: traversal sequence and single-source shortest path.

Functions provided:
  bfs(graph, start) -> list visiting sequence (prints each visited node)
  bfs_sssp(graph, start, goal, verbose=False) -> shortest path list or None
    verbose=True prints: shortest path, minimum hops, traversal order until goal.
  print_bfs_sssp(graph, start, goal) -> convenience wrapper (verbose output)
"""

def bfs(graph, start):
    visited = []
    queue = [start]
    sequence = []
    visited.append(start)
    while queue:
        node = queue.pop(0)
        print(node)
        sequence.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return sequence


def _bfs_shortest_path(graph, start, goal, record_order=False):
    if start not in graph or goal not in graph:
        return None, [] if record_order else None
    visited = set([start])
    queue = [start]
    predecessor = {start: None}
    order = [] if record_order else None
    while queue:
        node = queue.pop(0)
        if record_order:
            order.append(node)
        if node == goal:
            route = []
            while node is not None:
                route.append(node)
                node = predecessor[node]
            route.reverse()
            return route, order if record_order else route
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                predecessor[neighbour] = node
                queue.append(neighbour)
    return None, order if record_order else None


def bfs_path(graph, start, goal, verbose=False):
    """Shortest path (unweighted) using BFS. Returns list or None.
    verbose=True also prints path, hop count, and traversal order."""
    route, order = _bfs_shortest_path(graph, start, goal, record_order=verbose)
    if verbose:
        if route:
            print("Shortest path from", start, "to", goal, ":", route)
            print("Minimum number of hops:", len(route) - 1)
            print("Nodes traversed (in order):", order)
        else:
            print("Nodes traversed (in order):", order)
            print("No route found.")
    return route


def print_bfs_path(graph, start, goal, path=None):  # compatibility helper
    return bfs_path(graph, start, goal, verbose=True)
