# Define DFS function using stack
def dfs(graph, start):
    visited = [] # List to keep track of visited nodes
    stack = [] # Stack to hold nodes to explore
    sequence = [] # List to store the visiting sequence
    
    visited.append(start)
    stack.append(start)
    
    while stack:
        node = stack.pop()
        print(node)
        sequence.append(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.append(neighbour)
                stack.append(neighbour)
                
    return sequence


def dfs_path(graph, start, goal, verbose=False):
    """
    Depth-First Search path from start to goal (not guaranteed shortest).
    Accepts a graph as a dictionary: node -> list of neighbors.
    Returns a list of nodes from start to goal (inclusive), or None if unreachable.

    Set verbose=True to print the path and number of nodes traversed.
    """
    if start not in graph or goal not in graph:
        route = None
        if verbose:
            print("No route found.")
        return route

    visited = set([start])
    stack = [start]
    predecessor = {start: None}
    order = []  # traversal order until goal discovered

    while stack:
        node = stack.pop()
        order.append(node)
        if node == goal:
            route = []
            while node is not None:
                route.append(node)
                node = predecessor[node]
            route.reverse()
            if verbose:
                print("Shortest path from", start, "to", goal, ":", route)
                print("Minimum number of hops:", len(route) - 1)
                print("Nodes traversed (in order):", order)
            return route
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                predecessor[neighbour] = node
                stack.append(neighbour)

    if verbose:
        print("Nodes traversed (in order):", order)
        print("No route found.")
    return None


def print_dfs_path(graph, start, goal, path=None):
    # Printing helper delegating to dfs_path
    return dfs_path(graph, start, goal, verbose=True)
