# Recursive DFS function
def dfs_recursive(graph, node, visited):
    print(node)
    visited.append(node)
    for neighbour in graph.get(node, []):
        if neighbour not in visited:
            dfs_recursive(graph, neighbour, visited)
    return visited
