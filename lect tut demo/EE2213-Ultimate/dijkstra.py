from heapq import heappush, heappop
import math

def dijkstra(graph, s):
    node_data = {node: {'dist': math.inf, 'prev': []} for node in graph}
    X = set()  # processed/visited nodes
    Q = []  # frontier, implemented as a priority queue
    heappush(Q, (0, s))  # enqueue starting node (source) s with distance 0
    node_data[s]['dist'] = 0
    pop_order = []  # record the order nodes are popped (finalized)
    while Q:
        u_dist, u = heappop(Q)  # extractMin(Q)
        if u in X:
            continue
        X.add(u)  # mark u as processed
        pop_order.append(u)

        for v in graph[u]:
            if v not in X:
                dist = node_data[u]['dist'] + graph[u][v]
                if dist < node_data[v]['dist']:
                    node_data[v]['dist'] = dist  # update shortest distance
                    node_data[v]['prev'] = node_data[u]['prev'] + [u]
                    heappush(Q, (node_data[v]['dist'], v))  # updateQueue
    node_data['_pop_order'] = pop_order  # expose pop order for external printing
    return node_data



