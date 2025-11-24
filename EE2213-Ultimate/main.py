from EnterMetrics import EnterMetrics
from LinearRegression import linear_regression
from PolynomialRegression import polynomial_regression
from PolyRegressionClassification import polynomial_regression_with_classification
from RidgePolynomialRegression import ridge_poly_regression
from RidgeRegression import ridge_regression
from BinomialLogicticRegression import logistic_regression
from MultinomialLogisticRegression import multinomial_logistic_regression
from OneHotLinearClassification import onehot_linearclassification
from pearson_correlation import pearson_correlation
from bfs import bfs
from bfs import bfs_path
from dfs import dfs
from dfs import dfs_path
from dfs_recursive import dfs_recursive
from dijkstra import dijkstra
from greedy_best_first import (
    greedy_best_first,
    greedy_best_first_raw,
    greedy_best_first_l1,
    greedy_best_first_l2,
)
from a_star import (
    a_star,
    a_star_raw,
    a_star_l1,
    a_star_l2,
    a_star_tree_raw,
    a_star_tree_l1,
    a_star_tree_l2,
)
import numpy as np
from sklearn.metrics import mean_squared_error
from EnterMetrics import EnterMetrics

# no need to add column of 1s to X for regression. X_fitted does it already. First row of w is w0.
X=np.array(
    [[0.5, 1.2, -0.3],
     [-1, 0.8, 1.5],
     [2.3, -0.7, 0.5],
     [0, 1.5, -1.0]
   ]
);

# for one hot encoding (e.g. for multinomial logistic regression), simply type the class number 1, 2, 3...etc
# for binary logistic regression, assign label 1 to the class in interest
# the code will convert to onehot encoding internally
Y=np.array(
    [[1],
     [2],
     [3],
     [1]
     ]
);

# same dont add one column of 1s to X_test for regression
X_test=np.array(
    [[7, 10, 1]
    ]
)



X_fitted=np.hstack((np.ones((len(X),1)),X))
X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))
# linear_regression(X_fitted,Y, X_test_fitted)
# polynomial_regression(X,Y,order=3,X_test=X_test) #order=1 is linear regression
# polynomial_regression_with_classification(X, Y, order=1, X_test=X_test) #use this if linear regression with classification needed
# ridge_regression(X_fitted,Y,LAMBDA=0.1, X_test=X_test_fitted, form='auto') #linear model
# ridge_poly_regression(X, Y, LAMBDA=1, order=2, form='auto', X_test=X_test)

w_initial = np.array(
    [[0, 0, 0], 
     [0.01, -0.02, 0.03],
     [0.05, 0.04, -0.01],
     [-0.03, 0.02, 0.01]
    ]
)
# logistic_regression(X, Y, X_test, w_initial, learning_rate=0.1, num_iters=10000)
# multinomial_logistic_regression(X, Y, X_test, w_initial, learning_rate=0.1, num_iters=10000)



# onehot_linearclassification(X_fitted,Y,X_test_fitted)

# pearson_correlation(X,Y)

'''  
uninformed unweighted search algorithms (BFS, DFS)
    Define the unweighted graph using a dictionary. Each node maps to a list of its connected neighbors.
    bfs(unweighted_graph, start)
    dfs(unweighted_graph, start)
    dfs_recursive(unweighted_graph, node, visited)
    bfs_path(unweighted_graph, start, goal, verbose=True)
    dfs_path(unweighted_graph, start, goal, verbose=True)
'''
unweighted_graph = { # if undirected, each edge stored twice
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B', 'G'],
    'E': ['B', 'G', 'H'],
    'F': ['C', 'H'],
    'G': ['D', 'E', 'S'],
    'H': ['E', 'F', 'S'],
    'S': ['G', 'H']
}

# visit_sequence = bfs(unweighted_graph, 'A')
# visit_sequence = dfs(unweighted_graph, 'A')
# visit_sequence = dfs_recursive(unweighted_graph, 'A', [])
# print("Full_visit_sequence:", visit_sequence)

# shortest_route = bfs_path(unweighted_graph, 'A', 'B', verbose=True)
# shortest_route = dfs_path(unweighted_graph, 'A', 'B', verbose=True)

'''
uninformed weighted search algorithm (Dijkstra)
    Define the weighted graph as an adjacency list (dictionary of dictionaries), pair of(neighbour, weight) as edges 
    (each edge only stored once)
    dijkstra(weighted_graph, start)
    Returns shortest distances and paths from source to all other nodes.
    
Informed search algorithm (Greedy Best-First Search and A*)
    Define the graph as an adjacency list (dictionary of dictionaries). Each node maps to a dictionary of its connected neighbors and edge weights.
    Heuristics: admissible = 'l1' (Manhattan), 'l2' (Euclidean) using coordinates or direct heuristic values.
    All 3 GBFS returns each heuristics, path(S,G), hops, costs, traversed nodes.
    A* also returns g(n) costs and check for admissibility and consistency.
    Note that A* uses graph/tree search, need consistency/admissible respectively to guarantee optimality
'''
weighted_graph = { # if undirected, each edge stored twice. If no weights (GBFS), put dummy weights
    'S': {'A': 4, 'C':6,'D':3},
    'A': {'G': 6},
    'B': {'G':1},
    'C': {'D':2,'B':3,'G':3},
    'D': {'B':4},
    'G': {}
}
coords = { # [x, y] coordinates for each node, can do [x] only for 1D
    'S': [0, 0],
    'A': [1, 1],
    'B': [1, -1],
    'C': [2, 1.5],
    'G': [4, 0]
}
heuristics = {  # if not coord but direct heuristic values for each node (e.g., straight-line distance to goal)
    'S': 7.0,
    'A': 6.0,
    'B': 2.0,
    'C': 1.0,
    'G': 0.0
}

result = dijkstra(weighted_graph, 'S') # any but graphs with negative weights
for node in result: # Print shortest distances from the source to all other nodes
    dist = result[node]['dist']
    path = result[node]['prev'] + [node] # build the final path
    print("Shortest Distance to", node, ":", dist)
    print("Shortest Path to", node, ":", path)

# greedy_direct = greedy_best_first_raw(weighted_graph, 'S', 'G', heuristics=heuristics, verbose=True)
# greedy_l1 = greedy_best_first_l1(weighted_graph, 'S', 'G', coords=coords, verbose=True)
# greedy_l2 = greedy_best_first_l2(weighted_graph, 'S', 'G', coords=coords, verbose=True)

# astar_graph_direct = a_star_raw(weighted_graph, 'S', 'G', heuristics=heuristics, verbose=True)
# astar_graph_l1 = a_star_l1(weighted_graph, 'S', 'G', coords=coords, verbose=True)
# astar_graph_l2 = a_star_l2(weighted_graph, 'S', 'G', coords=coords, verbose=True)

# astar_tree_direct = a_star_tree_raw(weighted_graph, 'S', 'G', heuristics=heuristics, verbose=True)
# astar_tree_l1 = a_star_tree_l1(weighted_graph, 'S', 'G', coords=coords, verbose=True)
# astar_tree_l2 = a_star_tree_l2(weighted_graph, 'S', 'G', coords=coords, verbose=True)


'''
all has been adjacency list, if adjacency matrix needed, convert using:
# User input
n m
u v w
u v w

n, m = map(int, input().split())
graph = [[] for _ in range(n)]

for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))   # if undirected

'''