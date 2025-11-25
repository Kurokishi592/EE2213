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
from GradientDescent import GradientDescent
from ProjectedGradientDescent import SimpleProjectedGradientDescent
from k_means_cluster import custom_kmeans
from fuzzy_cmeans import fuzzy_Cmeans
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

# result = dijkstra(weighted_graph, 'S') # any but graphs with negative weights
# for node in result: # Print shortest distances from the source to all other nodes
#     dist = result[node]['dist']
#     path = result[node]['prev'] + [node] # build the final path
#     print("Shortest Distance to", node, ":", dist)
#     print("Shortest Path to", node, ":", path)

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
linear programming and convex problems 
- go to cvxpy_solver.py to solve any convex problem and run there.
- go to LP_2var_graph.py for graphical method with two decision variables (linear) and run there.
'''


'''
perform gradient descent (for multivariable functions) and unconstrained and differentiable convex
GradientDescent(f, f_prime, initial, learning_rate, num_iters)
    use lambda functions for f and f_prime, parameters: function (xyz: [0] refers to x, [1] refers to y, [2] refers to z ...)
    initial: initial values of all variables as a tuple
    [0]: steps at each iteration
    [1]: function values at each iteration
    [2]: gradient vectors at each iteration
if multiple variables, put PARTIAL derivative of each variable in f_prime return tuple (now optional)
(x,y,z) => (df/dx, df/dy, df/dz)

not as good, but can also go to gradient_descent_int to try with pytorch tensors for automatic differentiation
'''
learning_rate = 0.1
num_iters = 3

# print("Values of parameters at each step (first row is initial values, 2nd row first iter, post 1st gradient step): \n")
# print(GradientDescent(lambda xy:((xy[0]-1)**2 + (xy[1]-2)**2), None, (0,0), learning_rate, num_iters)[0], "\n")
# print("Function values at each step, first row uses init, 2nd row first iter, post 1st gradient step: \n")
# print(GradientDescent(lambda xy:((xy[0]-1)**2 + (xy[1]-2)**2), None, (0,0), learning_rate, num_iters)[1], "\n")
# print("Gradient vectors (partial derivatives) at each step (first row considered first iter, uses init): \n")
# print(GradientDescent(lambda xy:((xy[0]-1)**2 + (xy[1]-2)**2), None, (0,0), learning_rate, num_iters)[2], "\n")

# print("Values of parameters at each step (first row is initial values): \n")
# print(GradientDescent(lambda b:np.sin(np.exp(b))**2, None, 6, learning_rate, num_iters)[0], "\n")
# print("Function values at each step: \n")
# print(GradientDescent(lambda b:np.sin(np.exp(b))**2, None, 6, learning_rate, num_iters)[1], "\n")
# print("Gradient vectors (partial derivatives) at each step: \n")
# print(GradientDescent(lambda b:np.sin(np.exp(b))**2, None, 6, learning_rate, num_iters)[2], "\n")

# print("Values of parameters at each step (first row is initial values): \n")
# print(GradientDescent(lambda x:4*x**3, None, 2, learning_rate, num_iters)[0], "\n")
# print("Function values at each step: \n")
# print(GradientDescent(lambda x:4*x**3, None, 2, learning_rate, num_iters)[1], "\n")
# print("Gradient vectors (partial derivatives) at each step: \n")
# print(GradientDescent(lambda x:4*x**3, None, 2, learning_rate, num_iters)[2], "\n")

'''
Projected Gradient Descent for constrained but differentiable convex
    # Minimize (x-1)^2 + (y-2)^2 subject to x >= 0, y >= 0, x + y <= 2
    pgd_constraints = [([-1,0],0), ([0,-1],0), ([1,1],2)]  # a^T x <= b forms
    print("Projected Gradient Descent (with linear constraints) iteration log:\n")
    pgd_steps, pgd_vals, pgd_grads, pgd_proj, pgd_cvals = SimpleProjectedGradientDescent(
        lambda v: (v[0]-1)**2 + (v[1]-2)**2,
        pgd_constraints,
        initial=[3.0,-1.0],
        learning_rate=0.15,
        num_iters=5,
        verbose=True
    )
'''
pgd_constraints = [
    (1, 0, '>=', 0),   # x >= 0
    # (0, 1, '>=', 0),   # y >= 0
    # (1, 1, '<=', 2)    # x + y <= 2
]
# print("Projected Gradient Descent iteration log:\n")
# pgd_steps, pgd_vals, pgd_grads, pgd_proj, pgd_cvals = SimpleProjectedGradientDescent(
#     lambda x: ((x[0]-2)**2), # g(x) = (x-2)^2
#     pgd_constraints,
#     initial=[-1.0], # x0 = -1
#     learning_rate=0.15,
#     num_iters=5,
#     verbose=True
# )
# print("PGD Steps:\n", pgd_steps)
# print("PGD Objective values:\n", pgd_vals)
# print("PGD Constraint dot products per step (a_i^T x):\n", pgd_cvals)


''' 
no need to add column of 1s to X for regression. X_fitted does it already. First row of w is w0.
for correlation, row is sample, column is feature. (only use training set) Comparing each feature column to one target Y 
'''
X=np.array(
    [[-1],
     [0],
     [0.5],
     [0.3],
     [0.8]
   ]
);


'''
for binary logistic regression, assign label 1 to the class in interest
for binary classification, all functions are the same, key in -1 or 1 will do.
for multi-class classification only onehot and multi logistic has one hot (0, 1, 2), the rest need manual one hot
for correlation, row is samples. should be comparing to one target only so only 1 column
'''
Y=np.array(
    [[1],
     [1],
     [0],
     [1],
     [0]
     ]
);

''' same dont add one column of 1s to X_test for regression'''
X_test=np.array(
    [[-0.1]
    ]
)

''' only for linear, since for poly, the code will add the bias term internally (PolynomialFeatures from sklearn)'''
X_fitted=np.hstack((np.ones((len(X),1)),X))
X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))

''' first is used for regression task, or binary classification task, for custom binary threshold, use poly order=1'''
''' other is used for multi-category classification task (auto one hot, key in 0, 1, 2 ... for y) '''
# linear_regression(X_fitted,Y, X_test_fitted)
# onehot_linearclassification(X_fitted,Y,X_test_fitted)

'''
used for regression tasks, binary classification (can key in a float for binary threshold, default 0)
and multi-category classification tasks (manually one hot encode y for multi-category class 0 = [1, 0, 0], class 1 = [0, 1, 0], class 2 = [0, 0, 1] ...)
    also, bias are auto in this order for poly fit by sklearn
        (for poly order 3) Bias (1) x1 x2 x1^2 x1*x2 x2^2     x1^3 x1^2*x2 x1*x2^2 x2^3
    but in lecture/tutorial, it may be
        (for poly order 2) Bias (1) x1 x2 x1*x2 x1^2 x2^2
    so read qn carefuly may need to reorder rows of w if they ask for it (and watch P's columns) accordingly
'''
# polynomial_regression(X, Y, order=1, X_test=X_test, binary_threshold=None) #order=1 is linear regression
# ridge_regression(X_fitted,Y,LAMBDA=0.1, X_test=X_test_fitted, binary_threshold=None) #linear model
# ridge_poly_regression(X, Y, LAMBDA=1, order=1, X_test=X_test, binary_threshold=None)

'''
used for classification tasks only, one for binary logistic regression, one for multinomial logistic regression
- uses sigmoid and softmax function which is now about probability estimation
- for binary, use 1 to label the class in interest, other class use 0
- for multi-class, use (auto-one hot) 0, 1, 2 ...etc for classes
- uses gradient descent to train weights because CCE and LCE is not closed form
'''
w_initial = np.array(
    [[0.1],
     [-1]
    ]
)
# logistic_regression(X, Y, X_test, w_initial, learning_rate=0.1, num_iters=10000, binary_threshold=0.5)
# multinomial_logistic_regression(X, Y, X_test, w_initial, learning_rate=0.5, num_iters=10000)


'''
used for feature selection, to see which X features have high correlation with Y, to reduce dimension of X
    option 1: pick k features with highest pearson correlation values
    option 2: pick features with pearson correlation values above a threshold c
    k and c are magic numbers decided outside of this function
'''
# pearson_correlation(X,Y)


'''
perform kmeans clustering and fuzzy c-means clustering, can be 2D or higher dimensions
    returns converged centers and cluster labels for k means, membership matrix for fuzzy c-means. 
    fuzzier determines the level of cluster fuzziness (of a point overlapping into multiple clusters)
    Auto stop iterations upon convergence.
'''
x1 = np.array([5])
x2 = np.array([6])
x3 = np.array([9])
x4 = np.array([11])
x5 = np.array([13])
x6 = np.array([14])
x7 = np.array([15])
x8 = np.array([18])
x9 = np.array([18])
x10 = np.array([22])
x11 = np.array([23])
x12 = np.array([26])
data_points = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12])
c1_init = np.array([6])
c2_init = np.array([13.0])
c3_init = np.array([22.0])
centers_init = np.array([c1_init, c2_init, c3_init])

# custom_kmeans(data_points, centers_init, n_clusters=3, max_iterations=100)
# centers, W = fuzzy_Cmeans(data_points, centers_init, fuzzier=2, max_iterations=100, verbose=True)


'''
multi-layer perceptron (MLP) neural network
- Use MultilayerPerceptron(MLP).py directly for flexible MLP. Read comments and example usage
- Can refer to Activation_Functions.py and TwoLayerMLP for simpler versions

- ReLU a nonlinear activation function is usually used for hidden layers to avoid vanishing gradients into linear regression. Only to learn features.
- softmax is commonly used for the output layer in classification tasks to produce probability distributions, which sums to 1, positive values.
- linear activation is used for regression tasks to allow unbounded output values.
- sigmoid activation is used for binary classification tasks to produce probabilities between 0 and 1.
'''

