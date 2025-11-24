import cvxpy as cp
import numpy as np

# Given data
X = np.array([[1, 0],
              [0, 1],
              [1, 1]])  
y = np.array([[2],
              [3],
              [5]])    

# Define variable
w = cp.Variable((2,1))  # w is a 2x1 column vector

# Define the objective function
objective = cp.Minimize(cp.sum_squares(X @ w - y))

# Define problem
problem = cp.Problem(objective)

# Solve
problem.solve()

# Results
print("\nOptimal weights w:")
print(w.value)
print("\nMinimum cost:", problem.value)


