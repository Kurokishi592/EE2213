import cvxpy as cp # For solving any convex optimisation problems
import numpy as np

''' 1. Define decision variables (cp.Variable(name="x", integer=True) by default. add boolean=True for binary variables only take 1 or 0)'''
x1 = cp.Variable(name="product_1")
x2 = cp.Variable(name="product_2")
# x3 = cp.Variable(name="product_3")
# x4 = cp.Variable(name="product_4")
# x5 = cp.Variable(name="product_5")

# X = np.array([[1,0],
#               [0,1],
#               [1,1]])  # example feature matrix
# y = np.array([[2],
#               [3],
#               [5]])  # example target vector

# x1 = cp.Variable(boolean=True, name='EE2213')  # 1 if course is taken, 0 otherwise
# x2 = cp.Variable(boolean=True, name='EE2026')
# x3 = cp.Variable(boolean=True, name='MA1512')
# x4 = cp.Variable(boolean=True, name='CS1010')
# x5 = cp.Variable(boolean=True, name='MA1512')

''' 2. Define the objective function (replace function or change to cp.Minimize/cp.Maximize as needed)'''
objective = cp.Maximize(40*x1 + 30*x2)
# objective = cp.Minimize(cp.sum_squares(X@w-y))  # example of least squares minimization
# objective = cp.Maximize(9*x1 + 7*x2 + 8*x3 + 5*x4 + 6*x5)  # maximise benefit score tagged to each course

''' 3. Define constraints (replace as needed, use == for equality) - each var shld be constrained too. Usually >=0'''
constraints = [
#     x1 - x2 + x3 <= 5,
#     x1 - x2 + 4 * x3 <= 7,
#     x1 + 2* x2 - x3 + x4 <= 14,
#     x3 - x4 + x5 <= 7,
    x1 >= 0,
    x2 >= 0,
    2*x1 + x2 <= 100,
    x2 >= 10,
    x1 + x2 <= 40
#     x3 >= -15,
#     x4 >= -15,
#     x5 >= -15,
#     x1 <= 15,
#     x2 <= 15,
#     x3 <= 15,
#     x4 <= 15,
#     x5 <= 15
]

# constraints = [ 
#         4*x1+4*x2+4*x3+2*x4+3*x5 <= 20,  # max workload constraint
#         10*x1 + 9*x2 + 11*x3 + 5*x4 + 7*x5 <= 40,  # max weekly workload
#         x1 + x2 <= 1,  # cannot take both EE2213 and EE2026 together
#         x1 + x2 + x3 + x4 + x5 <= 2,  # max 3 courses
# ]

''' 4. Create the CVXPY optimisation problem and solve it '''
problem = cp.Problem(objective, constraints)
problem.solve()

''' 5. Print the results '''
print(f"Status: {problem.status}")
print(f"Optimal value: {problem.value}")
print(f"Optimal x: {x1.value}")
print(f"Optimal y: {x2.value}")

# for course selection example
# if problem.status == "optimal":
#     print("Optimal course selection:")
#     for var in [x1, x2, x3, x4, x5]:
#         if var.value >= 0.99:  # since boolean vars may not be exactly 1 due to solver tolerance
#             print(f"Take course: {var.name()}")
#     print(f"Maximum benefit score: {problem.value}")
# elif problem.status == "infeasible":
#     print("No feasible course selection found under the given constraints.")
# elif problem.status == "unbounded":
#     print("The problem is unbounded.")
# else:
#     print(f"Solver ended with status: {problem.status}")
