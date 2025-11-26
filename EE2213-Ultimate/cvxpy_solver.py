import cvxpy as cp # For solving any convex optimisation problems
import numpy as np

''' 1. Define decision variables (cp.Variable(name="x", integer=True) by default. add boolean=True for binary variables only take 1 or 0)'''
x1 = cp.Variable(name="product_1", integer=True) # chairs
x2 = cp.Variable(name="product_2", integer=True) # tables
x3 = cp.Variable(name="product_3", integer=True) # desks
x4 = cp.Variable(name="product_4", integer=True) # cabinet
x5 = cp.Variable(name="product_5", integer=True) # bookcases

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
objective = cp.Maximize(40*x1 + 75*x2 + 110*x3 + 140*x4 + 90*x5)
# objective = cp.Minimize(cp.sum_squares(X@w-y))  # example of least squares minimization
# objective = cp.Maximize(9*x1 + 7*x2 + 8*x3 + 5*x4 + 6*x5)  # maximise benefit score tagged to each course

''' 3. Define constraints (replace as needed, use == for equality) - each var shld be constrained too. Usually >=0'''
constraints = [
    x1 >= 0,
    x2 >= 0,
    x3 >= 0,
    x4 >= 0,
    x5 >= 0,
    8*x1 + 18*x2 + 25*x3 + 35*x4 + 20*x5 <= 600, # wood
    4*x1 + 7*x2 + 10*x3 + 13*x4 + 8*x5 <= 250, # labour
    x1 + 3*x2 + 5*x3 + 6*x4 + 4*x5 <= 120, # assembly
    x1 >= 10,
    x2 <= 15,
    x3 + x4 + x5 >= 12,
    2*x5 - x1 >= 0
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
print(f"Optimal x1: {x1.value}")
print(f"Optimal x2: {x2.value}")
print(f"Optimal x3: {x3.value}")
print(f"Optimal x4: {x4.value}")
print(f"Optimal x5: {x5.value}")

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
