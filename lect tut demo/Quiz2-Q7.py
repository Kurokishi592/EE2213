# Import libraries
import cvxpy as cp  # For solving linear programs

# 1. Define variables
x_1 = cp.Variable()  # defines a continuous decision variable x_1.
x_2 = cp.Variable()  # defines a continuous decision variable x_2.

# 2. Define objective
objective = cp.Maximize(3 * x_1 + 2 * x_2)

# 3. Define constraints
constraints = [
    x_1 >= 0,          
    x_2 >= 0,          
    x_1 + x_2 <= 5,  
    2 * x_1 + x_2 >= 12, 
    -x_1 + 2 * x_2 <= 4,     
    x_1 - x_2 <= 3
]

# 4. Create a CVXPY problem object
prob = cp.Problem(objective, constraints)

# 5. Solve the LP problem
prob.solve() 

# 6. Results
x1_opt = x_1.value  # Get the optimal value of decision variable x_1
x2_opt = x_2.value  # Get the optimal value of decision variable x_2
min_obj = prob.value # Get the minimum from the objective function
problem_status = prob.status # Get the solution status

# Output results
print("\nSolution Status:", problem_status)
print("x1:", x1_opt)
print("x2:", x2_opt)
print("Minimum objective value:", min_obj)
