# Import libraries
import cvxpy as cp  # For solving linear programs

# 1. Define variables
x1 = cp.Variable(name="x1", integer=True)    # integer variable
x2 = cp.Variable(name="x2", integer=True)    # integer variable
x3 = cp.Variable(name="x3", integer=True)    # integer variable
x4 = cp.Variable(name="x4", integer=True)    # integer variable
x5 = cp.Variable(name="x5", integer=True)    # integer variable

# 2. Define objective
objective = cp.Maximize(2 * x1 - 3 * x2 + x3)

# 3. Define constraints
constraints = [
    x1 >= -15,       
    x1 <= 15,  
    x2 >= -15,       
    x2 <= 15, 
    x3 >= -15,       
    x3 <= 15,  
    x4 >= -15,       
    x4 <= 15,                
    x5 >= -15,       
    x5 <= 15,      
    x1 - x2 + x3 <=5,
    x1 - x2 + 4 * x3 <= 7,
    x1 + 2 * x2 - x3 + x4 <= 14,
    x3 - x4 + x5 <= 7                
]

# 4. Creates the CVXPY optimization problem by combining the objective and the constraints
prob = cp.Problem(objective, constraints)

# 5. Solve the LP problem
prob.solve() 

# 6. Results
x1_opt = x1.value # Get the optimal value of decision variable x_A
x2_opt = x2.value # Get the optimal value of decision variable x_B
x3_opt = x3.value # Get the optimal value of decision variable x_A
x4_opt = x4.value # Get the optimal value of decision variable x_B
x5_opt = x5.value # Get the optimal value of decision variable x_A
max_profit = prob.value # Get the maximum profit from the objective function
problem_status = prob.status # Get the solution status

# Output results
print("\nSolution Status:", problem_status)
print("Optimal number of units to purchase/sell:")
print("Stock 1:", x1_opt)
print("Product 2:", x2_opt)
print("Product 3:", x3_opt)
print("Product 4:", x4_opt)
print("Product 5:", x5_opt)
print("Maximum Profit: $", max_profit)

