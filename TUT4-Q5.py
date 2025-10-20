# Import libraries
import cvxpy as cp

# Decision variables: 1 if course is selected, 0 otherwise
x1 = cp.Variable(boolean=True, name='EE2213')
x2 = cp.Variable(boolean=True, name='MA2101')
x3 = cp.Variable(boolean=True, name='CDE2212')
x4 = cp.Variable(boolean=True, name='CE3201')
x5 = cp.Variable(boolean=True, name='EE3801')

# Objective function: maximize total benefit
objective = cp.Maximize(9*x1 + 7*x2 + 8*x3 + 5*x4 + 6*x5)

# Constraints
constraints = [
    4*x1 + 4*x2 + 4*x3 + 2*x4 + 3*x5 <= 20,   # Credit limit
    10*x1 + 9*x2 + 11*x3 + 5*x4 + 7*x5 <= 40, # Workload limit
    x1 + x2 <= 1,                             # Time conflict
    x1 + x2 + x3 + x4 + x5 <= 3               # Limit on total number of courses
]

# Define the problem
prob = cp.Problem(objective, constraints)

# Solve the problem
prob.solve()  

# Output results
status_text = prob.status

print("Solver Status:", status_text)

# Show selected courses if solution is optimal
if status_text == "optimal":
    print("Selected courses:")
    for var in [x1, x2, x3, x4, x5]:
        if var.value == 1:
            print(var.name(), "is selected")
    print("Total benefit:", prob.value)
elif status_text == "unbounded":
    print("The problem is unbounded. Add more constraints.")
elif status_text == "infeasible":
    print("The problem has no feasible solution.")
else:
    print("Solver returned an undefined status. Please check your model.")

