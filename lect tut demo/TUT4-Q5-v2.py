# Import libraries
from pulp import *

# Define the problem
prob = LpProblem("Course_Selection", LpMaximize)

# Decision variables: 1 if course is selected, 0 otherwise
x1 = LpVariable('EE2213', cat='Binary') 
x2 = LpVariable('MA2101', cat='Binary')  
x3 = LpVariable('CDE2212', cat='Binary') 
x4 = LpVariable('CE3201', cat='Binary')  
x5 = LpVariable('EE3801', cat='Binary')  

# Objective function: maximize total benefit
prob += 9*x1 + 7*x2 + 8*x3 + 5*x4 + 6*x5 # Total benefit

# Constraints
prob += 4*x1 + 4*x2 + 4*x3 + 2*x4 + 3*x5 <= 20   # Credit_Limit"
prob += 10*x1 + 9*x2 + 11*x3 + 5*x4 + 7*x5 <= 40 # Workload_Limit
prob += x1 + x2 <= 1 # Time conflict: EE2213 and MA2101 have the same time slot
prob += x1 + x2 + x3 + x4 + x5 <= 3 # Limit on total number of courses

# Solve it
prob.solve()

# Output results

# Check and print solver status
status_code = prob.status               # Numeric code (e.g., 1 for Optimal)
status_text = LpStatus[status_code]     # Convert to readable text (e.g., 'Optimal')

print("Solver Status:", status_text)    # Display the outcome

# Show selected courses if solution is optimal
if status_text == "Optimal":
    print("Selected courses:")
    for var in prob.variables():
        if var.varValue == 1:
            print(var.name, "is selected")
    print("Total benefit:", value(prob.objective))
elif status_text == "Unbounded":
    print("The problem is unbounded. Add more constraints.")
elif status_text == "Infeasible":
    print("The problem has no feasible solution.")
else:
    print("Solver returned an undefined status. Please check your model.")



