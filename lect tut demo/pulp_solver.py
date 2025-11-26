# Import libraries
import matplotlib.pyplot as plt # For plotting
import numpy as np  # For numerical computations (used for plotting here)
from pulp import *  # PuLP: Python library for modeling linear programming problems

# Define the LP problem
prob = LpProblem("Maximize_Profit", LpMaximize)

# Decision variables
x1 = LpVariable("x_A", lowBound=0)   
x2 = LpVariable("x_B", lowBound=0)   
x3 = LpVariable("x_B", lowBound=0)   
x4 = LpVariable("x_B", lowBound=0)   
x5 = LpVariable("x_B", lowBound=0)   

# Objective function: Maximize profit
prob += 40*x1 + 75*x2 + 110*x3 + 140*x4 + 90*x5        # Total profit

# Constraints

prob +=x1 >= 0
prob +=x2 >= 0
prob +=x3 >= 0
prob +=x4 >= 0
prob +=x5 >= 0
prob +=8*x1 + 18*x2 + 25*x3 + 35*x4 + 20*x5 <= 600 # wood
prob +=4*x1 + 7*x2 + 10*x3 + 13*x4 + 8*x5 <= 250 # labour
prob +=x1 + 3*x2 + 5*x3 + 6*x4 + 4*x5 <= 120 # assembly
prob +=x1 >= 10
prob +=x2 <= 15
prob +=x3 + x4 + x5 >= 12
prob +=2*x5 - x1 >= 0

# Solve the LP problem
prob.solve() 

# Results
xA_opt = value(x1) # Get the optimal value of decision variable x_A
xB_opt = value(x2) # Get the optimal value of decision variable x_B
xC_opt = value(x3)
xD_opt = value(x4)
xE_opt = value(x5)
max_profit = value(prob.objective) # Get the maximum profit from the objective function

# Output results
print("Optimal number of units to produce:")
print("Product A:", xA_opt)
print("Product B:", xB_opt)
print("Product B:", xC_opt)
print("Product B:", xD_opt)
print("Product B:", xE_opt)
print("Maximum Profit: $", max_profit)


# Visualization (for illustration only, not required in exams)

x_vals = np.linspace(0, 50, 400)
x_B1 = 100 - 2 * x_vals
x_B2 = 40 - x_vals
x_B_min = 10

x_B_upper = np.minimum(x_B1, x_B2)
x_B_lower = np.maximum(x_B_min, 0)

feasible_x = []
feasible_y_lower = []
feasible_y_upper = []

for i in range(len(x_vals)):
    if x_B_upper[i] >= x_B_lower:
        feasible_x.append(x_vals[i])
        feasible_y_lower.append(x_B_lower)
        feasible_y_upper.append(x_B_upper[i])

# Profit lines
Z_values = [300, 700, 1200, 1500]
profit_lines = [(Z, (Z - 40 * x_vals) / 30) for Z in Z_values]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_vals, x_B1, label=r'$2x_A + x_B \leq 100$', color='green')
plt.plot(x_vals, x_B2, label=r'$x_A + x_B \leq 40$', color='blue')
plt.axhline(y=10, color='orange', linestyle='-', label=r'$x_B \geq 10$')

for Z, line in profit_lines:
    plt.plot(x_vals, line, linestyle=':', label=f'Profit = ${Z}')

plt.fill_between(feasible_x, feasible_y_lower, feasible_y_upper, color='gray', alpha=0.4, label='Feasible Region')
plt.plot(xA_opt, xB_opt, 'ro', label='Optimal Solution')
plt.text(xA_opt + 1, xB_opt + 1, f'({xA_opt:.0f}, {xB_opt:.0f})\nOptimal profit=${max_profit:.0f}', color='red')

plt.xlim((0, 50))
plt.ylim((0, 60))
plt.xlabel('Product A (x_A)')
plt.ylabel('Product B (x_B)')
plt.title('Solving and Visualizing LP with PuLP')
plt.legend()
plt.grid(True)
plt.show()
