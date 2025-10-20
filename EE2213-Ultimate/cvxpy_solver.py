import cvxpy as cp # For solving linear programs

# 1. Define decision variables (cp.Variable(name="x", integer=True) by default. add boolean=True for binary variables only take 1 or 0)
x_1 = cp.Variable(name="product_1")
x_2 = cp.Variable(name="product_2")
x_3 = cp.Variable(name="product_3")
x_4 = cp.Variable(name="product_4")
x_5 = cp.Variable(name="product_5")

# 2. Define the objective function (replace function or change to cp.Minimize/cp.Maximize as needed)
objective = cp.Maximize(2 * x_1 -3 * x_2 + x_3)

# 3. Define constraints (replace as needed, use == for equality)
constraints = [
    x_1 - x_2+x_3<= 5,
    x_1-x_2+4*x_3 <= 7,
    x_1 + 2* x_2-x_3+x_4<=14,
    x_3-x_4+x_5<=7,
    x_1 >= -15,
    x_2 >= -15,
    x_3 >= -15,
    x_4 >= -15,
    x_5 >= -15,
    x_1 <= 15,
    x_2 <= 15,
    x_3 <= 15,
    x_4 <= 15,
    x_5 <= 15
]

# 4. Create the CVXPY optimisation problem and solve it
problem = cp.Problem(objective, constraints)
problem.solve()

# 5. Print the results
print(f"Status: {problem.status}")
print(f"Optimal value: {problem.value}")
print(f"Optimal x: {x_1.value}")
print(f"Optimal y: {x_2.value}")


