import cvxpy as cp # For solving linear programs

# 1. Define decision variables (cp.Variable(name="x", integer=True) by default. add boolean=True for binary variables only take 1 or 0)
x1 = cp.Variable(name="product_1")
x2 = cp.Variable(name="product_2")
x3 = cp.Variable(name="product_3")
x4 = cp.Variable(name="product_4")
x5 = cp.Variable(name="product_5")

# 2. Define the objective function (replace function or change to cp.Minimize/cp.Maximize as needed)
objective = cp.Maximize(2 * x1 -3 * x2 + x3)

# 3. Define constraints (replace as needed, use == for equality) - each var shld be constrained too. Usually >=0
constraints = [
    x1 - x2 + x3 <= 5,
    x1 - x2 + 4 * x3 <= 7,
    x1 + 2* x2 - x3 + x4 <= 14,
    x3 - x4 + x5 <= 7,
    x1 >= -15,
    x2 >= -15,
    x3 >= -15,
    x4 >= -15,
    x5 >= -15,
    x1 <= 15,
    x2 <= 15,
    x3 <= 15,
    x4 <= 15,
    x5 <= 15
]

# 4. Create the CVXPY optimisation problem and solve it
problem = cp.Problem(objective, constraints)
problem.solve()

# 5. Print the results
print(f"Status: {problem.status}")
print(f"Optimal value: {problem.value}")
print(f"Optimal x: {x1.value}")
print(f"Optimal y: {x2.value}")


