import numpy as np
import cvxpy as cp

# Please replace "StudentMatriculationNumber" with your actual matric number in the filename
# Please do NOT change the function names in this file.
# Filename should be: A3_StudentMatriculationNumber.py (replace “StudentMatriculationNumber” with your own your student matriculation number).

def optimize_shipments(supply, demand, cost_matrix):
    """
    Problem 2: Logistics Optimization
    
    Inputs:
    :supply: list of int
        List of factory capacities [China, India, Brazil]
    :demand: list of int
        List of market demands [Singapore, US, Germany, Japan]
    :cost_matrix: 2D list (3x4)
        3x4 matrix where cost_matrix[i][j] is cost from factory i to market j
        Rows correspond to factories [China, India, Brazil].
        Columns correspond to markets [Singapore, US, Germany, Japan].
        
    Returns:
    :minimal_cost: float
        The total minimized transportation cost.
    :shipment_matrix: numpy.ndarray
        3x4 array of integers where shipment_matrix[i, j] is units 
        shipped from factory i to market j.
        Rows correspond to factories [China, India, Brazil].
        Columns correspond to markets [Singapore, US, Germany, Japan].
    """
    # Your ILP formulation and solution code goes here

    # Convert inputs to numpy arrays since cost and shipment will be in array form
    supply = np.asarray(supply)
    demand = np.asarray(demand)
    cost = np.asarray(cost_matrix)

    # 1. Define decision variables: integer shipments from each factory (i) to each market (j)
    shipments = cp.Variable((3, 4), integer=True)

    # 2. Define the objective function (minimize total transportation cost)
    objective = cp.Minimize(cp.sum(cp.multiply(cost, shipments)))
    
    # 3. Define the constraints
    constraints = []
    # Non-negativity (whole units where integer=True and >= 0 in general)
    constraints.append(shipments >= 0)
    # Supply constraints: row sums cannot exceed capacity
    constraints += [cp.sum(shipments[i, :]) <= supply[i] for i in range(3)]
    # Demand constraints: column sums must meet demand exactly
    constraints += [cp.sum(shipments[:, j]) == demand[j] for j in range(4)]

    # 4. Create the CVXPY optimisation problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Extract results; round to nearest int to guard against tiny numerical noise, then cast to int
    minimal_cost = problem.value                                            # Get the minimal transportation cost
    shipment_matrix = shipments.value.astype(int)  

    return minimal_cost, shipment_matrix


def gradient_descent(learning_rate, num_iters):
    """
    Problem 2: Gradient Descent

    Inputs:
    :learning_rate: float
        The learning rate for gradient descent. Value between 0 and 0.2.
    :num_iters: int
        Number of gradient descent iterations.

    Returns:
    :w_out: numpy.ndarray
        Array of length num_iters containing updated w values at each step.
    :f_out: numpy.ndarray
        Array of length num_iters containing f(w) = 1 + (w - 5)^2 at each step.
    """

    # Initialization of learning rate, value of w and f at each iteration
    w = 3.5
    w_out = np.zeros(num_iters)
    f_out = np.zeros(num_iters)

    # Your gradient descent code goes here
    # Define the objective function
    def f(w):
        return 1 + (w - 5) ** 2
    
    # Define the derivative of the objective function
    def df_dw(w):
        return 2 * (w - 5)
    
    # Gradient descent iterations
    for i in range(num_iters):
        # Update w using gradient descent: w = w - learning_rate * gradient
        w -= learning_rate * df_dw(w)
        
        # Store the updated w and f(w) values
        w_out[i] = w
        f_out[i] = f(w)

    return w_out, f_out
