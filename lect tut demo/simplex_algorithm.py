import numpy as np

def simplex(c, A, b):
    """
    Solves a linear programming problem using the Simplex algorithm.

    This function is designed to solve problems in a specific format (standard form):
    Maximize: c^T * x
    Subject to: A * x <= b
                x >= 0

    Args:
        c (np.array): A 1D NumPy array representing the coefficients of the
                      objective function (the function you want to maximize).
                      Example: For P = 4x + 3y, c would be [4, 3].
        A (np.array): A 2D NumPy array (matrix) representing the coefficients
                      of the variables in the constraints.
                      Example: For 3x + 2y <= 6, a row in A would be [3, 2].
        b (np.array): A 1D NumPy array for the right-hand side of each constraint.
                      Example: For 3x + 2y <= 6 and 2x + 3y <= 6, b would be [6, 6].

    Returns:
        tuple: A tuple containing the optimal solution, optimal value, and status.
               - solution (np.array): The values of the variables (x, y, etc.)
               - value (float): The maximum value of the objective function.
               - status (str): A string indicating the result ('Optimal', 'Unbounded', 'Infeasible').
    """
    
    # --- Step 1: Convert to Standard Form and Create the Tableau ---
    # The Simplex algorithm works with a special matrix called a "tableau".
    # This tableau represents the entire problem (objective function and all constraints) in a single, organized format. 
    # To create it, we first need to convert our inequality constraints (<=) into equalities (=) by adding "slack variables".
    # For each constraint, we add a new variable that represents the unused amount of the resource.
    # Example: 3x + 2y <= 6 becomes 3x + 2y + s1 = 6, where s1 is the slack variable.

    # Get the number of variables (x, y, etc.) and constraints from the input arrays.
    num_vars = len(c)
    num_constraints = len(b)

    # Create an empty matrix (tableau) filled with zeros.
    # The dimensions are:
    # - Rows: One for each constraint + one for the objective function.
    # - Columns: One for each original variable + one for each slack variable
    #            + one for the right-hand side (RHS) of the equations.
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    
    # Fill the tableau with the problem data:
    
    # Fill in the coefficients for the original variables and slack variables.
    # The top-left part of the tableau gets the A matrix.
    tableau[:num_constraints, :num_vars] = A
    # The middle part gets an identity matrix for the slack variables.
    tableau[:num_constraints, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    # The right-most column gets the b array (the RHS values).
    tableau[:num_constraints, -1] = b
    
    # Fill in the objective function row at the bottom of the tableau.
    # We negate the coefficients of the objective function because the Simplex
    # algorithm's goal is to make all these values non-negative.
    tableau[-1, :num_vars] = -c
    
    print("Initial Tableau:")
    print(tableau)
    print("--------------------------------------------------")

    # --- Step 2: The Main Simplex Algorithm Loop ---
    # The algorithm works by repeatedly "pivoting" until it finds the optimal solution.
    # Each pivot moves from one corner of the feasible region to a better, adjacent one.
    while True:
        # Step A: Check for optimality
        # We are at the optimal solution when there are no more negative numbers in the objective function row (the last row)
        # because it means we cannot increase the objective value any further.
        if np.all(tableau[-1, :-1] >= 0):
            # If all are non-negative, we've found the optimal solution!
            solution = np.zeros(num_vars)
            
            # Read the final values for the original variables from the tableau.
            for i in range(num_constraints):
                pivot_col = np.where(tableau[i, :num_vars] == 1)[0]
                if len(pivot_col) == 1:
                    solution[pivot_col[0]] = tableau[i, -1]
                    
            # Calculate the final optimal value using the original objective function.
            optimal_value = np.dot(c, solution)
            return solution, optimal_value, "Optimal"

        # Step B: Select the Pivot Column (Entering Variable)
        # We choose the column with the most negative number in the objective row.
        # This represents the variable that, if increased, will improve the objective
        # function the most.
        pivot_col = np.argmin(tableau[-1, :-1])
        
        # Check for an unbounded solution.
        # An unbounded solution occurs if all values in the pivot column are zero or negative.
        # This means we can increase the entering variable infinitely without violating
        # any constraints.
        if np.all(tableau[:-1, pivot_col] <= 0):
            return None, None, "Unbounded"
            
        # Step C: Select the Pivot Row (Leaving Variable)
        # We use the "Minimum Ratio Test" to find the pivot row.
        # This test prevents us from jumping outside the feasible region.
        # We calculate the ratio of the RHS value to the pivot column value for each row.
        ratios = np.zeros(num_constraints)
        for i in range(num_constraints):
            # We only consider rows with a positive value in the pivot column.
            if tableau[i, pivot_col] > 0:
                ratios[i] = tableau[i, -1] / tableau[i, pivot_col]
            else:
                ratios[i] = np.inf # Use infinity for non-positive values to ignore them.
        
        # The pivot row is the one with the smallest positive ratio.
        pivot_row = np.argmin(ratios)
        
        # Step D: The Pivot Operation
        # The pivot is the single element at the intersection of the pivot row and column.
        # This element is the key to transforming the tableau.
        # We perform row operations to make the pivot element 1 and all other elements in the pivot column 0. 
        # This process is like moving from one corner point of the feasible region to a better, adjacent one.
        
        # Normalize the pivot row (divide the entire row by the pivot element).
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        
        # Use the normalized pivot row to zero out the other rows in the pivot column.
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
        
        print(f"\nTableau after pivoting on element at ({pivot_row}, {pivot_col}):")
        print(tableau)
        print("--------------------------------------------------")

# --- Example Usage ---

if __name__ == '__main__':
    # This is the example problem from the prompt:
    # Maximize P = 4x + 3y
    # Subject to:
    # 3x + 2y <= 6
    # 2x + 3y <= 6
    # x, y >= 0

    # The coefficients of the objective function (P = 4x + 3y), below we use the concise format
    c = np.array([4, 3])

    # The coefficients of the constraints
    A = np.array([
        [3, 2],
        [2, 3]
    ])

    # The right-hand side of the constraints
    b = np.array([6, 6])
    
    # Call the simplex function with the problem data
    solution, value, status = simplex(c, A, b)
    
    print("\n--- Final Solution for the Main Problem ---")
    if status == 'Optimal':
        # If the solution is optimal, print the results.
        print(f"Status: {status}")
        print(f"Optimal Solution (x, y): {solution}")
        print(f"Optimal Value: {value}")
    else:
        # If the solution is not optimal (e.g., unbounded), print the status.
        print(f"Status: {status}")

    # Another example to demonstrate an unbounded problem
    print("\n--- Unbounded Example ---")
    # Maximize x + y
    # Subject to: x - y <= 1
    # This problem has an open feasible region, so it's unbounded.
    c_unbounded = np.array([1, 1])
    A_unbounded = np.array([[1, -1]])
    b_unbounded = np.array([1])
    solution_unbounded, value_unbounded, status_unbounded = simplex(c_unbounded, A_unbounded, b_unbounded)
    print(f"Status: {status_unbounded}")
