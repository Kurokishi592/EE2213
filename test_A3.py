import numpy as np
from A3_A0303203A import optimize_shipments, gradient_descent

def test_optimize_shipments():
    
    # Problem data
    supply = [50, 30, 40]  # China, India, Brazil
    demand = [20, 45, 25, 30]  # Singapore, US, Germany, Japan
    cost_matrix = [
        [10, 25, 30, 20],  # China to [SG, US, DE, JP]
        [12, 32, 25, 22],  # India to [SG, US, DE, JP]
        [35, 20, 15, 40]   # Brazil to [SG, US, DE, JP]
    ]
    
    try:
        minimal_cost, shipment_matrix = optimize_shipments(supply, demand, cost_matrix)
        
        print(f"Minimal cost: {minimal_cost}")
        print("Shipment matrix:")
        print(shipment_matrix)
        print(f"Supply used: {shipment_matrix.sum(axis=1)} (capacity: {supply})")
        print(f"Demand met: {shipment_matrix.sum(axis=0)} (required: {demand})")
        
        # Verify constraints
        supply_used = shipment_matrix.sum(axis=1)
        demand_met = shipment_matrix.sum(axis=0)
        
        print("\nConstraint verification:")
        print(f"Supply constraints satisfied: {all(supply_used[i] <= supply[i] for i in range(3))}")
        print(f"Demand constraints satisfied: {all(demand_met[j] == demand[j] for j in range(4))}")
        print(f"Non-negativity satisfied: {np.all(shipment_matrix >= 0)}")
        
    except Exception as e:
        print(f"Error in optimize_shipments: {e}")

def test_gradient_descent():
    """Test the gradient descent function"""
    print("\n=== Testing gradient_descent ===")
    
    learning_rate = 0.1
    num_iters = 10
    
    try:
        w_out, f_out = gradient_descent(learning_rate, num_iters)
        
        print(f"Learning rate: {learning_rate}")
        print(f"Number of iterations: {num_iters}")
        print(f"Initial w: 3.5")
        print(f"Target w (minimum): 5.0")
        
        print("\nIteration results:")
        for i in range(min(5, num_iters)):  # Show first 5 iterations
            print(f"Iter {i+1}: w = {w_out[i]:.4f}, f(w) = {f_out[i]:.4f}")
        
        if num_iters > 5:
            print("...")
            for i in range(max(5, num_iters-2), num_iters):  # Show last 2 iterations
                print(f"Iter {i+1}: w = {w_out[i]:.4f}, f(w) = {f_out[i]:.4f}")
        
        print(f"\nFinal w: {w_out[-1]:.4f} (should approach 5.0)")
        print(f"Final f(w): {f_out[-1]:.4f} (should approach 1.0)")
        
        # Verify arrays have correct length
        print(f"\nArray lengths: w_out={len(w_out)}, f_out={len(f_out)} (expected: {num_iters})")
        
    except Exception as e:
        print(f"Error in gradient_descent: {e}")

if __name__ == "__main__":
    test_optimize_shipments()
    test_gradient_descent()