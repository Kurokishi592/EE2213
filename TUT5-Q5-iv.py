import numpy as np

# Given data
X = np.array([[1, 0],
              [0, 1],
              [1, 1]])  
y = np.array([[2], [3], [5]]) 
w = np.array([[0], [0]])  # initial w
eta = 0.01  # learning rate

# Compute the gradient
grad = 2 * X.T @ (X @ w - y)

# Update w
w_new = w - eta * grad

print("\nGradient (∇C(w)):", grad)
print("\nUpdated w after the first iteration:", w_new)


