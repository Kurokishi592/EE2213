import numpy as np

# This function will take in a function, compute the gradient of the cost function and perform gradient descent
# can do for multiple iterations if needed, and print at every iteration the gradient, current w and new w

# Given data, where w is the convex function, e.g. w = x^2
x = np.array([1, 2, 3, 4, 5])
w = x**2
w_new = np  # initial w
eta = 0.01  # learning rate
iterations = 1  # number of iterations
grad = np.array


for _ in range(iterations):
    # Compute the gradient
    grad = 2 * w
    # Update w
    w = w - eta * grad
    w_new = w

print("current w:", w)
print("\nGradient (∇C(w)):", grad)
print("\nUpdated w after the first iteration:", w_new)


