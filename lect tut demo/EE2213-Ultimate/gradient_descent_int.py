import torch

# Define the function you want to minimize
# Let's use a simple convex function: f(x) = x^2
def f(x):
    return (x+2)**2+7

# Initialize x (as a PyTorch tensor with gradient tracking)
x = torch.tensor([4.0], requires_grad=True)

# Set learning rate and number of iterations
learning_rate = 0.01
iterations = 4

print("Starting gradient descent...\n")
for i in range(iterations):
    # Compute the function value
    y = f(x)

    # Compute the gradient
    y.backward()

    # Perform a gradient descent step (manual update)
    with torch.no_grad():
        x -= learning_rate * x.grad  # x.grad is the gradient of y w.r.t. x

    # Zero the gradients so they don't accumulate
    x.grad.zero_()

    # Round values for nicer display
    x_value = round(x.item(), 9)
    y_value = round(f(x).item(), 9)
    print(f"Iteration {i+1}: x = {x_value:.8f}, f(x) = {y_value:.8f}")