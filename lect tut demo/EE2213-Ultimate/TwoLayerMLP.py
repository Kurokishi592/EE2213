import numpy as np

X = np.array([[1,0.5, 1.2, -0.3],[1,-1, 0.8, 1.5], [1,2.3,-0.7,0.5],[1,0,1.5,-1]]) # 2 samples, 3 input features
y = np.array([[1],[2],[3],[1]])

# original weights
W1=np.array([[0,0,0],[0.01,-0.02,0.03],[0.05,0.04,-0.01],[-0.03,0.02,0.01]]) # 2 neurons for the 1st layer
W2=np.array([[0,0,0],[0.01,-0.02,0.03],[0.05,0.04,-0.01],[-0.03,0.02,0.01]]) # 1 neuron for the output layer

lr=0.1


def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(o):
    return (o > 0).astype(float)
  
def squared_error_loss_derivative(y, y_hat):
    return 2*(y_hat-y)

def forward_pass(X, W1, W2):
    """
    INPUT:
    X: Input data, dimensions of n_samples x n_features
    W1: Weights for the first layer, dimensions of n_features x n_neurons1
    W2: Weights for the second layer, dimensions of n_neurons1 x n_neurons2

    OUTPUT:
    y_hat: Predicted output, dimensions of n_samples x n_neurons2
    A2: Inputs to the second layer, dimensions of n_samples x (n_neurons1+1)
    O1: Outputs at the first layer, dimensions of n_samples x n_neurons1
    """
    # first layer output
    O1=ReLU(X @ W1)

    # second layer output
    # Column of 1s for bias term
    ones = np.ones((O1.shape[0], 1))
    # Concatenate along columns (axis=1)
    A2 = np.hstack((ones, O1))
    y_hat = A2 @ W2

    return y_hat, A2, O1

def backward_pass_output(y, y_hat, A_o, W_o, lr):
    """
    INPUT:
    y: ground truth, dimensions of n_samples x n_neurons
    y_hat: outputs of the output layer
    A_o: input to the output layer, dimensions of n_samples x (n_neurons+1)
    W_o: weight at output layer
    lr: learning rate

    OUTPUT:
    E_o: Error at output layer
    G_o: Gradient at output layer
    W_o_new: Updated weights at output layer
    """
    N = y.shape[0]

    # Error at output layer
    E_o = squared_error_loss_derivative(y, y_hat)
    # Gradient at output layer
    G_o = (A_o.T @ E_o)/N
    # Weights update at output layer
    W_o_new = W_o - lr * G_o

    return E_o, G_o, W_o_new

def backward_pass_hidden(E_ladd1, W_ladd1, A_l, O_l, W_l, lr):
    """
    INPUT:
    E_ladd1: Error at layer l+1, dimensions of n_samples x n_neurons_(l+1)
    W_ladd1: Weights at layer l+1, dimensions of (n_neurons_l+1) x n_neurons_(l+1)
    A_l: inputs to layer l, dimensions of n_samples x (n_neurons_l+1)
    O_l: Outputs at layer l, dimensions of n_samples x n_neurons_l
    W_l: Weights at layer l, dimensions of (n_neurons_(l-1)+1) x n_neurons_l
    lr: Learning rate

    OUTPUT:
    E_l: Error at hidden layer l, dimensions of n_samples x n_neurons_l
    G_l: Gradient at hidden layer l, dimensions of (n_neurons_(l-1)+1) x n_neurons_l
    W_l_new: Updated weights at hidden layer l, dimensions of (n_neurons_(l-1)+1) x n_neurons_l
    """
    N = A_l.shape[0]
    # Error at hidden layer l
    E_l = E_ladd1 @ W_ladd1[1:].T * ReLU_derivative(O_l)  # ReLU derivative
    # Gradient at hidden layer l
    G_l = (A_l.T @ E_l)/N
    # Weights update at hidden layer l
    W_l_new = W_l - lr * G_l

    return E_l, G_l, W_l_new




y_hat, A2, O1=forward_pass(X, W1, W2)
print(f'Predicted output is {y_hat}')
print(f'Input to output layer A2 is {A2}')

E2, G2, W2_new = backward_pass_output(y, y_hat, A2, W2, lr)
print(f'Error at output layer E2 is {E2}')
print(f'Gradient at output layer G2 is {G2}')
print(f'Updated W2 is {W2_new}')

E1, G1, W1_new = backward_pass_hidden(E2, W2, X, O1, W1, lr)
print(f'Error at hidden layer E1 is {E1}')
print(f'Gradient at hidden layer G1 is {G1}')
print(f'Updated W1 is {W1_new}')