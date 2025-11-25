import numpy as np
"""
A minimal, general multi-layer perceptron implementation for classification or regression.
- ReLU a nonlinear activation function is usually used for hidden layers to avoid vanishing gradients into linear regression. Only to learn features.
- softmax is commonly used for the output layer in classification tasks to produce probability distributions, which sums to 1, positive values.
- linear activation is used for regression tasks to allow unbounded output values.
- sigmoid activation is used for binary classification tasks to produce probabilities between 0 and 1.

Features:
- Arbitrary number of layers (list of explicit user-supplied weight matrices).
- Bias handled by explicit leading column of ones in layer inputs (automatically inserted before each weight multiplication).
- Supported activations: relu, linear, softmax, sigmoid.
- Supported losses: mse (with linear final activation), cross_entropy (with softmax + one-hot targets), binary_cross_entropy (with sigmoid + 0/1 targets).
- Full-batch gradient descent training (no stochasticity; weights must be provided by caller).
- Verbose mode prints forward activations (Z objective function output and A activation output per layer) plus backprop gradients and updated weights each iteration.

Interface Overview:
1. Initialize: mlp = SimpleMLP(layer_sizes=[n_in, h1, ..., n_out], activations=[act1, ..., actL], weights=[W1,...,WL])
    - layer_sizes excludes bias terms. Each weight matrix shape must be (prev_size+1, next_size).
    - activations length must equal len(layer_sizes)-1.
2. Train: mlp.train(X, Y, learning_rate=0.1, iters=100, loss='cross_entropy', verbose=True)
    - X shape: (N, n_in). Do NOT include bias column.
    - Y shape: (N, n_out) manual one-hot for softmax+cross_entropy; (N, n_out) for mse; (N,1) or (N,) for sigmoid+binary_cross_entropy.
3. Predict: mlp.predict(X) returns network output.
4. History: mlp.history['loss'] stores loss per iteration.

Verbose Printing (if verbose=True):
Iter k:
  Loss=...
  Forward pass:
      Layer i: Z shape ..., Z full matrix, A shape ..., A full matrix
  Backpropagation:
      Layer i: gradient shape ..., full gradient matrix, updated W shape ..., full updated weight matrix

This implementation is intentionally compact and uses only numpy.
"""

# ---------------- Activation Functions ----------------

def relu(z):
    return np.maximum(0.0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def linear(z):
    return z

def linear_deriv(z):
    return np.ones_like(z)

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

ACTIVATIONS = {
    'relu': (relu, relu_deriv),
    'linear': (linear, linear_deriv),
    'softmax': (softmax, None),      # derivative handled with cross-entropy simplification
    'sigmoid': (sigmoid, sigmoid_deriv)
}

# ---------------- Loss Functions ----------------

def mse(pred, Y):
    return np.mean(np.sum((pred - Y)**2, axis=1))

def cross_entropy(pred, Y):
    eps = 1e-12
    p = np.clip(pred, eps, 1 - eps)
    return -np.mean(np.sum(Y * np.log(p), axis=1))

def binary_cross_entropy(pred, Y):
    # Y: shape (N,1) or (N,) with 0/1 labels, pred: probabilities in (0,1)
    eps = 1e-12
    p = np.clip(pred, eps, 1 - eps)
    Y_flat = Y.reshape(p.shape)
    return -np.mean(Y_flat * np.log(p) + (1 - Y_flat) * np.log(1 - p))

class SimpleMLP:
    def __init__(self, layer_sizes, activations, weights, fmt_precision=8):
        """Initialize MLP with explicit user-supplied weights.

        layer_sizes: [n_in, h1, ..., n_out] (without bias terms)
        activations: list of activation names per layer (len = len(layer_sizes)-1)
        weights: list of np.ndarray, each shape (prev_size+1, next_size) including bias row.
        """
        if len(layer_sizes) < 2:
            raise ValueError('layer_sizes must have at least input and output size.')
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError('activations length must equal number of layers minus one.')
        if len(weights) != len(activations):
            raise ValueError('weights length must match activations length.')
        for i, W in enumerate(weights):
            expected = (layer_sizes[i] + 1, layer_sizes[i+1])
            if W.shape != expected:
                raise ValueError(f'Weight {i} shape {W.shape} != expected {expected}')
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = [W.astype(float).copy() for W in weights]
        self.history = {'loss': []}
        # formatting settings (number of decimal places for verbose prints)
        self._fmt_precision = int(fmt_precision)

    def set_print_precision(self, precision):
        """Update number of decimal places used in verbose matrix printing."""
        self._fmt_precision = int(precision)

    @staticmethod
    def _format_matrix(M, precision=8, indent='      '):
        """Return a string with columns padded to equal width.
        Uses fixed-point with given precision; falls back to scientific for very small values.
        """
        if M.ndim != 2:
            return indent + str(M)
        # Prepare formatted strings
        raw = []
        sci_threshold = 1e-4
        for r in M:
            row_fmt = []
            for v in r:
                if v == 0:
                    s = f"{0:.{precision}f}"
                elif abs(v) < sci_threshold:
                    s = f"{v:.{precision}e}"
                else:
                    s = f"{v:.{precision}f}"
                row_fmt.append(s)
            raw.append(row_fmt)
        # Determine column widths
        col_widths = [max(len(raw[i][j]) for i in range(len(raw))) for j in range(len(raw[0]))]
        lines = []
        for row in raw:
            padded = [s.rjust(col_widths[j]) for j, s in enumerate(row)]
            lines.append(indent + ' '.join(padded))
        return '\n'.join(lines)

    # Forward pass returns (outputs, Z_list, A_list)
    def forward(self, X):
        # X: (N, n_in). Build bias-augmented activations per layer except output.
        A_list = []  # activations including bias augmentations except final
        Z_list = []
        A = X
        for idx, (W, act_name) in enumerate(zip(self.weights, self.activations)):
            # add bias column to A
            A_aug = np.hstack([np.ones((A.shape[0], 1)), A])
            Z = A_aug @ W
            Z_list.append(Z)
            act_fn, _ = ACTIVATIONS[act_name]
            A = act_fn(Z)
            A_list.append(A_aug)  # store augmented input that produced Z
        return A, Z_list, A_list  # final output, all Zs, all augmented As

    def predict(self, X):
        out, _, _ = self.forward(X)
        return out

    def _compute_loss_and_delta(self, out, Y, loss):
        N = Y.shape[0]
        if loss == 'mse':
            if self.activations[-1] != 'linear':
                raise ValueError('MSE loss requires final activation to be linear.')
            L = mse(out, Y)
            delta = 2 * (out - Y) / N  # dJ/dZ_final (linear activation derivative = 1)
        elif loss == 'cross_entropy':
            if self.activations[-1] != 'softmax':
                raise ValueError('Cross-entropy loss requires final activation softmax.')
            L = cross_entropy(out, Y)
            delta = (out - Y) / N  # softmax + cross entropy gradient
        elif loss == 'binary_cross_entropy':
            if self.activations[-1] != 'sigmoid':
                raise ValueError('Binary cross-entropy loss requires final activation sigmoid.')
            L = binary_cross_entropy(out, Y)
            # derivative of BCE wrt pre-activation z using sigmoid: (sigmoid(z)-Y)/N == (out - Y)/N
            delta = (out - Y.reshape(out.shape)) / N
        else:
            raise ValueError('Unsupported loss.')
        return L, delta

    def train(self, X, Y, learning_rate=0.1, iters=100, loss='cross_entropy', verbose=False, print_every=1, X_test=None):
        # Basic shape validation
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f'Sample count mismatch: X has {X.shape[0]} rows, Y has {Y.shape[0]} rows')
        for it in range(1, iters + 1):
            # Forward pass
            out, Z_list, A_list = self.forward(X)
            # Collect activation outputs (post-activation for each layer)
            activation_outputs = []
            for zi, act_name in zip(Z_list, self.activations):
                act_fn, _ = ACTIVATIONS[act_name]
                activation_outputs.append(act_fn(zi))
            # Loss & initial delta at output layer
            L, delta = self._compute_loss_and_delta(out, Y, loss)
            # Backpropagation
            deltas = [None] * len(self.weights)
            deltas[-1] = delta
            gradients = [None] * len(self.weights)
            updated_weights = [None] * len(self.weights)
            for layer_idx in reversed(range(len(self.weights))):
                W = self.weights[layer_idx]
                A_aug_in = A_list[layer_idx]
                # Gradient wrt weights
                grad_W = A_aug_in.T @ deltas[layer_idx]
                gradients[layer_idx] = grad_W
                # Weight update
                self.weights[layer_idx] = W - learning_rate * grad_W
                updated_weights[layer_idx] = self.weights[layer_idx]
                # Propagate delta backward (except for first layer)
                if layer_idx > 0:
                    prev_W = W  # original weights before update for proper gradient flow
                    back_no_bias = deltas[layer_idx] @ prev_W[1:, :].T
                    prev_act_name = self.activations[layer_idx - 1]
                    _, prev_deriv = ACTIVATIONS[prev_act_name]
                    if prev_deriv is not None:
                        deriv_prev = prev_deriv(Z_list[layer_idx - 1])
                        deltas[layer_idx - 1] = back_no_bias * deriv_prev
                    else:
                        deltas[layer_idx - 1] = back_no_bias
            # Record loss
            self.history['loss'].append(L)
            # Verbose printing
            if verbose and (it % print_every == 0):
                print(f"Iter {it}: Loss={L:.6f}")
                print("  Forward pass:")
                for li, (Z, A_out) in enumerate(zip(Z_list, activation_outputs)):
                    act_name = self.activations[li]
                    print(f"    Layer {li+1} ({act_name}) pre-activation Z shape {Z.shape}")
                    print("      Z:")
                    print(self._format_matrix(Z, precision=self._fmt_precision, indent='        '))
                    print(f"      Activation output shape {A_out.shape}")
                    print("      A:")
                    print(self._format_matrix(A_out, precision=self._fmt_precision, indent='        '))
                print("  Backpropagation:")
                for li in range(len(self.weights)):
                    layer_name = 'output' if li == len(self.weights) - 1 else f'hidden {li+1}'
                    print(f"    Layer {li+1} ({layer_name}) gradient shape {gradients[li].shape}")
                    print("      Gradient matrix:")
                    print(self._format_matrix(gradients[li], precision=self._fmt_precision, indent='        '))
                    print(f"      Updated weights shape {updated_weights[li].shape}")
                    print("      Updated weights matrix:")
                    print(self._format_matrix(updated_weights[li], precision=self._fmt_precision, indent='        '))
                if X_test is not None:
                    test_pred = self.predict(X_test)
                    print(f"  Test sample prediction: {test_pred[0]}")
                print("")
        return self

# ---------------- Usage Examples ----------------
if __name__ == '__main__':
    
    # no need to pad with bias since input X already has bias column
    X_train = np.array([
        [1.2, -0.4, 0.8],
        [-0.6, 2.0, -0.5],
        [0.3, -1.2, 1.7],
        [2.1, 0.5, -0.8]
    ])
    # need to manually one-hot encode the labels
    Y_train = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    X_test = np.array([
        [1, 1, 1]
    ])
    # col is neuron i weights in current layer, row must include bias row, then no of samples/neurons from prev layer
    w1 = np.array([
        [0, 0, 0],
        [0.02, -0.01, 0.03],
        [-0.05, 0.04, 0.01],
        [0.03, 0.02, -0.02]
    ])
    w2 = w1.copy()
    # mlp: f(x) = softmax([1, relu(X@w1)]@w2) : 1 n_in -> 1 hidden -> 1 n_out. Layer size all 3 (excludes bias)
    # so 3 layers of 3 neurons each, hidden layer 1 with relu, output with softmax.
    # weights col = no of neurons in that layer. rows = bias + no of neurons in prev layer
    mlp = SimpleMLP(layer_sizes=[3, 3, 3], activations=['relu', 'softmax'], weights=[w1, w2])
    # loss function is categorical cross entropy, and we want to see verbose output
    mlp.train(X_train, Y_train, learning_rate=0.1, iters=1, loss='cross_entropy', verbose=True, X_test=X_test)
    
    
    ''' Classification example '''
    # X_train = np.array([[1, 3.0], [2, 2.5]])
    # Y_train = np.array([[1, 0], [0, 1]])
    # X_test = np.array([[1.5, 2.7]])
    # Wc1 = np.array([[0.2, -0.1], [-0.3, 0.4], [0.1, -0.2]])  # (2+1,2)
    # Wc2 = np.array([[0.05, -0.05], [0.2, 0.3], [-0.4, 0.1]]) # (2+1,2)
    # mlp_c = SimpleMLP(layer_sizes=[2, 2, 2], activations=['relu','softmax'], weights=[Wc1, Wc2])
    # mlp_c.train(X_train, Y_train, learning_rate=0.1, iters=5, loss='cross_entropy', verbose=True, X_test=X_test)
    # print('Train predictions:\n', mlp_c.predict(X_train))
    # print('Test prediction:\n', mlp_c.predict(X_test))

    ''' Binary classification (sigmoid + BCE) '''
    # Xb = np.array([[0.0, 1.0], [1.0, -0.5], [2.0, 0.3]])
    # Yb = np.array([[1], [0], [1]])
    # Wb1 = np.array([[0.1, -0.05, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.25, 0.15]])  # (2+1,3)
    # Wb2 = np.array([[0.05], [0.2], [-0.1], [0.3]])  # (3+1,1)
    # mlp_b = SimpleMLP(layer_sizes=[2,3,1], activations=['relu','sigmoid'], weights=[Wb1, Wb2])
    # mlp_b.train(Xb, Yb, learning_rate=0.1, iters=8, loss='binary_cross_entropy', verbose=True)
    # print('Binary probabilities:\n', mlp_b.predict(Xb))

    ''' Regression example '''
    # Xr = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]])
    # Yr = np.array([[2.0], [0.5], [1.5]])
    # Wr1 = np.array([[0.1, -0.2, 0.05], [0.3, 0.0, -0.1], [-0.25, 0.2, 0.15]]) # (2+1,3)
    # Wr2 = np.array([[0.05],[0.2],[-0.1],[0.3]]) # (3+1,1)
    # mlp_r = SimpleMLP(layer_sizes=[2,3,1], activations=['relu','linear'], weights=[Wr1, Wr2])
    # mlp_r.train(Xr, Yr, learning_rate=0.05, iters=8, loss='mse', verbose=True)
    # print('Regression predictions:\n', mlp_r.predict(Xr))
