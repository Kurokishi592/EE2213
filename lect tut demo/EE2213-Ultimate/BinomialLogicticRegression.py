def logistic_regression(X, y, X_test, w, learning_rate, num_iters, binary_threshold=0.5):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from matplotlib import pyplot as plt
    
    # step 0: prepare data
    poly = PolynomialFeatures(1)
    P = poly.fit_transform(X)

    # step 1: learning
    pred_y, cost, gradient = logistic_cost_gradient(P, w, y)
    # Allow scientific notation for very small numbers
    np.set_printoptions(precision=4, suppress=False)
    def _fmt_scalar(val):
        return f"{val:.4e}" if (val != 0 and abs(val) < 1e-4) else f"{val:.4f}"
    print('Initial Cost =', _fmt_scalar(cost))
    print('Initial Gradient =\n', gradient)
    print('Initial Weights =\n', w)
    print("")
    cost_vec = np.zeros(num_iters+1) # Creates a 1D NumPy array filled with zeros
    cost_vec[0] = cost

    for i in range(1, num_iters+1):

        # update w
        w = w - learning_rate*gradient

        # compute updated cost and new gradient
        pred_y, cost, gradient = logistic_cost_gradient(P, w, y)
        cost_vec[i] = cost

        if(i % 1000 == 0):
            print('Iter', i, ': cost =', _fmt_scalar(cost))
        if(i<3):
            print('Iter', i, ': cost =', _fmt_scalar(cost))
            print('Gradient =\n', gradient)
            print('Weights =\n', w)
            print("")

    print('Final Cost =', _fmt_scalar(cost))
    print('Final Weights =\n', w)
    print("")

    # Training predictions
    y_pred_classes = (pred_y >= binary_threshold).astype(int)
    print("y_pred_prob (P(class=1)) is: \n" + str(pred_y))
    print("y_pred_classes (zero-based 0/1): \n" + str(y_pred_classes))
    print("")

    # Evaluation metrics (train set)
    # y expected shape (n,1) with 0/1 labels; flatten for metrics
    y_true_flat = y.ravel()
    y_pred_flat = y_pred_classes.ravel()
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    print("Train Confusion Matrix:\n" + str(cm))
    print("Train Accuracy:" , _fmt_scalar(acc))
    print("Train Precision:" , _fmt_scalar(prec))
    print("Train Recall:" , _fmt_scalar(rec))
    print("Train F1:" , _fmt_scalar(f1))

    # Plot cost function values over iterations (uncomment to see the plot)
    # plt.figure(0, figsize=[9,4.5])
    # plt.rcParams.update({'font.size': 16})
    # plt.plot(np.arange(0, num_iters+1, 1), cost_vec)
    # plt.xlabel('Iteration Number')
    # plt.ylabel('Mean Binary Cross-entropy Loss')
    # plt.xticks(np.arange(0, num_iters+1, 1000)) #Sets the x-axis tick locations in the current plot to those values to prevent being cluttered.
    # plt.title('Learning rate = ' + str(learning_rate))
    
    #step 2: prediction
    X_test = poly.transform(X_test)   # use same feature mapping
    z = X_test @ w
    y_test = 1/(1+np.exp(-z)) # sigmoid function
    print("")
    print("z: " + str(z))
    y_test_classes = (y_test >= binary_threshold).astype(int)
    print("y_test_prob (P(class=1)): " + str(y_test))
    print("y_test_classes (zero-based 0/1): " + str(y_test_classes))

    # Test metrics (if ground truth y_test available user can compute externally; here only predictions returned)
    # Return unchanged signature w, y_test
    
    return w, y_test

def logistic_cost_gradient(X, w, y, eps=1e-15):
    # Compute prediction, cost and gradient based on mean binary cross-entropy loss.
    import numpy as np
    # eps is a small value to prevent log(0) and log(1)
    pred_y = 1/(1+np.exp(-X @ w))

    # Clip predictions to prevent log(0) and log(1)
    # Due to floating-point precision (finite machine representation), 
    # when numbers are very small, they can be rounded down to 0.0. 
    # Similarly, very large exponentials can push probabilities to exactly 1.0.
    pred_y = np.clip(pred_y, eps, 1 - eps)
    cost   = np.sum(- (y * np.log(pred_y) + (1 - y) * np.log(1 - pred_y)))/X.shape[0]
    gradient = (X.T @ (pred_y-y))/X.shape[0]

    return pred_y, cost, gradient