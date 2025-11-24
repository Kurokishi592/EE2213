def multinomial_logistic_regression(X, Y, X_test, W, learning_rate, num_iters):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from matplotlib import pyplot as plt
    
    # step 0: prepare data
    poly = PolynomialFeatures(1)
    P = poly.fit_transform(X)
    np.set_printoptions(precision=4, suppress=False)
    def _fmt_scalar(val):
        return f"{val:.4e}" if (val != 0 and abs(val) < 1e-4) else f"{val:.4f}"
    print("P: " + str(P))
    print("")
    # class1: y=[1,0,0]; class2: y=[0,1,0]; class3: y=[0,0,1]
    onehot_encoder = OneHotEncoder(sparse_output=False)
    Y_onehot = onehot_encoder.fit_transform(Y)
    print("the onehot encoded Y (columns = zero-based classes) is:\n", Y_onehot)
    print("")

    # step 1: learning
    pred_y, cost, gradient = multi_logistic_cost_gradient(P, W, Y_onehot)
    print('Initial Cost =', _fmt_scalar(cost))
    print('Initial Gradient =\n', gradient)
    print('Initial Weights =\n', W)
    print("")

    cost_vec = np.zeros(num_iters+1) # Creates a 1D NumPy array filled with zeros
    cost_vec[0] = cost

    for i in range(1, num_iters+1):

        # update w
        W = W - learning_rate*gradient

        # compute updated cost and new gradient
        pred_y, cost, gradient = multi_logistic_cost_gradient(P, W, Y_onehot)
        cost_vec[i] = cost

        if(i % 1000 == 0):
            print('Iter', i, ': cost =', _fmt_scalar(cost))
        if(i<3):
            print('Iter', i, ': cost =', _fmt_scalar(cost))
            print('Gradient =\n', gradient)
            print('Weights =\n', W)
            print("")

    print('Final Cost =', _fmt_scalar(cost))
    print('Final Weights =\n', W)
    print("")
    
    # print("Confusion Matrix (rows: actual class, columns: predicted class. Diagonal are correct predictions):")
    # print(confusion_matrix(Y_onehot, pred_y))
    # print("Accuracy:", accuracy_score(Y_onehot, pred_y))
    
    # Training predictions & metrics
    train_pred_classes = np.argmax(pred_y, axis=1)
    true_train_classes = np.argmax(Y_onehot, axis=1)
    cm = confusion_matrix(true_train_classes, train_pred_classes)
    acc = accuracy_score(true_train_classes, train_pred_classes)
    prec_macro = precision_score(true_train_classes, train_pred_classes, average='macro', zero_division=0)
    rec_macro = recall_score(true_train_classes, train_pred_classes, average='macro', zero_division=0)
    f1_macro = f1_score(true_train_classes, train_pred_classes, average='macro', zero_division=0)
    print("Confusion Matrix (rows: actual class, columns: predicted class. Diagonal are correct predictions):")
    print("Train Confusion Matrix:\n" + str(cm))
    print("Train Accuracy:" , _fmt_scalar(acc))
    print("Train Precision (macro):" , _fmt_scalar(prec_macro))
    print("Train Recall (macro):" , _fmt_scalar(rec_macro))
    print("Train F1 (macro):" , _fmt_scalar(f1_macro))

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
    z = X_test @ W
    Y_test = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True) #softmax function
    print("")
    print("z: " + str(z))
    Y_test_classes = np.argmax(Y_test, axis=1)
    print("Y_test (class probabilities, columns zero-based): " + str(Y_test))
    print("Y_test_classes (zero-based, rmb + 1): " + str(Y_test_classes))

    # Return unchanged signature
    
    return W, Y_test

def multi_logistic_cost_gradient(X, W, Y, eps=1e-15):
    import numpy as np
    # Compute prediction, cost and gradient based on cross entropy
    z = X @ W
    z_max = np.max(z, axis=1, keepdims=True)  # return maximum value per row (axis=1: row-wise); 
                                              # keepdims=True: keep the dimension (2D) for later broadcasting
    exp_z = np.exp(z - z_max) # prevent overflow
    pred_Y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Clip predictions to prevent log(0)
    pred_Y = np.clip(pred_Y, eps, 1 - eps)

    cost   = np.sum(-(Y * np.log(pred_Y)))/X.shape[0]
    gradient = X.T @ (pred_Y-Y) / X.shape[0]

    return pred_Y, cost, gradient