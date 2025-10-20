def logistic_regression(X, y, X_test, w, learning_rate, num_iters):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import confusion_matrix, accuracy_score
    from matplotlib import pyplot as plt
    
    # step 0: prepare data
    poly = PolynomialFeatures(1)
    P = poly.fit_transform(X)

    # step 1: learning
    pred_y, cost, gradient = logistic_cost_gradient(P, w, y)
    print('Initial Cost =', cost)
    print('Initial Gradient =', gradient)
    print('Initial Weights =', w)

    cost_vec = np.zeros(num_iters+1) # Creates a 1D NumPy array filled with zeros
    cost_vec[0] = cost

    for i in range(1, num_iters+1):

        # update w
        w = w - learning_rate*gradient

        # compute updated cost and new gradient
        pred_y, cost, gradient = logistic_cost_gradient(P, w, y)
        cost_vec[i] = cost

        if(i % 1000 == 0):
            print('Iter', i, ': cost =', cost)
        if(i<3):
            print('Iter', i, ': cost =', cost)
            print('Gradient =', gradient)
            print('Weights =', w)

    print('Final Cost =', cost)
    print('Final Weights =', w)
    print("y_pred is: " + str(pred_y))
    
    # print("Confusion Matrix (rows: actual class, columns: predicted class. Diagonal are correct predictions):")
    # print(confusion_matrix(y, pred_y))
    # print("Accuracy:", accuracy_score(y, pred_y))

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
    y_test = 1/(1+np.exp(-z)) #signoid function
    print("")
    print("z: " + str(z))
    print("y_test: " + str(y_test))
    
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