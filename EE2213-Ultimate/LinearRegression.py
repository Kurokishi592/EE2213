def linear_regression(X, y, X_test):
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    w=np.linalg.inv(X.T@X)@X.T@y
    print("w* is: \n", w, "\n")

    y_calculated=X@w

    print("y_train_predicted is\n" , (y_calculated), "\n")
    # print("y_train_predicted_classified is\n" , np.sign(y_calculated), "\n")
    print("MEAN square error is", mean_squared_error(y, y_calculated), "\n")
    print("MEAN absolute error is", mean_absolute_error(y, y_calculated), "\n")

    y_predicted=X_test@w
    print("y_test_predicted is\n", y_predicted, "\n")
    # print("y_test_predicted_classified is\n", np.sign(y_predicted), "\n")

    # print("X rank:", np.linalg.matrix_rank(X))
    # result=np.hstack((X,y))
    # print("X|y rank: ", np.linalg.matrix_rank(result))


    return(w, y_calculated, y_predicted)
