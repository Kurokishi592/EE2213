from numpy.ma.core import identity


def ridge_regression(X, y, LAMBDA, X_test, form="auto"):
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    I = np.identity(X.shape[1])
    w = np.linalg.inv(X.T @ X+LAMBDA*I) @ X.T @ y
    print("w is: ")
    print(w)
    print("")

    y_calculated=X@w
    print("y_train_predicted is: \n", y_calculated, "\n")
    print("MEAN square error is", mean_squared_error(y, y_calculated), "\n")
    print("MEAN absolute error is", mean_absolute_error(y, y_calculated), "\n")

    y_predicted=X_test@w
    print("y_test_predicted is\n", y_predicted, "\n")
