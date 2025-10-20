def polynomial_regression(X,y,order,X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X) # includes column of 1s
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])

    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    w = np.linalg.inv(P.T @ P) @ P.T @ y

    print("w is: ")
    print(w)
    print("")

    P_train_predicted=P@w
    print("y_train_predicted is:\n ", P_train_predicted)
    print("MEAN square error is", mean_squared_error(y, P_train_predicted), "\n")
    print("MEAN absolute error is", mean_absolute_error(y, P_train_predicted), "\n")

    P_test = poly.fit_transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_test_predicted is")
    print(y_predicted)
    print("")
    
