def polynomial_regression_with_classification(X,y,order,X_test):
    import numpy as np
    from numpy.linalg import matrix_rank, inv
    from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X) # includes column of 1s
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])

    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    # if binary class classification using negative/positive thresholding
    # can use the following to convert to -1/+1 for output
    
    # print("y_train_predicted_classified is:\n ", np.sign(P_train_predicted))
    # print("y_test_predicted_classified is\n", np.sign(y_predicted), "\n")

    # if multi-class classification using onehot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(y)
    print("the onehot encoded y is:\n", y_onehot)
    print("")
    
    # Check inverse
    P_sqr = P.T @ P
    rank = matrix_rank(P_sqr)
    print("P.T @ P is \n" + str(P_sqr))
    print("P.T @ P rank is: " + str(rank))
    print("P.T @ P dimension is: " + str(P_sqr.shape) + "\n")
    
    w = inv(P.T @ P) @ P.T @ y_onehot
    print("w is: ")
    print(w)
    print("")
    
    P_train_predicted=P@w
    print("y_train_predicted is:\n ", P_train_predicted)
    print("")
    print("Trained Predicted Class (indexed 0) is [sample 1 class   sample 2 class ...] :")
    print(np.argmax(P_train_predicted, axis=1))
    print("")

    P_test = poly.fit_transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    
    y_test_predicted = P_test @ w
    print("y_test_predicted is")
    print(y_test_predicted)
    print("")
    print("Predicted Class (indexed 0) is [sample 1 class   sample 2 class ...] :")
    print(np.argmax(y_test_predicted, axis=1)) # axis = 1 means per sample
