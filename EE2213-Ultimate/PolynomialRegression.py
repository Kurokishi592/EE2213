def polynomial_regression(X, y, order, X_test, binary_threshold=None):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from numpy.linalg import inv, matrix_rank, det
    from evaluation_metrics import (
        compute_regression_metrics, print_regression_metrics,
        compute_binary_classification_metrics, print_binary_classification_metrics,
        compute_multiclass_metrics, print_multiclass_metrics,
    )
    
    np.set_printoptions(precision=4, suppress=True)
    
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X) # includes column of 1s
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])

    print("the polynomial transformed matrix P is:", "\n")
    print(P)
    print("")
    
    # check inverse
    P_sqr = P.T @ P
    rank = matrix_rank(P_sqr)
    print("P.T @ P is : \n"+ str(P_sqr))
    print("P.T @ P rank is : "+ str(rank))
    print("P.T @ P dimension is : "+ str(P_sqr.shape))
    print("P.T @ P determinant is : "+ str(det(P_sqr)))
    if P_sqr.shape[0] == P_sqr.shape[1]:
       if rank == P_sqr.shape[0]:
           print("P.T @ P is invertible")
       else:
           print("P.T @ P is not invertible, use ridge regression instead")
    else:
       print("P.T @ P is not square, hence not invertible, use ridge regression instead")
    print("")

    w = inv(P.T @ P) @ P.T @ y

    print("w is (first row is bias): ")
    print(w)
    print("")

    P_train_predicted=P@w
    print("y_train_predicted is:\n ", np.round(P_train_predicted, 4))
    print("")
    print("if one hot encoding multi-class classification, y_train_classes are (transpose urself for argmax of each row): \n", np.argmax(P_train_predicted, axis=1), "\n")
    # Binary classification threshold handling
    y_arr = np.asarray(y)
    vals = np.unique(y_arr.ravel())
    is_binary_minus_plus = set(vals).issubset({-1, 1})
    is_binary_zero_one = set(vals).issubset({0, 1})
    if binary_threshold is None:
        if is_binary_minus_plus:
            binary_threshold_used = 0.0
        elif is_binary_zero_one:
            binary_threshold_used = 0.5
        else:
            binary_threshold_used = None
    else:
        binary_threshold_used = float(binary_threshold)

    if is_binary_minus_plus and binary_threshold_used is not None:
        y_train_bin = np.where(P_train_predicted.ravel() > binary_threshold_used, 1, -1)
        print(f"binary classification (labels -1/1) threshold={binary_threshold_used}:\n ", y_train_bin, "\n")
    elif is_binary_zero_one and binary_threshold_used is not None:
        y_train_bin = (P_train_predicted.ravel() > binary_threshold_used).astype(int)
        print(f"binary classification (labels 0/1) threshold={binary_threshold_used}:\n ", y_train_bin, "\n")
    else:
        print("if binary classification, y_train_predicted_classified (default sign or >=0.5) is:\n ", np.sign(P_train_predicted), "\n")
    
    # Metrics: regression always, plus classification when labels indicate it
    try:
        train_reg = compute_regression_metrics(y_true=y, y_pred=P_train_predicted)
        print("[polynomial_regression] Train regression metrics:")
        print_regression_metrics(train_reg)
    except Exception:
        pass

    # Classification metrics detection
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        # Multiclass one-hot
        y_true_cls = np.argmax(y_arr, axis=1)
        y_pred_cls = np.argmax(P_train_predicted, axis=1)
        print("\n[polynomial_regression] Train multiclass metrics:")
        print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    else:
        if is_binary_minus_plus:
            if binary_threshold_used is None:
                y_pred_cls = np.sign(P_train_predicted).ravel()
            else:
                y_pred_cls = np.where(P_train_predicted.ravel() > binary_threshold_used, 1, -1)
            pos = 1
            print("\n[polynomial_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
        elif is_binary_zero_one:
            if binary_threshold_used is None:
                y_pred_cls = (P_train_predicted.ravel() >= 0.5).astype(int)
            else:
                y_pred_cls = (P_train_predicted.ravel() > binary_threshold_used).astype(int)
            pos = 1
            print("\n[polynomial_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
    print("")

    P_test = poly.fit_transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_test_predicted is")
    print(np.round(y_predicted, 4))
    print("")
    print("if one hot encoding multi-class classification, y_test_classes are (transpose urself for argmax of each row): \n", np.argmax(y_predicted, axis=1), "\n")
    if is_binary_minus_plus and binary_threshold_used is not None:
        y_test_bin = np.where(y_predicted.ravel() > binary_threshold_used, 1, -1)
        print(f"binary classification test (labels -1/1) threshold={binary_threshold_used}:\n ", y_test_bin)
    elif is_binary_zero_one and binary_threshold_used is not None:
        y_test_bin = (y_predicted.ravel() > binary_threshold_used).astype(int)
        print(f"binary classification test (labels 0/1) threshold={binary_threshold_used}:\n ", y_test_bin)
    else:
        print("if binary classification, y_test_predicted_classified is:\n ", np.sign(y_predicted))
    
    # Test metrics
    # try:
    #     test_reg = compute_regression_metrics(y_true=y, y_pred=y_predicted)  # train metrics already shown
    # except Exception:
    #     pass

    # if y_arr.ndim == 2 and y_arr.shape[1] > 1:
    #     y_true_cls = np.argmax(y_arr, axis=1)
    #     y_pred_cls = np.argmax(y_predicted, axis=1)
    #     print("[ridge_regression] Test multiclass metrics:")
    #     print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    # else:
    #     vals = np.unique(y_arr.ravel())
    #     if set(vals).issubset({-1, 1}):
    #         y_pred_cls = np.sign(y_predicted).ravel()
    #         print("[ridge_regression] Test binary metrics:")
    #         print_binary_classification_metrics(
    #             compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
    #         )
    #     elif set(vals).issubset({0, 1}):
    #         y_pred_cls = (y_predicted.ravel() >= 0.5).astype(int)
    #         print("[ridge_regression] Test binary metrics:")
    #         print_binary_classification_metrics(
    #             compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
    #         )