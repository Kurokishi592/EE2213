def ridge_poly_regression(X, y, LAMBDA, order, X_test, binary_threshold=None):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from evaluation_metrics import (
        compute_regression_metrics, print_regression_metrics,
        compute_binary_classification_metrics, print_binary_classification_metrics,
        compute_multiclass_metrics, print_multiclass_metrics,
    )
    np.set_printoptions(precision=4, suppress=True)
    
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    I = np.identity(P.shape[1])
    w = np.linalg.inv(P.T @ P+LAMBDA*I) @ P.T @ y
    print("w is (first row is order 0): ")
    print(w)
    print("")

    P_train_predicted=P@w
    print("y_train_predicted is: \n", np.round(P_train_predicted, 4), "\n")
    print("if one hot encoding multi-class classification, y_train_classes are (transpose urself for argmax of each row): \n", np.argmax(P_train_predicted, axis=1), "\n")
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
        print(f"binary classification (labels -1/1) threshold={binary_threshold_used}: \n", y_train_bin, "\n")
    elif is_binary_zero_one and binary_threshold_used is not None:
        y_train_bin = (P_train_predicted.ravel() > binary_threshold_used).astype(int)
        print(f"binary classification (labels 0/1) threshold={binary_threshold_used}: \n", y_train_bin, "\n")
    else:
        print("if binary classification, y_train_predicted_classified is: \n", np.sign(P_train_predicted) , "\n")

    # Train metrics
    try:
        train_reg = compute_regression_metrics(y_true=y, y_pred=P_train_predicted)
        print("[ridge_poly] Train regression metrics:")
        print_regression_metrics(train_reg)
    except Exception:
        pass

    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        y_true_cls = np.argmax(y_arr, axis=1)
        y_pred_cls = np.argmax(P_train_predicted, axis=1)
        print("\n[ridge_poly] Train multiclass metrics:")
        print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    else:
        if is_binary_minus_plus:
            if binary_threshold_used is None:
                y_pred_cls = np.sign(P_train_predicted).ravel()
            else:
                y_pred_cls = np.where(P_train_predicted.ravel() > binary_threshold_used, 1, -1)
            print("\n[ridge_poly] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
            )
        elif is_binary_zero_one:
            if binary_threshold_used is None:
                y_pred_cls = (P_train_predicted.ravel() >= 0.5).astype(int)
            else:
                y_pred_cls = (P_train_predicted.ravel() > binary_threshold_used).astype(int)
            print("\n[ridge_poly] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
            )
    print("")

    P_test = poly.fit_transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_test_predicted is:")
    print(np.round(y_predicted, 4))
    print("")
    print("if one hot encoding multi-class classification, y_test_classes are (transpose urself for argmax of each row): \n", np.argmax(y_predicted, axis=1), "\n")
    if is_binary_minus_plus and binary_threshold_used is not None:
        y_test_bin = np.where(y_predicted.ravel() > binary_threshold_used, 1, -1)
        print(f"binary classification test (labels -1/1) threshold={binary_threshold_used}: \n", y_test_bin, "\n")
    elif is_binary_zero_one and binary_threshold_used is not None:
        y_test_bin = (y_predicted.ravel() > binary_threshold_used).astype(int)
        print(f"binary classification test (labels 0/1) threshold={binary_threshold_used}: \n", y_test_bin, "\n")
    else:
        print("if binary classification, y_test_predicted_classified is: \n", np.sign(y_predicted), "\n")
    
    # Test metrics
    # if y_arr.ndim == 2 and y_arr.shape[1] > 1:
    #     y_true_cls = np.argmax(y_arr, axis=1)
    #     y_pred_cls = np.argmax(y_predicted, axis=1)
    #     print("[ridge_poly] Test multiclass metrics:")
    #     print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    # else:
    #     vals = np.unique(y_arr.ravel())
    #     if set(vals).issubset({-1,1}):
    #         y_pred_cls = np.sign(y_predicted).ravel()
    #         print("[ridge_poly] Test binary metrics:")
    #         print_binary_classification_metrics(
    #             compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
    #         )
    #     elif set(vals).issubset({0,1}):
    #         y_pred_cls = (y_predicted.ravel() >= 0.5).astype(int)
    #         print("[ridge_poly] Test binary metrics:")
    #         print_binary_classification_metrics(
    #             compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
    #         )


    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #HI
    # return(system, P, w, y_predicted, y_classified))










def ridge_poly_regression_simplified(X,y,LAMBDA,order, form, X_test, y_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    # print("the number of parameters: ", P.shape[1])
    # print("the number of samples: ", P.shape[0])
    if form=="auto":
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    # print(system, "system   ", form)
    # print("")
    # print("the polynomial transformed matrix P is:")
    # print(P)
    # print("")

    if form=="primal form":
        I = np.identity(P.shape[1])
        w = np.linalg.inv(P.T @ P+LAMBDA*I) @ P.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = P.T @ np.linalg.inv(P @ P.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(P) @ y

    # print("w is: ")
    # print(w)
    # print("")

    P_test = poly.fit_transform(X_test)
    # print("transformed test sample P_test is")
    # print(P_test)
    # print("")
    y_predicted = P_test @ w
    # print("y_predicted is")
    # print(y_predicted)

    P_train_predicted=P@w
    # print("y_train_predicted is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_train_predicted-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    # print("square error is", sum_of_square)
    print("ridge train MEAN square error is", mean_squared_error)

    P_test_predicted=P_test@w
    # print("y_train_predicted is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_test_predicted-y_test)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y_test.shape[0]
    # print("square error is", sum_of_square)
    print("ridge test MEAN square error is", mean_squared_error, "\n")

