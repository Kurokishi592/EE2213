def ridge_regression(X, y, LAMBDA, X_test, binary_threshold=None):
    import numpy as np
    from evaluation_metrics import (
        compute_regression_metrics, print_regression_metrics,
        compute_binary_classification_metrics, print_binary_classification_metrics,
        compute_multiclass_metrics, print_multiclass_metrics,
    )
    np.set_printoptions(precision=4, suppress=True)
    I = np.identity(X.shape[1])
    w = np.linalg.inv(X.T @ X+LAMBDA*I) @ X.T @ y
    print("w is (first row is for bias): ")
    print(w)
    print("")

    y_calculated=X@w
    print("y_train_predicted is: \n", np.round(y_calculated, 4), "\n")
    print("if one hot encoding multi-class classification, y_train_classes are (transpose urself for argmax of each row): \n", np.argmax(y_calculated, axis=1), "\n")
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
        y_train_bin = np.where(y_calculated.ravel() > binary_threshold_used, 1, -1)
        print(f"binary classification (labels -1/1) threshold={binary_threshold_used}:\n", y_train_bin, "\n")
    elif is_binary_zero_one and binary_threshold_used is not None:
        y_train_bin = (y_calculated.ravel() > binary_threshold_used).astype(int)
        print(f"binary classification (labels 0/1) threshold={binary_threshold_used}:\n", y_train_bin, "\n")
    else:
        print("if binary classification, y_train_predicted_classified is\n", np.sign(y_calculated), "\n")
    
    try:
        train_reg = compute_regression_metrics(y_true=y, y_pred=y_calculated)
        print("[ridge_regression] Train regression metrics:")
        print_regression_metrics(train_reg)
    except Exception:
        pass

    # Classification metrics detection
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        # Multiclass one-hot
        y_true_cls = np.argmax(y_arr, axis=1)
        y_pred_cls = np.argmax(y_calculated, axis=1)
        print("\n[ridge_regression] Train multiclass metrics:")
        print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    else:
        if is_binary_minus_plus:
            if binary_threshold_used is None:
                y_pred_cls = np.sign(y_calculated).ravel()
            else:
                y_pred_cls = np.where(y_calculated.ravel() > binary_threshold_used, 1, -1)
            pos = 1
            print("\n[ridge_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
        elif is_binary_zero_one:
            if binary_threshold_used is None:
                y_pred_cls = (y_calculated.ravel() >= 0.5).astype(int)
            else:
                y_pred_cls = (y_calculated.ravel() > binary_threshold_used).astype(int)
            pos = 1
            print("\n[ridge_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
    print("")
    
    y_predicted=X_test@w
    print("y_test_predicted is\n", np.round(y_predicted, 4), "\n")
    print("if one hot encoding multi-class classification, y_test_classes are (transpose urself for argmax of each row): \n", np.argmax(y_predicted, axis=1), "\n")
    if is_binary_minus_plus and binary_threshold_used is not None:
        y_test_bin = np.where(y_predicted.ravel() > binary_threshold_used, 1, -1)
        print(f"binary classification test (labels -1/1) threshold={binary_threshold_used}:\n", y_test_bin, "\n")
    elif is_binary_zero_one and binary_threshold_used is not None:
        y_test_bin = (y_predicted.ravel() > binary_threshold_used).astype(int)
        print(f"binary classification test (labels 0/1) threshold={binary_threshold_used}:\n", y_test_bin, "\n")
    else:
        print("if binary classification, y_test_predicted_classified is\n", np.sign(y_predicted), "\n")
    
    # Test metrics
    # try:
    #     test_reg = compute_regression_metrics(y_true=y, y_pred=y_calculated)  # train metrics already shown
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
