def linear_regression(X, y, X_test):
    import numpy as np
    from numpy.linalg import inv, matrix_rank, det
    from evaluation_metrics import (
    compute_regression_metrics, print_regression_metrics,
    compute_binary_classification_metrics, print_binary_classification_metrics,
    compute_multiclass_metrics, print_multiclass_metrics,
    )
    
    np.set_printoptions(precision=4, suppress=True)
    
    # check inverse
    X_sqr = X.T @ X
    rank = matrix_rank(X_sqr)
    print("X.T @ X is : \n"+ str(X_sqr))
    print("X.T @ X rank is : "+ str(rank))
    print("X.T @ X dimension is : "+ str(X_sqr.shape))
    print("X.T @ X determinant is : "+ str(det(X_sqr)))
    if X_sqr.shape[0] == X_sqr.shape[1]:
       if rank == X_sqr.shape[0]:
           print("X.T @ X is invertible")
       else:
           print("X.T @ X is not invertible, use ridge regression instead")
    else:
       print("X.T @ X is not square, hence not invertible, use ridge regression instead")
    print("")
    
    w=inv(X.T@X)@X.T@y
    print("w* is (first row is for bias): \n", w, "\n")

    y_calculated=X@w
    
    train_metrics = compute_regression_metrics(y_true=y, y_pred=y_calculated)
    print("[linear_regression] Train metrics:")
    print_regression_metrics(train_metrics)

    # Classification metrics detection
    y_arr = np.asarray(y)
    vals = np.unique(y_arr.ravel())
    if set(vals).issubset({-1, 1}):
        y_pred_cls = np.sign(y_calculated).ravel()
        pos = 1
        print("\n[linear_regression] Train binary metrics:")
        print_binary_classification_metrics(
            compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
        )
    elif set(vals).issubset({0, 1}):
        y_pred_cls = (y_calculated.ravel() >= 0.5).astype(int)
        pos = 1
        print("\n[linear_regression] Train binary metrics:")
        print_binary_classification_metrics(
            compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
        )
    print("")

    y_predicted_train=X@w
    print("y_train_predicted is\n" , np.round(y_predicted_train, 4), "\n")
    print("if binary classification, y_train_predicted_classified is\n" , np.sign(y_predicted_train), "\n")

    y_predicted_test=X_test@w
    print("y_test_predicted is\n", np.round(y_predicted_test, 4), "\n")
    print("if binary classification, y_test_predicted_classified is\n", np.sign(y_predicted_test), "\n")


    return(w, y_calculated, y_predicted_test)