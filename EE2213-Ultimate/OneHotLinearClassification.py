def onehot_linearclassification(X, y, X_test):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from numpy.linalg import inv, matrix_rank, det
    from evaluation_metrics import (
    compute_regression_metrics, print_regression_metrics,
    compute_binary_classification_metrics, print_binary_classification_metrics,
    compute_multiclass_metrics, print_multiclass_metrics,
    )
    
    np.set_printoptions(precision=4, suppress=True)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(y)
    
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
    
    w=inv(X.T@X)@X.T@y_onehot
    print("w* is (first row is for bias): \n", w, "\n")

    y_calculated=X@w
    print("y_train_raw is (col pos of largest per row is the class index):\n", y_calculated, "\n")
    y_train_pred = np.argmax(y_calculated,axis=1)
    print("y_train_classified (transpose urself for argmax of each row. Remember + 1 if qn's class starts with 1) is\n", y_train_pred, "\n")
    # Metrics
    y_true_cls = np.argmax(y_onehot, axis=1)
    print("[onehot_linearclassification] Train multiclass metrics:")
    print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_train_pred))
    print("")
    
    y_predicted=X_test@w
    y_predicted_classes=np.argmax(y_predicted,axis=1)
    print("y_test_raw is (col pos of largest per row is the class index):\n", y_predicted, "\n")
    print("y_test_classified (transpose urself for argmax of each row. Remember + 1 if qn's class starts with 1) is\n", y_predicted_classes, "\n")

    # Metrics
    # print("[onehot_linearclassification] Test multiclass metrics:")
    # print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_predicted_classes))
    # print("")