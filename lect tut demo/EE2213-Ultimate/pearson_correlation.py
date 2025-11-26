def pearson_correlation(X, Y, return_sorted: bool = True, print_output: bool = True):
    """
    Compute Pearson r between each feature column in X and a target Y.

    Expected shapes:
    - X: (n_samples, n_features)
    - Y: (n_samples,) or (n_samples, 1)

    Returns:
    - r: 1D numpy array of shape (n_features,), Pearson coefficients per feature.
    - idx_sorted (optional): indices sorted by descending |r| if return_sorted=True
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be 2D: (n_samples, n_features)")
    n_samples, n_features = X.shape
    if y.ndim != 1:
        raise ValueError("Y must be 1D or convertible to 1D (n_samples,)")
    if y.shape[0] != n_samples:
        raise ValueError(f"Sample count mismatch: X has {n_samples}, Y has {y.shape[0]}")

    # Center columns (features) and target
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    # Numerators: covariance (unnormalized) for each feature with y
    num = Xc.T @ yc  # shape: (n_features,)

    # Denominators: std_x * std_y
    denom_x = np.sqrt(np.sum(Xc**2, axis=0))  # (n_features,)
    denom_y = np.sqrt(np.sum(yc**2))          # scalar

    # Avoid division by zero: if std is zero, set r to 0.0 for that feature
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / (denom_x * denom_y)
    r = np.where(np.isfinite(r), r, 0.0)

    if print_output:
        import numpy as _np
        _np.set_printoptions(precision=4, suppress=True)
        print("Pearson r per feature (columns of X):")
        print(r)
        if return_sorted:
            idx_sorted = _np.argsort(-_np.abs(r))
            print("Top features by |r| (WARNING index 0! remember + 1: r):")
            for i in idx_sorted:
                print(f"  {i}: {r[i]:.4f}")

    if return_sorted:
        idx_sorted = np.argsort(-np.abs(r))
        return r, idx_sorted
    return r
