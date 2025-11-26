import numpy as np

def custom_kmeans(data_points, centers_init, n_clusters, max_iterations=100, tol=1e-4, verbose=True):
  """
  K-means clustering with per-iteration reporting.

  Prints at every iteration (if verbose):
  - Distances from each data point to each centroid (rows=samples, cols=centroids)
  - Cluster assignment (nearest centroid index per sample)
  - Centroid positions before and after the update, and their movement
  Returns final centers, labels, and final distance matrix.
  """
  # Ensure arrays and validate shapes
  X = np.asarray(data_points, dtype=float)
  C = np.asarray(centers_init, dtype=float).copy()
  if C.shape != (n_clusters, X.shape[1]):
    raise ValueError(f"centers_init must have shape (n_clusters, n_features) = {(n_clusters, X.shape[1])}, got {C.shape}")

  if verbose:
    np.set_printoptions(precision=4, suppress=True)

  for it in range(max_iterations):
    # Distances: (n_samples, n_clusters)
    distances = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
    labels = np.argmin(distances, axis=1)

    if verbose:
      print(f"Iteration {it}")
      print("Distances (rows: points, cols: centroids):\n", np.round(distances, 4))
      print("Assignments:", labels)
      print("Centers before update:\n", np.round(C, 4))

    # Update centers; handle empty clusters by keeping previous center
    new_C = C.copy()
    for k in range(n_clusters):
      mask = (labels == k)
      if np.any(mask):
        new_C[k] = X[mask].mean(axis=0)
      else:
        new_C[k] = C[k]  # keep previous center (alternative: reinit to a random point)

    # Check convergence via max shift across centers
    shifts = np.linalg.norm(new_C - C, axis=1)
    max_shift = float(np.max(shifts))
    C = new_C

    if verbose:
      print("Centers after update:\n", np.round(C, 4))
      print("Center shifts:", np.round(shifts, 4))
      print("-" * 40)

    if max_shift < tol:
      if verbose:
        print(f"Converged at iteration {it}")
      break

  # Final distances and labels for the converged centers
  distances = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
  labels = np.argmin(distances, axis=1)

  if verbose:
    print("Final centers:\n", np.round(C, 4))
    print("Final assignments:", labels)
    print("Final distances:\n", np.round(distances, 4))

  return C, labels, distances