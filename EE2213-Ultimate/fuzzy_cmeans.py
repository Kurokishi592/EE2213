import numpy as np

# Assignment Step: Fix centers, update membership
def update_membership(data_points, centers, fuzzier=2):
		'''
		Parameters:
				datapoints: ndarray of shape (n_samples, n_features)
				centers: ndarray of shape (n_clusters, n_features)
				fuzzier: fuzzifier ([1.25,2])
				
		Returns:
				W: ndarray of shape (n_samples, n_clusters)
		'''
		n_samples = data_points.shape[0]
		n_clusters = centers.shape[0]
		W = np.zeros((n_samples, n_clusters)) # initialize membership matrix

		for i in range(n_samples):
				for k in range(n_clusters):
						denom = 0.0 # Denominator for membership calculation

						# Calculate ||x_i - c_k||
						dist_k = np.linalg.norm(data_points[i] - centers[k]) + 1e-10  # Avoid division by zero

						for j in range(n_clusters):
								# Calculate ||x_i - c_j||
								dist_j = np.linalg.norm(data_points[i] - centers[j]) + 1e-10
		
								ratio = (dist_k / dist_j)
								denom += ratio ** (2 / (fuzzier - 1))
						W[i, k] = 1 / denom
		return W

# Centroid Update Step: Fix membership, update centers
def update_centers(data_points, W, fuzzier=2):
		'''
		Parameters:
				datapoints: ndarray of shape (n_samples, n_features)
				W: ndarray of shape (n_samples, n_clusters)
				fuzzier: fuzzifier ([1.25,2])
				
		Returns:
				centers: ndarray of shape (n_clusters, n_features)
		'''
		n_clusters = W.shape[1]
		centers = np.zeros((n_clusters, data_points.shape[1]))

		for k in range(n_clusters):
				numerator = data_points.T @ (W[:, k] ** fuzzier)
				denominator = np.sum(W[:, k] ** fuzzier)
				centers[k] = numerator / denominator
		return centers

# Fuzzy_Cmeans Clustering
def fuzzy_Cmeans(data_points, centers_init, fuzzier=2, max_iterations=100, tol=1e-4, verbose=False):
	"""Fuzzy C-Means clustering.

	Parameters:
		data_points : ndarray (n_samples, n_features)
		centers_init: ndarray (n_clusters, n_features)
		fuzzier     : fuzzifier (>1), commonly 2
		max_iterations : cap on iterations
		tol         : convergence tolerance on center shift (L2 norm)
		verbose     : if True, print per-iteration centers, distance matrix, membership matrix

	Returns:
		centers : final cluster centers
		W       : final membership matrix (n_samples, n_clusters)
	"""
	centers = centers_init.copy()  # work copy
	W = None  # to hold latest membership
	for it in range(1, max_iterations+1):
		# Membership update (assignment step) for current centers
		W = update_membership(data_points, centers, fuzzier)
		# Distance matrix based on current centers
		dist_matrix = np.linalg.norm(data_points[:, None, :] - centers[None, :, :], axis=2)
		# Compute new centers from current membership
		new_centers = update_centers(data_points, W, fuzzier=fuzzier)
		shift = np.linalg.norm(new_centers - centers)
		if verbose:
			print(f"Iter {it}:")
			print("Centers (before update):\n", centers)
			print("Distance matrix (rows samples, cols clusters):\n", dist_matrix)
			print("Membership matrix W (rows samples, cols clusters):\n", W)
			print("Hard cluster assignments (argmax per sample):\n", np.argmax(W, axis=1))
			print("Proposed new centers:\n", new_centers)
			print(f"Center shift L2: {shift:.6f}\n")
		# Convergence check
		centers = new_centers
		if shift < tol:
			break
	# Recompute membership for final centers to align outputs
	W_final = update_membership(data_points, centers, fuzzier)
	if verbose:
		final_dist = np.linalg.norm(data_points[:, None, :] - centers[None, :, :], axis=2)
		print("Final Centers:\n", centers)
		print("Final Distance matrix:\n", final_dist)
		print("Final Membership matrix W:\n", W_final)
		print("Final hard cluster assignments (argmax per sample):\n", np.argmax(W_final, axis=1))
	return centers, W_final