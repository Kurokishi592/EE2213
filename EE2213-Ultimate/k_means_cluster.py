import numpy as np

# Data points
x1 = np.array([0, 0])
x2 = np.array([1, 0])
x3 = np.array([0, 1])
x4 = np.array([1, 1])

data_points = np.array([x1, x2, x3, x4])

# Initial centers
c1_init = x1.copy()
c2_init = x2.copy()

centers_init = np.array([c1_init, c2_init])

def k_means(data_points, centers_init, n_clusters, max_iterations=100, tol=1e-4) :
  
  centers = centers_init.copy() # Make a copy of the initial centers

  for _ in range(max_iterations): #The underscore _ is a throwaway variable, meaning “I don’t care about the loop variable.”
    
    # Compute squared Euclidean distances to each centroid
    # Result shape: (n_samples, k)
    distances = np.linalg.norm(data_points[:, np.newaxis] - centers, axis=2)

    # Assign each point to the index of the closest centroid
    closest_centroids = np.argmin(distances, axis=1) # Finds the index of the minimum value along each row

    # Update centroids to be the mean of the data points assigned to them
    new_centers = np.zeros((n_clusters, data_points.shape[1])) # Initialize new centers
    for i in range(n_clusters):
      new_centers[i] = data_points[closest_centroids == i].mean(axis=0)
    
    # End if centroids no longer change
    if np.linalg.norm(new_centers - centers) < tol:
      break
    centers = new_centers
  return centers, closest_centroids


# distances = np.linalg.norm(data_points[:, np.newaxis] - centers, axis=2)
# break step by step
print(data_points[:, np.newaxis]) #add a new axis between the two existing ones.
data_points[:, np.newaxis].shape

# distances = np.linalg.norm(data_points[:, np.newaxis] - centers, axis=2)
# break step by step
# NumPy automatically broadcasts centers along the second axis of data_points
data_points[:, np.newaxis] - centers_init

# distances = np.linalg.norm(data_points[:, np.newaxis] - centers, axis=2)
# break step by step
#computes the L2 norm along axis 2 
np.linalg.norm(data_points [:, np.newaxis] - centers_init, axis=2)

centers, labels = k_means(data_points, centers_init, n_clusters=2, max_iterations=100, tol=1e-4)
print("Converged centers :", centers)
print("cluster Labels :", labels)