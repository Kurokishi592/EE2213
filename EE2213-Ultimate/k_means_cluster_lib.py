import numpy as np
from sklearn.cluster import KMeans

x1 = np.array([0, 0])
x2 = np.array([1, 0])
x3 = np.array([0, 1])
x4 = np.array([1, 1])

data_points = np.array([x1, x2, x3, x4])

c1_init = x1.copy()
c2_init = x2.copy()

centers_init = np.array([c1_init, c2_init])

kmeans = KMeans(n_clusters=2, init=centers_init, max_iter=100, n_init=1) 
# n_init: The number of times the KMeans algorithm will run with different centroid seeds
# Setting n_init=1 means it will only run once, using the given centers_init
kmeans.fit(data_points)
print("KMeans centers from sklearn:", kmeans.cluster_centers_)
print("KMeans labels from sklearn:", kmeans.labels_)
