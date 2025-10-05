import pandas as pd
import numpy as np

df=pd.read_csv('k_means/Iris.csv')

selected_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
data_matrix = df[selected_columns].to_numpy()
print(data_matrix)
print(f"shape: {data_matrix.shape}")

max_values=df[selected_columns].max()
min_values=df[selected_columns].min()

initial_centroids=np.random.uniform(min_values, max_values, (3,4))
print(initial_centroids)

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

cluster_assignments = np.zeros(data_matrix.shape[0])
for i in range(data_matrix.shape[0]):
    distances = np.array([euclidean_distance(data_matrix[i], centroid) for centroid in initial_centroids])
    cluster_assignments[i] = np.argmin(distances)
print(cluster_assignments)

cluster_assignments=np.array(cluster_assignments)

for i in range(3):
    print(f"Cluster {i}: has {np.sum(cluster_assignments==i)} points")

step=10
for i in range(step):
    for j in range(3):
        points_in_cluster = data_matrix[cluster_assignments == j]
        if len(points_in_cluster) > 0:
            initial_centroids[j] = np.mean(points_in_cluster, axis=0)
    for k in range(data_matrix.shape[0]):
        distances = np.array([euclidean_distance(data_matrix[k], centroid) for centroid in initial_centroids])
        cluster_assignments[k] = np.argmin(distances)
    for i in range(3):
        print(f"Cluster {i}: has {np.sum(cluster_assignments==i)} points")
        