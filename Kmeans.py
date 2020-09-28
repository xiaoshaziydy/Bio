import numpy as np
import pandas as pd
class Kmeans():
    def __init__(self):
        pass

    def euclidean_distance(self,x1, x2):
        distance = 0
        # 计算平方再开根号
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return np.sqrt(distance)

    def centroids_init(self,k, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((k, n_features))
        for i in range(k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def closest_centroid(self,sample, centroids):
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = self.euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def create_clusters(self,centroids, k, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self.closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def calculate_centroids(self,clusters, k, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_labels(self,clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def kmeans(self,X, k, max_iterations):
        centroids = self.centroids_init(k, X)
        for _ in range(max_iterations):
            clusters = self.create_clusters(centroids, k, X)
            print("running...")
            for _ in range(max_iterations):
                clusters = self.create_clusters(centroids, k, X)
                # 保存当前中心点
                prev_centroids = centroids
                # 3.根据聚类结果计算新的中心点
                centroids = self.calculate_centroids(clusters, k, X)
                # 4.设定收敛条件为中心点是否发生变化
                diff = centroids - prev_centroids
                if not diff.any():
                    break
                # 返回最终的聚类标签
        return self.get_cluster_labels(clusters, X)



if __name__ == "__main__":
    model=Kmeans()
    X = pd.read_csv('D:/Desktop/MSSB/ClusterSamples.csv')
    a1 = X.values
    a = a1[:, 1:-1]
    # 每个点的标签
    labels = model.kmeans(a, 10, 1)
    for i in range(len(labels)):
        print(int(labels[i]))
    # 打印每个样本所属的类别标签
    #print(labels)

