import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
import pandas as pd


xigua = pd.read_csv('D:/Desktop/MSSB/ClusterSamples.csv')


estimator = KMeans(n_clusters=10,max_iter=2,)
#计算每个样本的聚类中心并预测聚类索引。
a1=xigua.values
a=a1.shape
print(a)
res = estimator.fit_predict(a1[:,1:788])
#每个点的标签
lable_pred = estimator.labels_
#每个点的聚类中心
centroids = estimator.cluster_centers_
#样本距其最近的聚类中心的平方距离之和。
inertia = estimator.inertia_
print(lable_pred)
