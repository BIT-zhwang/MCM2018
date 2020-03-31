'''
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
#随机生成一个实数，范围在（0.5,1.5）之间
cluster1=np.random.uniform(0.5,1.5,(2,10))
cluster2=np.random.uniform(3.5,4.5,(2,10))
#hstack拼接操作
X=np.hstack((cluster1,cluster2)).T
print(X)
plt.figure()
plt.axis([0,5,0,5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')
#coding:utf-8
#我们计算K值从1到10对应的平均畸变程度：
from sklearn.cluster import KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist
K=range(1,10)
meandistortions=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(
            cdist(X,kmeans.cluster_centers_,
                 'euclidean'),axis=1))/X.shape[0])
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel(u'平均畸变程度')
plt.title(u'用肘部法则来确定最佳的K值')
plt.show()
'''
