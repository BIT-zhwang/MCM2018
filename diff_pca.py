import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xlrd
workbook=xlrd.open_workbook(r'C:\Users\Leibniz\Documents\MATLAB\C\ProblemCData.xlsx')
TX=workbook.sheet_by_name('TX for PY')
AZ=workbook.sheet_by_name('AZ for PY')
NM=workbook.sheet_by_name('NM for PY')
CA=workbook.sheet_by_name('CA for PY')
MSN=workbook.sheet_by_name('MSN2')
MSNlist=MSN.col_values(0)
X=np.zeros([605,50])
color=['blue','green','purple','pink','brown','red','orange','yellow','grey','teal','olive','salmon','beige','tan','aqua']
#plt.subplot(221)
for i in range(0,605):
    X[i]=AZ.col_values(i)
#X=X.T
print(np.isnan(X).any())
#print(len(X))  #50
pca=PCA(n_components=2,svd_solver='full')
#pca=PCA(n_components='mle',whiten=True)
pca.fit(X)
#print(pca.transform(X))
Y=pca.transform(X)
print(pca.explained_variance_ratio_)
print(Y)
for i in range(50):
    if((Y.T[0]==X.T[i]).all()):
        print('第一维年份:'+str(i+1960))
    if((Y.T[1]==X.T[i]).all()):
        print('第二维年份:' + str(i + 1960))