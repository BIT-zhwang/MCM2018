# -*- coding:utf-8 -*-
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist

workbook=xlrd.open_workbook(r'C:\Users\Leibniz\Documents\MATLAB\C\ProblemCData.xlsx')
TX=workbook.sheet_by_name('TX for PY')
AZ=workbook.sheet_by_name('AZ for PY')
NM=workbook.sheet_by_name('NM for PY')
CA=workbook.sheet_by_name('CA for PY')
MSN=workbook.sheet_by_name('MSN2')
MSNlist=MSN.col_values(0)
X=np.zeros([605,50])
for i in range(0,605):
    X[i]=AZ.col_values(i)
#print(X)
#数据归一化

for i in range(0,605):
    Xmin=X[i].min()
    Xmax=X[i].max()
    if(Xmax==Xmin):
        X[i]=np.zeros(50)
    else:
        X[i]=(X[i]-Xmin)/(Xmax-Xmin)
#print(X)

print(np.isnan(X).any())


K=range(1,100)
meandistortions=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(
            cdist(X,kmeans.cluster_centers_,
                 'euclidean'),axis=1))/X.shape[0])
plt.figure()
plt.plot(K,meandistortions,'b')
plt.xlabel('k')
plt.ylabel('Average distortion.')
plt.title('Use the elbow rule to determine the best K value')

#print(meandistortions)
#plt.show()
'''
#另一种算法
K=range(1,50)
test=[]
for k in range(1,50):
    clf = KMeans(n_clusters=k)
    clf.fit(X)
    test.append(clf.inertia_)
plt.figure()
plt.plot(K,test,'b')
'''
clf = KMeans(n_clusters=15)
clf.fit(X)
a0=[]
a1=[]
a2=[]
a3=[]
a4=[]
a5=[]
a6=[]
a7=[]
a8=[]
a9=[]
a10=[]
a11=[]
a12=[]
a13=[]
a14=[]
b0=[]
b1=[]
b2=[]
b3=[]
b4=[]
b5=[]
b6=[]
b7=[]
b8=[]
b9=[]
b10=[]
b11=[]
b12=[]
b13=[]
b14=[]

'''
a15=[]
a16=[]
a17=[]
a18=[]
a19=[]
a20=[]
a21=[]
a22=[]
a23=[]
a24=[]
'''
for i in range(0,605):
    if(clf.labels_[i]==0):
        a0.append(MSNlist[i])
    if (clf.labels_[i]== 1):
        a1.append(MSNlist[i])
    if (clf.labels_[i]== 2):
        a2.append(MSNlist[i])
    if (clf.labels_[i] == 3):
        a3.append(MSNlist[i])
    if (clf.labels_[i]== 4):
        a4.append(MSNlist[i])
    if (clf.labels_[i]== 5):
        a5.append(MSNlist[i])
    if (clf.labels_[i]== 6):
        a6.append(MSNlist[i])
    if (clf.labels_[i] == 7):
        a7.append(MSNlist[i])
    if (clf.labels_[i] == 8):
        a8.append(MSNlist[i])
    if (clf.labels_[i] == 9):
        a9.append(MSNlist[i])
    if (clf.labels_[i] == 10):
        a10.append(MSNlist[i])
    if (clf.labels_[i] == 11):
        a11.append(MSNlist[i])
    if (clf.labels_[i] == 12):
        a12.append(MSNlist[i])
    if (clf.labels_[i] == 13):
        a13.append(MSNlist[i])
    if (clf.labels_[i] == 14):
        a14.append(MSNlist[i])
    '''
    if (clf.labels_[i] == 15):
        a15.append(MSNlist[i])
    if (clf.labels_[i] == 16):
        a16.append(MSNlist[i])
    if (clf.labels_[i] == 17):
        a17.append(MSNlist[i])
    if (clf.labels_[i] == 18):
        a18.append(MSNlist[i])
    if (clf.labels_[i] == 19):
        a19.append(MSNlist[i])
    if (clf.labels_[i] == 20):
        a20.append(MSNlist[i])
    if (clf.labels_[i] == 21):
        a21.append(MSNlist[i])
    if (clf.labels_[i] == 22):
        a22.append(MSNlist[i])
    if (clf.labels_[i] == 23):
        a23.append(MSNlist[i])
    if (clf.labels_[i] == 24):
        a24.append(MSNlist[i])
    '''
for i in range(0,605):
    if(clf.labels_[i]==0):
        b0.append(i)
    if (clf.labels_[i]== 1):
        b1.append(i)
    if (clf.labels_[i]== 2):
        b2.append(i)
    if (clf.labels_[i] == 3):
        b3.append(i)
    if (clf.labels_[i]== 4):
        b4.append(i)
    if (clf.labels_[i]== 5):
        b5.append(i)
    if (clf.labels_[i]== 6):
        b6.append(i)
    if (clf.labels_[i] == 7):
        b7.append(i)
    if (clf.labels_[i] == 8):
        b8.append(i)
    if (clf.labels_[i] == 9):
        b9.append(i)
    if (clf.labels_[i] == 10):
        b10.append(i)
    if (clf.labels_[i] == 11):
        b11.append(i)
    if (clf.labels_[i] == 12):
        b12.append(i)
    if (clf.labels_[i] == 13):
        b13.append(i)
    if (clf.labels_[i] == 14):
        b14.append(i)
#test.append(clf.inertia_)
print('中文介绍：')
print(a0)
print(a1)
print(a2)
print(a3)
print(a4)
print(a5)
print(a6)
print(a7)
print(a8)
print(a9)
print(a10)
print(a11)
print(a12)
print(a13)
print(a14)
print('序号：')
print(b0)
print(b1)
print(b2)
print(b3)
print(b4)
print(b5)
print(b6)
print(b7)
print(b8)
print(b9)
print(b10)
print(b11)
print(b12)
print(b13)
print(b14)
'''
print(a15)
print(a16)
print(a17)
print(a18)
print(a19)
print(a20)
print(a21)
print(a22)
print(a23)
print(a24)
'''
print('label：')
print(clf.labels_)
print('distance:')
print(clf.inertia_)
plt.show()
