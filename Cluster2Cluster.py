import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from scipy.spatial.distance import cdist

workbook=xlrd.open_workbook(r'C:\Users\Leibniz\Documents\MATLAB\C\ProblemCData.xlsx')
AZ=workbook.sheet_by_name('AZ for PY')
MSN=workbook.sheet_by_name('MSN2')
MSNlist=MSN.col_values(0)
X=np.zeros([605,50])

for i in range(0,605):
    X[i]=AZ.col_values(i)
print(np.isnan(X).any())
'''
clf0=KMeans(n_clusters=10)
clf0.fit(X)
joblib.dump(clf0,"train_model.m")

first=clf.labels_
for i in range(10):
    clf = KMeans(n_clusters=10)
    clf.fit(X)
    if(first.any()==clf.labels_.any()):
        print('True')
    else:
        print('False')
'''
clf0=joblib.load("train_model.m")
print('label：')
print(clf0.labels_)
print('distance:')
print(clf0.inertia_)

x0,x1,x2,x3,x4,x5,x6,x7,x8,x9=[],[],[],[],[],[],[],[],[],[]
y0,y1,y2,y3,y4,y5,y6,y7,y8,y9=[],[],[],[],[],[],[],[],[],[]

for i in range(605):
    if (clf0.labels_[i] == 0):
        x0.append(i)
        y0.append(MSNlist[i])
    if (clf0.labels_[i] == 1):
        x1.append(i)
        y1.append(MSNlist[i])
    if (clf0.labels_[i] == 2):
        x2.append(i)
        y2.append(MSNlist[i])
    if (clf0.labels_[i] == 3):
        x3.append(i)
        y3.append(MSNlist[i])
    if (clf0.labels_[i] == 4):
        x4.append(i)
        y4.append(MSNlist[i])
    if (clf0.labels_[i] == 5):
        x5.append(i)
        y5.append(MSNlist[i])
    if (clf0.labels_[i] == 6):
        x6.append(i)
        y6.append(MSNlist[i])
    if (clf0.labels_[i] == 7):
        x7.append(i)
        y7.append(MSNlist[i])
    if (clf0.labels_[i] == 8):
        x8.append(i)
        y8.append(MSNlist[i])
    if (clf0.labels_[i] == 9):
        x9.append(i)
        y9.append(MSNlist[i])
X_=[x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
Y_=[y0,y1,y2,y3,y4,y5,y6,y7,y8,y9]
for i in range(10):
    print(len(X_[i]))
    print(Y_[i])

#归一化
for i in range(605):
    Xmin=X[i].min()
    Xmax=X[i].max()
    if(Xmax==Xmin):
        X[i]=np.zeros(50)
    else:
        X[i]=(X[i]-Xmin)/(Xmax-Xmin)

#选择0列归一化聚类4层
m0,m1,m2,m3=[],[],[],[]
n0,n1,n2,n3=[],[],[],[]
print('归一化后第一列数据聚类:')
X0=np.zeros([len(x0),50])
for i in range(len(x0)):
    X0[i]=X[x0[i]]
clf=KMeans(n_clusters=4)
clf.fit(X0)
print('label：')
print(clf.labels_)
print('distance:')
print(clf.inertia_)
for i in range(len(x0)):
    if (clf.labels_[i] == 0):
        m0.append(x0[i])
        n0.append(MSNlist[x0[i]])
    if (clf.labels_[i] == 1):
        m1.append(x0[i])
        n1.append(MSNlist[x0[i]])
    if (clf.labels_[i] == 2):
        m2.append(x0[i])
        n2.append(MSNlist[x0[i]])
    if (clf.labels_[i] == 3):
        m3.append(x0[i])
        n3.append(MSNlist[x0[i]])

M=[m0,m1,m2,m3]
N=[n0,n1,n2,n3]
for i in range(4):
    print(len(M[i]))
    print(N[i])

#选择1列归一化聚类15层
m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
print('归一化后第2列数据聚类:')
X1=np.zeros([len(x1),50])
for i in range(len(x1)):
    X1[i]=X[x1[i]]
clf=KMeans(n_clusters=15)
clf.fit(X1)
print('label：')
print(clf.labels_)
print('distance:')
print(clf.inertia_)
for i in range(len(x1)):
    if (clf.labels_[i] == 0):
        m0.append(x1[i])
        n0.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 1):
        m1.append(x1[i])
        n1.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 2):
        m2.append(x1[i])
        n2.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 3):
        m3.append(x1[i])
        n3.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 4):
        m4.append(x1[i])
        n4.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 5):
        m5.append(x1[i])
        n5.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 6):
        m6.append(x1[i])
        n6.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 7):
        m7.append(x1[i])
        n7.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 8):
        m8.append(x1[i])
        n8.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 9):
        m9.append(x1[i])
        n9.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 10):
        m10.append(x1[i])
        n10.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 11):
        m11.append(x1[i])
        n11.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 12):
        m12.append(x1[i])
        n12.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 13):
        m13.append(x1[i])
        n13.append(MSNlist[x1[i]])
    if (clf.labels_[i] == 14):
        m14.append(x1[i])
        n14.append(MSNlist[x1[i]])
M=[m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14]
N=[n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14]
for i in range(15):
    print(len(M[i]))
    print(N[i])

#选择5列归一化聚类2层
m0,m1=[],[]
n0,n1=[],[]
print('归一化后第6列数据聚类:')
X5=np.zeros([len(x5),50])
for i in range(len(x5)):
    X5[i]=X[x5[i]]
clf=KMeans(n_clusters=2)
clf.fit(X5)
print('label：')
print(clf.labels_)
print('distance:')
print(clf.inertia_)
for i in range(len(x5)):
    if (clf.labels_[i] == 0):
        m0.append(x5[i])
        n0.append(MSNlist[x5[i]])
    if (clf.labels_[i] == 1):
        m1.append(x5[i])
        n1.append(MSNlist[x5[i]])

M=[m0,m1]
N=[n0,n1]
for i in range(2):
    print(len(M[i]))
    print(N[i])

#选择9列归一化聚类7层
m0,m1,m2,m3,m4,m5,m6=[],[],[],[],[],[],[]
n0,n1,n2,n3,n4,n5,n6=[],[],[],[],[],[],[]
print('归一化后第10列数据聚类:')
X9=np.zeros([len(x9),50])
for i in range(len(x9)):
    X9[i]=X[x9[i]]
clf=KMeans(n_clusters=7)
clf.fit(X9)
print('label：')
print(clf.labels_)
print('distance:')
print(clf.inertia_)
for i in range(len(x9)):
    if (clf.labels_[i] == 0):
        m0.append(x9[i])
        n0.append(MSNlist[x9[i]])
    if (clf.labels_[i] == 1):
        m1.append(x9[i])
        n1.append(MSNlist[x9[i]])
    if (clf.labels_[i] == 2):
        m2.append(x9[i])
        n2.append(MSNlist[x9[i]])
    if (clf.labels_[i] == 3):
        m3.append(x9[i])
        n3.append(MSNlist[x9[i]])
    if (clf.labels_[i] == 4):
        m4.append(x9[i])
        n4.append(MSNlist[x9[i]])
    if (clf.labels_[i] == 5):
        m5.append(x9[i])
        n5.append(MSNlist[x9[i]])
    if (clf.labels_[i] == 6):
        m6.append(x9[i])
        n6.append(MSNlist[x9[i]])
M=[m0,m1,m2,m3,m4,m5,m6]
N=[n0,n1,n2,n3,n4,n5,n6]
for i in range(7):
    print(len(M[i]))
    print(N[i])






