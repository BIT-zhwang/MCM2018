import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression,LinearRegression
from scipy.interpolate import spline
from sklearn.ensemble import RandomForestRegressor
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
for i in range(0,605):
    X[i]=CA.col_values(i)
print(np.isnan(X).any())
K1=[191,564,553,540,535,532,545,79,430]
K2=[362,466,70,422,351,487,82,429,361]
K3=[488,563,467,544,549]
K=K3
plt.figure(1)
for i in range(len(K)):
    k=K[i]
    plt.subplot(331+i)
    #k=465
    k=k-2
    y=X[k]
    x=[]
    for i in range(50):
        if(y[i]!=0):
            x.append([y[i]])
    length=len(x)
    print(len(x))
    y=[]
    for i in range(length):
        y.append([2010-50+i])
    print(len(y))
    #clf=linear_model.Ridge (alpha = .5)  #另回归
    #clf = LinearRegression()   #线型回归
    #clf = svm.SVR()
    #clf = linear_model.Lasso(alpha = 0.1)
    #clf = LogisticRegression()
    #clf = DecisionTreeRegressor (max_depth=8)
    clf = RandomForestRegressor()
    clf.fit(y, x)
    
    #模型拟合测试集
    y_pred = clf.predict(y)
    print(clf.predict([2050]))
    # 用scikit-learn计算MSE
    print("MSE:",metrics.mean_squared_error(x, y_pred))
    # 用scikit-learn计算RMSE
    print("RMSE:",np.sqrt(metrics.mean_squared_error(x, y_pred)))
    plt.scatter(y,x)
    plt.plot(y,clf.predict(y))
    plt.title(str(k))
plt.show()