import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

''' 数据生成
x = np.arange(0, 1, 0.002)
y = norm.rvs(0, size=500, scale=0.1)
y = y + x**2
'''
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
k=430
k=k-2
y=X[k]
x=[]
for i in range(50):
    if(y[i]!=0):
        x.append(y[i])
length=len(x)
print(len(x))
y=[]
for i in range(length):
    y.append(2010-50+i)
print(len(y))
temp=np.array(x)
x=np.array(y)
y=temp
x=x/2050
''' 均方误差根 '''
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

''' 与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档  '''
def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()


''' 这是Conway&White《机器学习使用案例解析》里的版本 '''
def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)


plt.scatter(x, y, s=5)
y_test = []
y_test = np.array(y_test)



clf = Pipeline([('poly', PolynomialFeatures(degree=1)),
    ('linear', LinearRegression(fit_intercept=False))])
clf.fit(x[:, np.newaxis], y)
y_test = clf.predict(x[:, np.newaxis])

print(clf.named_steps['linear'].coef_)
print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f'
        %(rmse(y_test, y),
        R2(y_test, y),
        R22(y_test, y),
        clf.score(x[:, np.newaxis], y)))
print('2025年预测值:%.4f,2050年预测值:%.4f.'%(float(clf.predict(np.array([[0.9878]]))),float(clf.predict(np.array([[1]])))))
plt.plot(x, y_test, linewidth=2)
plt.grid()
#plt.legend(['1','2','4','8','10'], loc='upper left')
plt.show()

