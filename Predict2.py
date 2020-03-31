import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import xlrd
import xlwt
workbook1=xlwt.Workbook()
worksheet = workbook1.add_sheet('My Worksheet')

workbook=xlrd.open_workbook(r'C:\Users\Leibniz\Documents\MATLAB\C\ProblemCData.xlsx')
TX=workbook.sheet_by_name('TX for PY')
AZ=workbook.sheet_by_name('AZ for PY')
NM=workbook.sheet_by_name('NM for PY')
CA=workbook.sheet_by_name('CA for PY')
MSN=workbook.sheet_by_name('MSN')
MSNlist=MSN.col_values(0)
X1=np.zeros([605,50])
X2=np.zeros([605,50])
X3=np.zeros([605,50])
X4=np.zeros([605,50])
for i in range(0,605):
    X1[i] = AZ.col_values(i)
    X2[i] = CA.col_values(i)
    X3[i] = NM.col_values(i)
    X4[i] = TX.col_values(i)
K=[191,564,553,540,535,532,545,79,430,362,466,
   70,422,351,487,82,429,361,198,142,208,606,
   530,488,563,467,544,549,383]
#均方误差根 '''
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))
#与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档  '''
def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()
#这是Conway&White《机器学习使用案例解析》里的版本 '''
def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)
count=0
for k in K:
    worksheet.write(count, 0, k)
    plt.figure(k)
    plt.subplot(221)
    k = k - 2
    y = X1[k]
    x = []
    for i in range(30):
        if (y[i+20] != 0):
            x.append(y[i])
    length = len(x)
    #print(len(x))
    y = []
    for i in range(length):
        y.append(2010 - 30 + i)
    #print(len(y))
    temp = np.array(x)
    x = np.array(y)
    y = temp
    x = x / 2050
    plt.scatter(x, y, s=5)
    y_test = []
    y_test = np.array(y_test)
    clf = Pipeline([('poly', PolynomialFeatures(degree=1)),
                    ('linear', LinearRegression(fit_intercept=False))])
    clf.fit(x[:, np.newaxis], y)
    y_test = clf.predict(x[:, np.newaxis])
    #print(clf.named_steps['linear'].coef_)
    print('AZ&%d:' %(k+2))
    print('degree=%d'%(1))
    print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f'
          % (rmse(y_test, y),
             R2(y_test, y),
             R22(y_test, y),
             clf.score(x[:, np.newaxis], y)))
    print('2025年预测值:%.4f,2050年预测值:%.4f.'%(float(clf.predict(np.array([[0.9878]]))),float(clf.predict(np.array([[1]])))))
    worksheet.write(count, 1, float(clf.predict(np.array([[0.9878]]))))
    worksheet.write(count, 2, float(clf.predict(np.array([[1]]))))
    worksheet.write(count, 10, label=MSNlist[k])
    x=[]
    for i in range(1960,2025):
        x.append([i/2050])
    y_test=clf.predict(x)
    plt.plot(x, y_test, linewidth=2)
    plt.title(str(MSNlist[k])+' of AZ')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(222)
    y = X2[k]
    x = []
    for i in range(50):
        if (y[i] != 0):
            x.append(y[i])
    length = len(x)
    # print(len(x))
    y = []
    for i in range(length):
        y.append(2010 - 50 + i)
    # print(len(y))
    temp = np.array(x)
    x = np.array(y)
    y = temp
    x = x / 2050
    plt.scatter(x, y, s=5)
    y_test = []
    y_test = np.array(y_test)
    clf = Pipeline([('poly', PolynomialFeatures(degree=1)),
                    ('linear', LinearRegression(fit_intercept=False))])
    clf.fit(x[:, np.newaxis], y)
    y_test = clf.predict(x[:, np.newaxis])
    #print(clf.named_steps['linear'].coef_)
    print('CA&%d:' %(k+2))
    print('degree=%d'%(1))
    # print(clf.named_steps['linear'].coef_)
    #print('CA&%d:' % k)
    print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f'
          % (rmse(y_test, y),
             R2(y_test, y),
             R22(y_test, y),
             clf.score(x[:, np.newaxis], y)))
    print('2025年预测值:%.4f,2050年预测值:%.4f.' % (float(clf.predict(np.array([[0.9878]]))), float(clf.predict(np.array([[1]])))))
    worksheet.write(count, 3, float(clf.predict(np.array([[0.9878]]))))
    worksheet.write(count, 4, float(clf.predict(np.array([[1]]))))
    x = []
    for i in range(1960, 2025):
        x.append([i / 2050])
    y_test = clf.predict(x)
    plt.plot(x, y_test, linewidth=2)
    plt.title(str(MSNlist[k])+ ' of CA')
    plt.xticks([])
    plt.yticks([])

    if(k!=381):
        plt.subplot(223)
        y = X3[k]
        x = []
        for i in range(50):
            if (y[i] != 0):
                x.append(y[i])
        length = len(x)
        # print(len(x))
        y = []
        for i in range(length):
            y.append(2010 - 50 + i)
        # print(len(y))
        temp = np.array(x)
        x = np.array(y)
        y = temp
        x = x / 2050
        plt.scatter(x, y, s=5)
        y_test = []
        y_test = np.array(y_test)
        clf = Pipeline([('poly', PolynomialFeatures(degree=1)),
                        ('linear', LinearRegression(fit_intercept=False))])
        clf.fit(x[:, np.newaxis], y)
        y_test = clf.predict(x[:, np.newaxis])
        #print(clf.named_steps['linear'].coef_)
        print('NM&%d:' %(k+2))
        print('degree=%d'%(1))
        # print(clf.named_steps['linear'].coef_)
        #print('NM&%d:' % k)
        print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f'
              % (rmse(y_test, y),
                 R2(y_test, y),
                 R22(y_test, y),
                 clf.score(x[:, np.newaxis], y)))
        print('2025年预测值:%.4f,2050年预测值:%.4f.' % (float(clf.predict(np.array([[0.9878]]))), float(clf.predict(np.array([[1]])))))
        worksheet.write(count, 5, float(clf.predict(np.array([[0.9878]]))))
        worksheet.write(count, 6, float(clf.predict(np.array([[1]]))))
        x = []
        for i in range(1960, 2025):
            x.append([i / 2050])
        y_test = clf.predict(x)

        plt.plot(x, y_test, linewidth=2)
        plt.title(str(MSNlist[k]) + ' of NM')
        plt.xticks([])
        plt.yticks([])
    plt.subplot(224)
    y = X4[k]
    x = []
    for i in range(50):
        if (y[i] != 0):
            x.append(y[i])
    length = len(x)
    # print(len(x))
    y = []
    for i in range(length):
        y.append(2010 - 50 + i)
    # print(len(y))
    temp = np.array(x)
    x = np.array(y)
    y = temp
    x = x / 2050
    plt.scatter(x, y, s=5)
    y_test = []
    y_test = np.array(y_test)
    clf = Pipeline([('poly', PolynomialFeatures(degree=1)),
                    ('linear', LinearRegression(fit_intercept=False))])
    clf.fit(x[:, np.newaxis], y)
    y_test = clf.predict(x[:, np.newaxis])
    # print(clf.named_steps['linear'].coef_)
    #print(clf.named_steps['linear'].coef_)
    print('TX&%d:' %(k+2))
    print('degree=%d'%(1))
    #print('TX&%d:' %k)
    print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f'
          % (rmse(y_test, y),
             R2(y_test, y),
             R22(y_test, y),
             clf.score(x[:, np.newaxis], y)))
    print('2025年预测值:%.4f,2050年预测值:%.4f.' % (float(clf.predict(np.array([[0.9878]]))), float(clf.predict(np.array([[1]])))))
    worksheet.write(count, 7, float(clf.predict(np.array([[0.9878]]))))
    worksheet.write(count, 8, float(clf.predict(np.array([[1]]))))
    x = []
    for i in range(1960, 2025):
        x.append([i / 2050])
    y_test = clf.predict(x)
    plt.plot(x, y_test, linewidth=2)
    plt.title(str(MSNlist[k]) + ' of TX')
    plt.xticks([])
    plt.yticks([])
    count=count+1

workbook1.save('Excel_Workbook.xls')
plt.show()