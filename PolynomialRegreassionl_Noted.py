# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.getcwd()

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values#因为只需要level作为因变量，包含自变量的应该是一个矩阵，所有要加上stop at column2
y = dataset.iloc[:, 2].values


#因为数据太少了，所以不分割训练和测试集。

#建立简单线性模型
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#建立多项式模型
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)#第一个参数就是指数，设为2
X_poly = poly_reg.fit_transform(X) #用多项式feature来拟合并转化原本的level，
#打开X_poly就可以看到三列，分别对应常数项，X的一次方，X的二次方。 
lin_reg_2 = LinearRegression() #然后建立第二个线性模型
lin_reg_2.fit(X_poly, y)#并用多项式feature处理过的X——poly来拟合一个新的多项式线性regressor

#visualizing the Linear Regression result
plt.scatter(X, y, color = 'red') #实际的结果
plt.plot(X, lin_reg.predict(X), color = 'blue') #预测的结果
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualizing the polynomial regression result
X_grid = np.arange(min(X),max(X), 0.1)#第一个参数是开始，第二个是最大的自变量，所以是从大到小，最后一个是X值之间的spacing单位，可以是1，也可以是0.1，0.01.。。
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red') #实际的结果
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') #预测的结果
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#多项式更好的表现了高级的工资，现在这个图是在多项式为指数2 的情况下，我们可以增加polynomial feature degree为3
#一直到四次多项式模型，模型预测的线才越接近实际的点，但是因为这个position level是以1为单位增加的，所以这个预测线会显示得比较不够平滑。
#所以我们可以使用np中的arrange的功能来改变图形中线性表示,通过把间隔的单位从1改为0.1. 这个线的展示会更加细节和平滑。

#用多项式模型来预测新人的薪水
lin_reg.predict(np.array([6.5]).reshape(1, 1))
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
lin_reg.predict([[6.5]])#跟第一个是一样的预测结果，因为6.5只是X的值，但是这个是2维的空间。所以需要按格式写。