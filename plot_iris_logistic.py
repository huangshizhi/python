# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:23:20 2017

@author: huangshizhi
3类logistic 回归
http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py

分别讨论了3类回归分析和多维度的2类回归分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()



X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

'''
#分成两类的高维特征
X = iris.data[:, :2][:100]  # we only take the first two features.
Y = iris.target[:100]
'''

'''
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None) # 加载Iris数据集作为DataFrame对象
X_train = df.iloc[:, [0, 2]].values # 取出2个特征，并把它们用Numpy数组表示
'''

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5,random_state = 42)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

#y_pred = logreg.predict(X)
'''
#得到显性的计算公式，分别得到三列的概率
a = logreg.coef_ 

b = logreg.intercept_

def model(x):
    return 1 / (1 + np.exp(-x))
   
#按照模型求得的最后结果    
y_1 = model(np.dot(a[0],X[50]) + b[0])
y_2 = model(np.dot(a[1],X[50]) + b[1])
y_3 = model(np.dot(a[2],X[50]) + b[2])

a1 = a[0]
b1 = b[0]
x1 = np.array([5.1,3.5])

y1 = np.dot(a1,x1)+b1
model(y1)
'''
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min( ) - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1],edgecolors='k',cmap=plt.cm.Paired)

#plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
