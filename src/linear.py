#重回帰分析（csvファイル、dateset）に対応
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import seaborn as sns

#csvファイルから読み込む場合
"""df = pd.read_csv('./data/regresstion/boston.csv')

X = df[['AGE','DIS']]
x1 = df[['AGE']]
x2 = df[['DIS']]
y = df[['MEDV']]
print(X)
"""
#sklearnのbostonのdatasetから読み込む場合
boston = datasets.load_iris()
df1 = pd.DataFrame(boston.data)
X = df1[[1,2]]
x1 = df1[[1]]
x2 = df1[[2]]
df2 = pd.DataFrame(boston.target)
y = df2[[0]]
print(y)
model = LinearRegression()

model.fit(X,y)

#調整されたパラメーターを見る
print(model.coef_)
#切片
model.intercept_

x = np.array([[60,3]])
print(model.predict(x))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1,x2,y,s=1)
plt.show()
