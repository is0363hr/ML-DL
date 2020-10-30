# -----------------------
# import ----------------
# -----------------------
import sys
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as ps

# def SVM():
#     model = SVC(kernel='linear', random_state=None)
#     print(type(model))
#     return model

# def logistic():
#     model = LogisticRegression(random_state=None)
#     return model

model = []

method = sys.argv[1]
# アヤメデータセットを用いる
iris = datasets.load_iris()
# 例として、3,4番目の特徴量の2次元データで使用
x = iris.data[:, [1,2]]
#クラスラベルを取得
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None)

# データの標準化処理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

if method == 'svm':
    # 線形SVMのインスタンスを生成
    model = SVC(kernel='linear', random_state=None)
elif method == 'logistic':
    # 線形SVMのインスタンスを生成
    model = LogisticRegression(random_state=None)

# モデルの学習。fit関数で行う。
model.fit(X_train_std, y_train)

# トレーニングデータに対する精度
pred_train = model.predict(X_train_std)
accuracy_train = accuracy_score(y_train, pred_train)
print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

pred_test = model.predict(X_test_std)
accuracy_test = accuracy_score(y_test, pred_test)
print('テストデータに対する正解率： %.2f' % accuracy_test)