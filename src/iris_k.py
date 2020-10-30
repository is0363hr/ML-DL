# scikit-learnからデータの取り出し
from keras.layers import Dense, Activation
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split as split
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
# アヤメの分類に使用するデータの確認
learn_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
learn_label = [0, 1, 1, 0]
print("aaaaa")
# アルゴリズムを指定。K最近傍法を採用
clf = KNeighborsClassifier(n_neighbors=1)

# 学習用のデータと結果を学習する,fit()
clf.fit(learn_data, learn_label)

# テストデータによる予測,predict()
test_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
test_label = clf.predict(test_data)

# テスト結果を評価する,accuracy_score()
print("予測対象：", test_data, ", 予測結果→", test_label)
print("正解率＝", accuracy_score([0, 1, 1, 0], test_label))
