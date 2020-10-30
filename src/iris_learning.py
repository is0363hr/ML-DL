#########################
# https://newtechnologylifestyle.net/tensorflow-keras%e3%81%a7%e3%81%ae%e6%95%99%e5%b8%ab%e3%81%82%e3%82%8a%e5%ad%a6%e7%bf%92%e3%80%80-%e3%82%a2%e3%83%a4%e3%83%a1%e3%81%ae%e5%88%86%e9%a1%9e/
#########################


# scikit-learnからデータの取り出し
from keras.layers import Dense, Activation
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split as split
from sklearn import datasets
iris = datasets.load_iris()

# アヤメの分類に使用するデータの確認
print(iris.DESCR)
iris.data
iris.target

# アヤメの分類の学習
x_train, x_test, y_train, y_test = split(
    iris.data, iris.target, train_size=0.8, test_size=0.2)

# ニュートラルネットワークで使用するモデル作成
model = keras.models.Sequential()       #線形モデルの宣言
model.add(Dense(units=32, input_dim=4))     #中間層32、入力層4
model.add(Activation('relu'))   #活性化関数
model.add(Dense(units=3))   #出力層3
model.add(Activation('softmax'))    #活性化関数
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])    #モデル作成


# 教師あり学習の実行
model.fit(x_train, y_train, epochs=100)

# 評価の実行
score = model.evaluate(x_test, y_test, batch_size=1)

print(score[1])

# 1つのデータに対する評価の実行方法
x = np.array([[195.1, 3.5, 1.4, 0.2]])
r = model.predict(x)
print(r)
r.argmax()
