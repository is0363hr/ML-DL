#https://qiita.com/kazuki_hayakawa/items/18b7017da9a6f73eba77
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt #pipでインストールした
from mlxtend.plotting import plot_decision_regions  #pipでインストールした

# アヤメデータセットを用いる
iris = datasets.load_wine()

# 例として、3,4番目の特徴量の2次元データで使用
X = iris.data[:, [2,3]]
print(X)
#クラスラベルを取得
y = iris.target

# トレーニングデータとテストデータに分割。
# 今回は訓練データを70%、テストデータは30%としている。
# 乱数を制御するパラメータ random_state は None にすると毎回異なるデータを生成する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None )

# データの標準化処理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

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

#2次元までのグラフ表示
plt.style.use('ggplot')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

fig = plt.figure(figsize=(13,8))
plot_decision_regions(X_combined_std, y_combined, clf=model,  res=0.02)
plt.show()

