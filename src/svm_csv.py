#https://qiita.com/kazuki_hayakawa/items/18b7017da9a6f73eba77
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt #pipでインストールした
from mlxtend.plotting import plot_decision_regions  #pipでインストールした
import csv
import pandas as pd
from operator import mul
from functools import reduce

#nCrの計算今回使用していない
def cmb(n,r):
	r = min(n-r,r)
	if r == 0: return 1
	over = reduce(mul, range(n, n - r, -1))
	under = reduce(mul, range(1,r + 1))
	return over // under

#変数の定義
X_train_std = 0
X_test_std = 0
y_train = 0
y_test = 0
i = 0
count = 0
data1 = []
data2 = []
col = []

df = pd.read_csv('./data/class/titanic_test.csv')
#データの抽出
num = len(df.columns) - 1
for d1 in df.columns:
	col.append(d1)
#2箇所選んだときの特徴量すべての組み合わせ
while i < num - 1:
	j = i
	for j in range(j,num-1):
		a = df.loc[:,col[num]] #分類先のラベル
		b = df.loc[:,col[i]] #特徴量一箇所
		c = df.loc[:,col[j+1]] #特徴量何箇所
		#データを配列に格納
		for p_l, p_w in zip(b,c):
			data1.append([p_l,p_w])
		for d in a:
			data2.append(d)
		#配列をndarrayに変換
		data = np.array(data1)
		target = np.array(data2, dtype=np.int64)

		# トレーニングデータとテストデータに分割。
		# 今回は訓練データを70%、テストデータは30%としている。
		# 乱数を制御するパラメータ random_state は None にすると毎回異なるデータを生成する
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=None)

		# データの標準化処理
		sc = StandardScaler()
		sc.fit(X_train)
		X_train_std = sc.transform(X_train)
		X_test_std = sc.transform(X_test)

		# 線形SVMのインスタンスを生成
		model = SVC(kernel='linear', random_state=None)
		# モデルの学習。fit関数で行う。
		model.fit(X_train_std, y_train)

		# トレーニングデータに対する精度
		pred_train = model.predict(X_train_std)
		accuracy_train = accuracy_score(y_train, pred_train)
		print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

		pred_test = model.predict(X_test_std)
		accuracy_test = accuracy_score(y_test, pred_test)
		print('テストデータに対する正解率： %.2f' % accuracy_test)
		data1.clear()
		data2.clear()
		#テストデータに対する最も高い正答率の場合のデータを格納
		if i == 0 and j == 0:
			max_accuracy_test = accuracy_test
			max_train_std = X_train_std
			max_test_std = X_test_std
			max_train = y_train
			max_test = y_test
			max_model = model
			max_data1 = i
			max_data2 = j + 1
		elif max_accuracy_test < accuracy_test:
			max_accuracy_test = accuracy_test
			max_train_std = X_train_std
			max_test_std = X_test_std
			max_train = y_train
			max_test = y_test
			max_model = model
			max_data1 = i
			max_data2 = j + 1
		print(i,j+1)
	i = i + 1

#2次元までのグラフ表示
print(len(df.columns))
print('\n')
print('最も高い正解率を導いた組み合わせは')
print('{} : {}, {} : {}'.format(max_data1,col[max_data1],max_data2,col[max_data2]))
print('最も高いテストデータに対する正解率： %.2f' % max_accuracy_test)




