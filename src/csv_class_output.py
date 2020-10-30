#データセットをcsvファイルに変換
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


iris = datasets.load_breast_cancer()


feature_name=np.insert(iris.feature_names,len(iris.feature_names),"label")
print(feature_name)

count = 1
with open('./data/breast_cancer.csv','w') as f:
	writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
	writer.writerow(feature_name)

	for data , target in zip(iris.data, iris.target):
		data_c = np.insert(data, len(data), target)
		writer.writerow(data_c)
