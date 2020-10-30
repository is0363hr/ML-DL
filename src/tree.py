#http://own-search-and-study.xyz/2016/08/16/python3%e3%81%a7scikit-learn%e3%81%ae%e6%b1%ba%e5%ae%9a%e6%9c%a8%e3%82%92%e6%97%a5%e6%9c%ac%e8%aa%9e%e3%83%95%e3%82%a9%e3%83%b3%e3%83%88%e3%81%a7%e7%94%bb%e5%83%8f%e5%87%ba%e5%8a%9b%e3%81%99%e3%82%8b/
#graphvizをインストールその後pythonで使用するためにcondaでgraphvizをインストールした
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus #pydotplusに変更/pipでインストールしてしまった

if __name__ == "__main__":
    #irisデータの読み込み
    iris = load_iris()

    #決定木学習
    clf = tree.DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)

    #決定木モデルの書き出し
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  #pydotplusに変更
    graph.write_png("iris.png")
