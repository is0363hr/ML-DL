##############################
# https://newtechnologylifestyle.net/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92-%E3%82%A2%E3%83%A4%E3%83%A1%E3%81%AE%E5%88%86%E9%A1%9E-%E8%A9%A6%E9%A8%93%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E3%83%97%E3%83%AD%E3%83%83%E3%83%88/
##############################


from sklearn import datasets
from matplotlib import pyplot as plt
import itertools


def main():
    iris = datasets.load_iris()

    # アヤメの各種データ
    features = iris.data
    # 各特徴量の名前
    feature_names = iris.feature_names
    # データと品種の対応
    targets = iris.target

    # グラフの全体サイズを指定する
    plt.figure(figsize=(20, 8))

    # 特徴量の組み合わせ
    for i, (x, y) in enumerate(itertools.combinations(range(4), 2)):
        plt.subplot(2, 3, i + 1)
        for t, marker, c in zip(range(3), '>ox', 'rgb'):
            plt.scatter(
                features[targets == t, x],
                features[targets == t, y],
                marker=marker,
                c=c,
            )
            plt.xlabel(feature_names[x])
            plt.ylabel(feature_names[y])
            plt.autoscale()

    plt.show()


if __name__ == '__main__':
    main()
