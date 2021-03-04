from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from pprint import pprint
import numpy as np
import pandas as pd
import category_encoders as ce
import random

from statistical_processing import StatisticalProcessing
# from regression_import import reg_dict


# 機械学習モジュール

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

reg_dict = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Polynomial_deg2": Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression())]),
        "Polynomial_deg3": Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression())]),
        "Polynomial_deg4": Pipeline([('poly', PolynomialFeatures(degree=4)),('linear', LinearRegression())]),
        "Polynomial_deg5": Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression())]),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=3),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1),
        "GaussianProcessRegressor": GaussianProcessRegressor(),
        "SGDRegressor": SGDRegressor(),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(10,10), max_iter=100, early_stopping=True, n_iter_no_change=5),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100), 
        "PLSRegression": PLSRegression(n_components=10),
        "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=100, tol=1e-3),
        "TheilSenRegressor": TheilSenRegressor(random_state=0),
        "RANSACRegressor": RANSACRegressor(random_state=0),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor(random_state=0, n_estimators=100),
        "BaggingRegressor": BaggingRegressor(base_estimator=SVR(), n_estimators=10),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=0),
        "VotingRegressor": VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=10))]),
        "StackingRegressor": StackingRegressor(estimators=[('lr', RidgeCV()), ('svr', LinearSVR())], final_estimator=RandomForestRegressor(n_estimators=10)),
        "ARDRegression": ARDRegression(),
        "HuberRegressor": HuberRegressor(),
}

def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    sp = StatisticalProcessing(isDebug=False)
    day_data = sp.day_sum()
    month_data = sp.month_sum()
    day_ratio = sp.day_ratio()
    holiday_flag = sp.holiday_search()

    # 月単位で日毎の予測
    datasets = pd.DataFrame()
    datasets.loc[:, "year"] = day_data.index.year
    datasets.loc[:, "month"] = day_data.index.month
    datasets.loc[:, "day"] = day_data.index.month
    datasets.loc[:, "holiday_flag"] = holiday_flag
    # 月の合計への割合
    datasets.loc[:, "month_data"] = list(month_data)
    datasets.loc[:, "day_ratio"] = list(day_ratio)
    datasets.loc[:, "day_data"] = list(day_data)

    datasets = datasets.dropna()
    pprint(datasets)

    # 正規化で使用する最小値と最大値を定義
    ms = MinMaxScaler(feature_range=(0.010, 1))
    mx_datasets = ms.fit_transform(datasets)

    # pprint(mx_datasets)
    # pprint(mx_datasets)
    x = mx_datasets[:, 0:2]
    y = mx_datasets[:, 2]
    pprint(y)
    test_size = 0.2  # 分割比率
    N_trials = 10  # 試行回数

    mape_dict = {reg_name:[] for reg_name in reg_dict.keys()}  # 精度の格納庫

    for i in range(N_trials):
        print(f"Trial {i+1}")
        random_state = random.randint(1, 500)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        for reg_name, reg in reg_dict.items():
            print(reg_name)
            reg.fit(x_train,y_train)
            y_pred = reg.predict(x_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)  # MAPEを算出
            mape_dict[reg_name].append(mape)  # 格納

    # MAPEの平均値でソート
    mape_dict_sorted = {key: value for key, value in reversed(sorted(mape_dict.items(), key=lambda x:np.mean(x[1])))}


    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    plt.rcParams["font.size"] = 14  # フォントサイズを大きくする
    scalarMap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=len(mape_dict)), cmap=plt.get_cmap('gist_rainbow_r'))

    plt.figure(figsize=(25,12))
    box=plt.boxplot(mape_dict_sorted.values(), vert=False, patch_artist=True,labels=mape_dict_sorted.keys())
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(scalarMap.to_rgba(i))
    plt.title("MAPE Box Plot")
    plt.xlabel("MAPE")
    plt.ylabel("Regressor Name")

    plt.savefig('result.png')



if __name__ == '__main__':
    main()