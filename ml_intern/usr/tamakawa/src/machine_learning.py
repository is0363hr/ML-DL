from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from statistics import mean
from pprint import pprint
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random

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

# lightgbmのsklearn API
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# 統計処理モジュール
from statistical_processing import StatisticalProcessing

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
    "LGBMRegressor": LGBMRegressor(),
}

lgbm_list = {
    "LGBMRegressor": LGBMRegressor(),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100),
    "RandomForestRegressor": RandomForestRegressor(),
}

class MachineLearn:
    def __init__(self):
        pass

    def mean_absolute_percentage_error(self, y_true, y_pred):
        """MAPE"""
        del_list = np.where(y_true == 0)
        y_true = np.delete(y_true, del_list, 0)
        y_pred = np.delete(y_pred, del_list, 0)

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    def graph(self, mape_dict, mape_dict_sorted):
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


    def machine(self, x, y, reg_list, isLog=False):

        test_size = 0.2  # 分割比率
        N_trials = 10  # 試行回数

        mape_dict = {reg_name:[] for reg_name in reg_list.keys()}  # 精度の格納庫

        for i in range(N_trials):
            print(f"Trial {i+1}")
            random_state = random.randint(1, 1000)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

            for reg_name, reg in reg_list.items():
                print(reg_name)
                reg.fit(x_train,y_train)
                y_pred = reg.predict(x_test)
                mape = self.mean_absolute_percentage_error(y_test, y_pred)  # MAPEを算出
                mape_dict[reg_name].append(mape)  # 格納

        # MAPEの平均値でソート
        mape_dict_sorted = {key: value for key, value in reversed(sorted(mape_dict.items(), key=lambda x:np.mean(x[1])))}
        self.graph(mape_dict, mape_dict_sorted)

        if isLog:
            log = {}
            for k, v in mape_dict_sorted.items():
                log.update({k : mean(v)})
            pprint(log)


    def light_machine(self, x, y, reg_list, isLog=False):

        test_size = 0.3  # 分割比率
        N_trials = 10  # 試行回数

        mape_dict = {reg_name:[] for reg_name in reg_list.keys()}  # 精度の格納庫

        for i in range(N_trials):
            print(f"Trial {i+1}")
            random_state = random.randint(1, 1000)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

            for reg_name, reg in reg_list.items():
                print(reg_name)
                reg.fit(x_train,y_train)
                test_df = pd.DataFrame(index=y_test.index)
                test_df.loc[:, "workload"] = y_test
                y_pred = reg.predict(x_test)
                test_df.loc[:, "forecast"] = y_pred

                actual = test_df.resample('1D').sum().loc[:, "workload"].values
                forecast = test_df.resample('1D').sum().loc[:, "forecast"].values
                mape = np.mean(np.abs(forecast - actual) / actual * 100)
                mape_dict[reg_name].append(mape)  # 格納

        # MAPEの平均値でソート
        mape_dict_sorted = {key: value for key, value in reversed(sorted(mape_dict.items(), key=lambda x:np.mean(x[1])))}
        self.graph(mape_dict, mape_dict_sorted)

        if isLog:
            log = {}
            for k, v in mape_dict_sorted.items():
                log.update({k : mean(v)})
            pprint(log)


    def machine_csvout(self, x, y, eval_list, date_list, isLog=False):
        predict_list = pd.DataFrame()
        predict_list["date"] = date_list
        model = LGBMRegressor()
        model.fit(x, y)
        pre = model.predict(eval_list)
        predict_list.loc[:, "predict"] = pre
        predict_list.to_csv('result.csv', header=False, index=False)
        print('csv_out')


def sum_feauture():
    sp = StatisticalProcessing(isDebug=False)
    day_data = sp.day_sum()
    month_data = sp.month_sum()
    day_ratio = sp.day_ratio()
    holiday_flag = sp.holiday_search()
    cost_flag = sp.cost_flag()

    # 月単位で日毎の予測
    datasets = pd.DataFrame()
    datasets.loc[:, "year"] = day_data.index.year
    datasets.loc[:, "month"] = day_data.index.month
    datasets.loc[:, "weekday"] = day_data.index.weekday
    datasets.loc[:, "day"] = day_data.index.day
    datasets.loc[:, "holiday_flag"] = holiday_flag
    # datasets.loc[:, "cost_flag"] = cost_flag
    # 月の合計への割合
    datasets.loc[:, "day_data"] = list(day_data)

    datasets = datasets.dropna()

    # 正規化
    # ms = MinMaxScaler(feature_range=(0, 1))
    # mx_datasets = ms.fit_transform(datasets)
    # mean, scale = ms.mean_, ms.scale_

    mx_datasets = datasets.to_numpy()
    # pprint(mx_datasets)
    x = mx_datasets[:, 0:5]
    y = mx_datasets[:, 5]
    return x, y


def tutorial_feature():
    sp = StatisticalProcessing(isDebug=False)
    datasets = sp.get_rowdata()
    datasets.loc[:, "year"] = datasets.index.year
    datasets.loc[:, "month"] = datasets.index.month
    datasets.loc[:, "day"] = datasets.index.day
    datasets.loc[:, "weekday"] = datasets.index.weekday
    datasets = datasets.dropna()
    temp = datasets[datasets['workload']==0].index
    datasets.drop(temp , inplace=True)
    y = datasets.loc[:, "workload"]
    x = datasets.drop("workload", axis=1)
    return x, y


def evaluation_list():
    start_date = datetime(2019, 9, 1)
    forecast_days = 30
    evaluation_list = pd.DataFrame()
    date_list = [(start_date + timedelta(days=i)) for i in range(forecast_days)]
    years = [(start_date + timedelta(days=i)).year for i in range(forecast_days)]
    months = [(start_date + timedelta(days=i)).month for i in range(forecast_days)]
    weekdays = [(start_date + timedelta(days=i)).weekday() for i in range(forecast_days)]
    days = [(start_date + timedelta(days=i)).day for i in range(forecast_days)]
    sp = StatisticalProcessing(isDebug=False)
    holiday_flag = sp.holiday_search(eval_=True)
    evaluation_list.loc[:, "year"] = years
    evaluation_list.loc[:, "month"] = months
    evaluation_list.loc[:, "weekday"] = weekdays
    evaluation_list.loc[:, "day"] = days
    evaluation_list.loc[:, "holiday_flag"] = holiday_flag
    return evaluation_list.to_numpy(), date_list


def main():
    x, y = sum_feauture()
    # x, y = tutorial_feature()
    eva, dat = evaluation_list()
    machine = MachineLearn()
    machine.machine_csvout(x, y, eva, dat)
    # machine.machine(x, y, lgbm_list, isLog=True)



if __name__ == '__main__':
    main()|