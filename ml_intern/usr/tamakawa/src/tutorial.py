from machine_learning import tutorial_feature

from pandas import read_csv, DataFrame, concat, Series
from datetime import datetime, timedelta
from lightgbm import Dataset, train, LGBMRegressor
import numpy as np
import random

from sklearn.model_selection import train_test_split
from machine_learning import MachineLearn, lgbm_list


def main():
    input_path = "../../../data/row/mainbranch_work.csv"
    main_branch = read_csv(input_path, index_col="date", parse_dates=True)
    counts = main_branch.shape[0]
    dfs = []
    for i, d in main_branch.items():
        df = DataFrame({"branch_id" : np.ones(counts) * float(i), "workload" : d})
        df.dropna(inplace=True)
        dfs.append(df)
    df = concat(dfs, axis=0)
    main_branches = [float(i) for i in main_branch.columns.tolist()]

    df.loc[:, "year"] = df.index.year
    df.loc[:, "month"] = df.index.month
    df.loc[:, "day"] = df.index.day
    df.loc[:, "weekday"] = df.index.weekday

    y = df.loc[:, "workload"]
    x = df.drop("workload", axis=1)

    machine = MachineLearn()
    machine.light_machine(x, y, lgbm_list, isLog=True)





    # test_df = DataFrame(index=y_test.index)
    # test_df.loc[:, "workload"] = y_test
    # y_pred = model.predict(x_test)
    # test_df.loc[:, "forecast"] = y_pred

    # actual = test_df.resample('1D').sum().loc[:, "workload"].values
    # forecast = test_df.resample('1D').sum().loc[:, "forecast"].values
    # mape = np.mean(np.abs(forecast - actual) / actual * 100)
    # print("MAPE =", mape)


if __name__ == '__main__':
    main()