# 統計処理モジュール

from datetime import datetime, timedelta
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class StatisticalProcessing:
    def __init__(self, isDebug=False):
        self.input_path = "../../../data/row/mainbranch_work.csv"
        self.holiday_path = "../../../data/row/syukujitsu.csv"
        self.data = pd.read_csv(self.input_path, index_col="date", parse_dates=True)
        self.holiday_data = pd.read_csv(self.holiday_path, index_col="date", parse_dates=True)
        self.isDebug = isDebug


    def get_rowdata(self):
        dfs = []
        counts = self.data.shape[0]
        for i, d in self.data.items():
            df = pd.DataFrame({"branch_id" : np.ones(counts) * float(i), "workload" : d})
            df.dropna(inplace=True)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        # main_branches = [float(i) for i in self.data.columns.tolist()]

        return df


    # 日毎の合計
    def day_sum(self, graph=False):
        luggage_day_total = self.data.sum(axis=1)
        if self.isDebug:
            pprint(luggage_day_total)
        if graph:
            save_path = '../img/day_sum.png'
            x = list(luggage_day_total.index)
            y = list(luggage_day_total)
            self.graph_data(x, y, save_path)
        return luggage_day_total


    # 月毎の合計
    def month_sum(self, graph=False):
        luggage_month_total = self.data.resample('M').sum().sum(axis=1)
        if self.isDebug:
            pprint(luggage_month_total)
        if graph:
            save_path = '../img/month_sum.png'
            x = list(luggage_month_total.index)
            y = list(luggage_month_total)
            self.graph_data(x, y, save_path)
        return luggage_month_total


    # 年ごとに月合計を分割
    def month_split_sum(self, year, graph=False):
        luggage_month_total = self.data.resample('M').sum().sum(axis=1)
        luggage_month_total_split = luggage_month_total[luggage_month_total.index.year == year]
        if self.isDebug:
            pprint(luggage_month_total_split)
        if graph:
            save_path = '../img/{}_month_sum.png'.format(year)
            x = list(luggage_month_total_split.index)
            y = list(luggage_month_total_split)
            self.graph_data(x, y, save_path)
        return luggage_month_total_split


    # 四半期毎の合計
    def quarter_sum(self, graph=False):
        luggage_quarter_total = self.data.resample('q').sum().sum(axis=1)
        if self.isDebug:
            pprint(luggage_quarter_total)
        if graph:
            save_path = '../img/quarter_sum.png'
            x = list(luggage_quarter_total.index)
            y = list(luggage_quarter_total)
            self.graph_data(x, y, save_path)
        return luggage_quarter_total


    # 移動平均
    def moving_average(self, window_size, graph=False):
        luggage_month_moving_average = self.data.resample('M').sum().rolling(window_size, center=True).sum().dropna().sum(axis=1)
        if self.isDebug:
            pprint(luggage_month_moving_average)
        if graph:
            save_path = '../img/moving_mean.png'
            x = list(luggage_month_moving_average.index)
            y = list(luggage_month_moving_average)
            self.graph_data(x, y, save_path)
        return luggage_month_moving_average


    # 前日比
    def day_ratio(self, graph=False):
        luggage_day_ratio = self.data.sum(axis=1).pct_change()
        if self.isDebug:
            pprint(luggage_day_ratio)
        if graph:
            save_path = '../img/day_ratio.png'
            x = list(luggage_day_ratio.index)
            y = list(luggage_day_ratio)
            self.graph_data(x, y, save_path)
        return luggage_day_ratio


    # 祝日の調査
    def holiday_search(self, eval_=False):
        holiday_flag = []
        if eval_:
            start_date = datetime(2019, 9, 1)
            forecast_days = 30
            for i in range(forecast_days):
                if (start_date + timedelta(days=i)).day in self.holiday_data.index:
                    holiday_flag.append(True)
                else:
                    holiday_flag.append(False)
        else:
            for d in list(self.data.index):
                if d in self.holiday_data.index:
                    holiday_flag.append(True)
                else:
                    holiday_flag.append(False)
        return holiday_flag


    # コストフラグ
    def cost_flag(self):
        cost_bool = (self.data.index >= datetime(2017, 10, 1))
        return cost_bool


    def day_trend(self, start, end, graph=False):
        luggage_day_trend = self.data[(self.data.index >= start) & (self.data.index < end)].sum(axis=1)
        # luggage_day_trend = self.day_sum().month
        # month_data = luggage_day_trend[luggage_day_trend.index.month == month]
        if self.isDebug:
            pprint(luggage_day_trend)
        if graph:
            save_path = '../img/day_trend.png'
            x = list(luggage_day_trend.index)
            y = list(luggage_day_trend)
            self.graph_weekend_marker(x, y, start, end, save_path)
        return luggage_day_trend


    # グラフ化
    def graph_data(self, x, y, save_path):
        fig = plt.figure(figsize=(10.0, 6.0))
        ax = fig.add_subplot(1,1,1)
        plt.xticks(fontsize=7)
        ax.plot(x, y)
        fig.savefig(save_path)


    def graph_weekend_marker(self, x, y, start, end, save_path):
        fig = plt.figure(figsize=(10.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        plt.xticks(fontsize=7)
        ax.plot(x, y)
        num = (end - start).days + 1
        for day in (start + timedelta(x) for x in range(num)):
            if day.weekday() == 5:
                plt.axvline(x=day, linewidth=4, color='lightblue')
            elif day.weekday() == 6:
                plt.axvline(x=day, linewidth=4, color='pink')
        fig.savefig(save_path)


def main():
    sp = StatisticalProcessing(isDebug=False)
    data = sp.day_sum(True)
    data = sp.month_sum(True)
    data = sp.quarter_sum(True)
    data = sp.moving_average(3, True)
    print(sp.cost_flag())


if __name__ == '__main__':
    main()