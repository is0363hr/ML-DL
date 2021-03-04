from datetime import datetime, timedelta
from pprint import pprint
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv("../../../../data/row/mainbranch_work.csv", index_col="date", parse_dates=True)
    syukujitu = pd.read_csv("../../../../data/row/syukujitsu.csv", index_col="date", parse_dates=True)
    # pprint(list(data.sum(axis=1)))
    # pprint(data.index[ (data.index >= datetime(2016,7,1)) and (data.index < datetime(2016,7,1))   ])
    holiday_flag = []
    for s in list(data.index):
        if s in syukujitu.index:
            holiday_flag.append(True)
        else:
            holiday_flag.append(False)

    # flag = data.index == syukujitu.index
    print(holiday_flag)

    # pprint(data.resample('M').sum().rolling(3, center=True).sum().dropna().sum(axis=1))
    # pprint(data.resample('M').sum())
    # luggage_day_total = []
    # for i in range(0, main_branch.shape[0]):
    #     luggage_day_total.append({
    #         'date': main_branch.index[i].strftime("%Y-%m"),
    #         'day_total': np.nansum(list(main_branch.iloc[i, :]))
    #     })
    # pprint(luggage_day_total)


if __name__ == '__main__':
    main()
