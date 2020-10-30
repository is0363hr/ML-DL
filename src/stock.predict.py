#import vital tools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model
# from selenium import webdriver
import requests
import json

# ---------------------------------------------------------
# method
# ---------------------------------------------------------


def csvReadPandas(fileName):
    tem = ps.read_csv(fileName, header=None)
    data = np.array(tem.dropna(how='all', axis=1))
    return data


def moving_average(dataset, section):
    value = np.ones(section)/section
    tem = [[]] * dataset.shape[1]
    for i in range(0, dataset.shape[1]):
        tem[i] = np.convolve(dataset[:, i], value, mode='valid')
    data = np.array(tem)
    return data.reshape(-1,)


def svm(data, target, num):
    # 線形SVMのインスタンスを生成
    model = SVC(kernel='linear', random_state=None)

    result_train = []
    result_test = []
    for i in range(0, num):
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=None)
        # モデルの学習。fit関数で行う。
        model.fit(x_train, y_train)

        # トレーニングデータに対する精度
        pred_train = model.predict(x_train)
        accuracy_train = accuracy_score(y_train, pred_train)
        print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

        # テストデータに対する精度
        pred_test = model.predict(x_test)
        accuracy_test = accuracy_score(y_test, pred_test)
        print('テストデータに対する正解率： %.2f' % accuracy_test)

        result_train.append(accuracy_train)
        result_test.append(accuracy_test)

    return result_train, result_test


def randomForest(data, target, num):

    model=RandomForestRegressor(n_estimators=1000)

    result_test = []

    for i in range(0, num):
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.27)

        #model making and prediction
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        #make result score and get accuracy score
        testUpDown=[]
        for test in y_test:
            if test>0:
                testUpDown.append(1)
            else:
                testUpDown.append(-1)
        predUpDown=[]
        for pred in y_pred:
            if pred>0:
                predUpDown.append(1)
            else:
                predUpDown.append(-1)
        print("確率："+str(metrics.accuracy_score(testUpDown,predUpDown)))

        result_test.append(metrics.accuracy_score(testUpDown,predUpDown))

    return result_test


def sgd(data, target, num):
    model = linear_model.SGDRegressor(max_iter=1000)

    result_test = []

    for i in range(0, num):
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.27)

        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)

        #make result score and get accuracy score
        testUpDown=[]
        for test in y_test:
            if test>0:
                testUpDown.append(1)
            else:
                testUpDown.append(-1)
        predUpDown=[]
        for pred in y_pred:
            if pred>0:
                predUpDown.append(1)
            else:
                predUpDown.append(-1)
        print("確率："+str(metrics.accuracy_score(testUpDown,predUpDown)))

        result_test.append(metrics.accuracy_score(testUpDown,predUpDown))

    return result_test


def plotFig(trial, result, result2):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')

    x = np.array(range(trial))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p1, = ax.plot(x, result)
    p2, = ax.plot(x, result2)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('accuracy')
    plt.legend([p1, p2],
        ["randomForest", "SGD"],
        loc=2)
    plt.show()


# ---------------------------------------------------------
# declaration
# ---------------------------------------------------------

train=pd.read_csv("../csv/stockPriceData.csv")
train.head()

features=['1329 iシェアーズ・コア 日経225ETF',
       '1364 ｉシェアーズ JPX日経400 ETF', '1369 One ETF 日経225',
       '1385 UBS ETF ユーロ圏大型株50(E･ストックス50)', '1386 UBS ETF 欧州株(MSCIヨーロッパ)',
       '1389 UBS ETF 英国大型株100(FTSE 100)', '1390 UBS ETF MSCIアジア太平洋株(除く日本)',
       '1391 UBS ETF スイス株(MSCIスイス20/35)', '1392 UBS ETF 英国株(MSCI英国)',
       '1458 楽天 ETF-日経レバレッジ指数連動型', '1459 楽天 ETF-日経ダブルインバース指数連動型',
       '1473 One ETF トピックス', '1474 One ETF JPX日経400',
       '1475 iシェアーズ・コア TOPIX ETF', '1476 iシェアーズ・コア Ｊリート ETF',
       '1477 iシェアーズ MSCI 日本株最小分散 ETF', '1478 iシェアーズ MSCI ジャパン高配当利回り ETF',
       '1482 iシェアーズ・コア 米国債7-10年ETF(H有)', '1483 iシェアーズ JPX/S&P 設備･人材投資ETF',
       '1489 (NEXT FUNDS)日経平均高配当株50指数連動型ETF', '1493 One ETF JPX日経中小型',
       '1494 One ETF 高配当日本株', '1496 iシェアーズ 米ドル建て投資適格社債ETF(H有)',
       '1497 iシェアーズ 米ドル建ハイイールド社債ETF(H有)', '1552 国際のETF VIX短期先物指数',
       '1557 SPDR S&P500 ETF',
       '1575 ChinaAMC CSI 300 Index ETF-JDR', '1576 南方 FTSE 中国A株 50 ETF',
       '1577 (NEXT FUNDS)野村日本株高配当70連動型ETF', '1655 iシェアーズ S&P500 米国株 ETF',
       '1656 iシェアーズ･コア　米国債7-10年 ETF', '1657 iシェアーズ･コア MSCI 先進国株(除く日本)ETF',
       '1658 iシェアーズ･コア MSCI 新興国株 ETF', '1659 iシェアーズ 米国リート ETF',
       '1683 One ETF 国内金先物',
       '(株)野村総合研究所：前日比']

num = 20
isDebag = False

# ---------------------------------------------------------
# class
# ---------------------------------------------------------

# ---------------------------------------------------------
# processing
# ---------------------------------------------------------


#reading csv file (*ETF=Exhange Traded Funds)


data = train[features]
target = train["(株)野村総合研究所：翌日比"]

result = sgd(data, target, num)
result2 = sgd(data, target, num)

# ---------------------------------------------------------
# plot
# ---------------------------------------------------------

plotFig(num, result, result2)



#feature evaluation and plots
# feature_imp = pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)
# print(feature_imp)
# sns.barplot(x=feature_imp, y=feature_imp.index)
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.figure(figsize=(30,50))
# plt.show()