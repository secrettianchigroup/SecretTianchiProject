# coding: utf-8
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

class DateTimeSplit2:

    def __init__(self, dataSeries, gap):
        self.D = dataSeries 
        self.gap = gap

    def get_n_splits(self, X, y, groups=None):
        return 1

    def split(self, X, y, groups=None):
        # X, y = indexable(X, y)
        train_indices = self.D[self.D['day']<self.gap].index.values
        test_indices  = self.D[self.D['day']==self.gap].index.values
        yield np.array(train_indices), np.array(test_indices)
        return


class DateTimeSplit:
    """
    分解训练, 测试

    Usage
    -----------
    按照时间序列的 kfold训练, 测试
    dtsv = DateTimeSplit(dateSeries=df['context_timestamp'], fmt="%Y-%m%-%d")
    dateSeries : 指定时间列
    gap : e.g.: 'days', 目前只支持days

    for train_i, test_i in dtsv(X, y):
        X_train, y_train = X[train_i], y[test_i]
        X_test, y_test = X[test_i], y[test_i]

        model.fit(X_train, y_train)
        logloss(y_test, model.predict_proba(X_test))

    生成以下train, test set
    ['2018-09-18']                                                                  ['2018-09-19']
    ['2018-09-18','2018-09-19']                                                     ['2018-09-20']
    ['2018-09-18','2018-09-19','2018-09-20']                                        ['2018-09-21']
    ['2018-09-18','2018-09-19','2018-09-20','2018-09-21']                           ['2018-09-22']
    ['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22','2018-09-23'] ['2018-09-24']

    """
    def __init__(self, dateSeries=None, fmt=None, gap='days'):
        if dateSeries is None or fmt is None:
            raise ValueError("dateSeries and fmt must specified")

        self.D = pd.DataFrame(dateSeries.apply(lambda x: self._fromtimestamp(x, fmt)))
        self.D.columns = ['DATE_COL']
        self.fmt = fmt
        self.n_split = len(self.D.groupby('DATE_COL').count())

    def _fromtimestamp(self, x, fmt):
        d = datetime.fromtimestamp(x)
        return d.strftime(fmt)

    def get_n_splits(self, X, y, groups=None):
        return self.n_split - 1

    def split(self, X, y, groups=None):
        F = self.fmt 
        D = self.D         

        dates = self.D.groupby('DATE_COL').count().sort_index(ascending=True)
        d0 = datetime.strptime(dates.index[0], F)
        dt = datetime.strptime(dates.index[-1], F)

        X, y = indexable(X, y)

        for i, row in dates.reset_index().iterrows():

            # train end bound
            d1 = d0 + timedelta(days=int(i))

            # test set
            d2 = d1 + timedelta(days=1)

            train_indices = D[(D['DATE_COL'] >= d0.strftime(F)) & (D['DATE_COL'] <= d1.strftime(F))].index.values
            test_indices = D[D['DATE_COL'] == d2.strftime(F)].index.values

            yield np.array(train_indices), np.array(test_indices)

            if d2 == dt:
                break