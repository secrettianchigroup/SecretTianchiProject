# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from sklearn import preprocessing
from datetime import datetime
import os
import xgboost

def extract_date(x):
    d = datetime.fromtimestamp(x)
    return d.strftime('%Y-%m-%d')


class BaseFrame:
    """
    分解训练, 测试, 验证集

    Usage
    -----------
    kfold训练, 测试, 验证
    fa = BaseFrame(train_df, test_df, 6)
    fa.df

    clf = XGBClassifier()
    fa.fit(clf, ['instance_id'], 'test_csv')

    """

    ###################
    # 直接访问成员, df为特征处理集
    df = None
    y_test = None
    ####################

    def __init__(self, offline_df, online_df, days, submit=False):

        self.offline_df = offline_df
        self.offline_df['data_set'] = 'training'
        self.online_df = online_df
        self.online_df['data_set'] = 'testing'
        self.submit = submit
        self._gen_total_df(days)

    def _gen_total_df(self, days):
        _TARGET = 'is_trade'

        # 线下线上数据统一进行特征处理
        self.online_df[_TARGET] = -1
        total_df = pd.concat([self.offline_df, self.online_df], axis=0, ignore_index=True)

        total_df['date'] = total_df['context_timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
        le = preprocessing.LabelEncoder()
        total_df['day'] = le.fit_transform(total_df['date'])
        print(total_df['day'].unique())
        # 获取训练测试的索引, 6全集数据验证, 7为生成上线文件
        starts = list(range(0, days))
        ends = [days]
        train_indices = total_df[total_df['day'].isin(starts)].index.values
        test_indices = total_df[total_df['day'].isin(ends)].index.values
        self.starts = starts
        self.ends = ends

        if not self.submit:
            # 测试
            self.y_test = total_df.iloc[test_indices]['is_trade']
            
            # 把需要训练和测试的数据提取, 并且强制去掉测试集的label列
            total_df.loc[test_indices, 'is_trade'] = np.nan

            train_tmp = total_df.iloc[train_indices].copy()
            train_tmp['data_set'] = 'training'
            test_tmp = total_df.iloc[test_indices].copy()
            test_tmp['data_set'] = 'testing'
            self.df = train_tmp.append(test_tmp)
        else:
            # 提交
            self.y_test = self.online_df[_TARGET]
            self.df = total_df

    def fit(self, clf, feat_cols, filename, drop=True, ab_rate=0.7, random_state=2, early_stopping_rounds=100):
        path = 'submits/'+filename+'.csv'
        if os.path.exists(path):
            print('model already exists!')
            return 

        train_df = self.df[self.df['data_set'] == 'training']
        test_df = self.df[self.df['data_set'] == 'testing']

        X_train = None 
        X_test = None 
        if drop:
            X_train = train_df.drop(feat_cols + ['data_set'], axis=1)
            X_test = test_df.drop(feat_cols + ['data_set'], axis=1)
            if 'date' in train_df.columns and 'date' in test_df.columns:
                X_train = X_train.drop(['date'], axis=1)
                X_test = X_test.drop(['date'], axis=1)        
        else:
            X_train = train_df[feat_cols]
            X_test = test_df[feat_cols]

        y_train = train_df[['is_trade']].values.ravel()
        y_test = self.y_test.values.ravel()
        print('X_train %s, y_train %s, X_test %s, y_test %s' % (X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        # y_test is already exists
        # 训练模型
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split

        print('>', X_train.shape, '\n')

        # 如果移动到线上集, 则输出模型
        if sum(y_test == -1) > 0:
            m = clf.fit(X_train, y_train)
            result = pd.DataFrame()
            result['instance_id'] = test_df['instance_id']
            result['predicted_score'] = pd.DataFrame(m.predict_proba(X_test))[1].values
            result.to_csv(path, sep = ' ', header=True, index = False)
        else:
            m = clf.fit(X_train, y_train, 
                eval_set=((X_train, y_train, 'train'), (X_test, y_test, 'test')), 
                eval_metric='logloss',
                early_stopping_rounds=early_stopping_rounds)
            # 分离a,b榜
            train_loss = log_loss(y_train, m.predict_proba(X_train))*100000
            test_loss  = log_loss(y_test, m.predict_proba(X_test))*100000
            print('> (%s -> %s) %.0f_%.0f %d\n' % \
                  (self.starts, self.ends, train_loss, clf.best_score*100000, m.best_ntree_limit))
        return m, X_train, X_test, y_train, y_test
          

    def fitXgb(self, param, feat_cols, filename, drop=True):
        path = 'submits/'+filename+'.csv'
        if os.path.exists(path):
            print('model already exists!')
            return 

        train_df = self.df[self.df['data_set'] == 'training']
        test_df = self.df[self.df['data_set'] == 'testing']

        X_train = None 
        X_test = None 
        if drop:
            X_train = train_df.drop(feat_cols + ['data_set'], axis=1, errors='ignore')
            X_test = test_df.drop(feat_cols + ['data_set'], axis=1, errors='ignore')
            if 'date' in train_df.columns and 'date' in test_df.columns:
                X_train = train_df.drop(feat_cols + ['date'], axis=1, errors='ignore')
                X_test = test_df.drop(feat_cols + ['date'], axis=1, errors='ignore')        
        else:
            X_train = train_df[feat_cols]
            X_test = test_df[feat_cols]

        y_train = train_df[['is_trade']].values.ravel()
        y_test = self.y_test.values.ravel()

        dtrain = xgboost.DMatrix(data=X_train.values, label=y_train)
        dtest = xgboost.DMatrix(data=X_test.values, label=y_test)

        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        progress = dict()

        bst = xgboost.train(param, dtrain, 3000, watchlist, early_stopping_rounds=150, evals_result=progress)
        bst.save_model('xgbModel_eval')
        xgb.plot_importance(bst)
        loglossList = progress['test']['logloss']
        return loglossList