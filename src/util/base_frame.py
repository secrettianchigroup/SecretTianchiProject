# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from sklearn import preprocessing
from datetime import datetime
import os

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

    def __init__(self, offline_df, online_df, days):

        self.offline_df = offline_df
        self.online_df = online_df
        self._gen_total_df(days)

    def _gen_total_df(self, days):

        # 线下线上数据统一进行特征处理
        self.online_df['is_trade'] = -1
        total_df = pd.concat([self.offline_df, self.online_df], axis=0, ignore_index=True)
        total_df['date'] = total_df['context_timestamp'].apply(lambda x: extract_date(x))
        le = preprocessing.LabelEncoder()
        total_df['day'] = le.fit_transform(total_df['date'])

        # 获取训练测试的索引, 6全集数据验证, 7为生成上线文件
        starts = list(range(0, days))
        ends = [days]
        train_indices = total_df[total_df['day'].isin(starts)].index.values
        test_indices = total_df[total_df['day'].isin(ends)].index.values
        self.starts = starts
        self.ends = ends

        # 把测试索引的label提取
        self.y_test = total_df.iloc[test_indices]['is_trade']

        # 把需要训练和测试的数据提取, 并且强制去掉测试集的label列
        total_df.loc[test_indices, 'is_trade'] = np.nan

        # 得出训练测试必须的数据集 ,并且添加一列data_set作为标记
        train_tmp = total_df.iloc[train_indices].copy()
        train_tmp['data_set'] = 'training'
        test_tmp = total_df.iloc[test_indices].copy()
        test_tmp['data_set'] = 'testing'
        self.df = train_tmp.append(test_tmp)

    def fit(self, clf, feat_cols, filename, drop=True, ab_rate=0.7, random_state=2):
        path = 'submits/'+filename+'.csv'
        if os.path.exists(path):
            print('model already exists!')
            return 

        train_df = self.df[self.df['data_set'] == 'training']
        test_df = self.df[self.df['data_set'] == 'testing']

        X_train = None 
        X_test = None 
        if drop:
            X_train = train_df.drop(feat_cols + ['data_set', 'date'], axis=1)
            X_test = test_df.drop(feat_cols + ['data_set', 'date'], axis=1)
        else:
            X_train = train_df[feat_cols]
            X_test = test_df[feat_cols]

        y_train = train_df[['is_trade']].values.ravel()

        # y_test is already exists
        # 训练模型
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split

        print('>', X_train.shape, '\n')
        m = clf.fit(X_train, y_train)

        # 如果移动到线上集, 则输出模型
        if sum(self.y_test == -1) > 0:
            result = pd.DataFrame()
            result['instance_id'] = test_df['instance_id']
            result['predicted_score'] = pd.DataFrame(m.predict_proba(X_test))[1].values
            result.to_csv(path, sep = ' ', header=True, index = False)     
        else:
            # 分离a,b榜
            X_val_a, X_val_b, y_val_a, y_val_b = train_test_split(X_test, self.y_test, test_size=0.7, shuffle=True, random_state=6)
            val_a_loss = log_loss(y_val_a, m.predict_proba(X_val_a))
            val_b_loss = log_loss(y_val_b, m.predict_proba(X_val_b))
            print('> (%s -> %s) train logloss: %.5f, test logloss: %.5f, a: %.5f, b: %.5f\n' % \
                  (self.starts, self.ends, \
                   log_loss(y_train, m.predict_proba(X_train)), \
                   log_loss(self.y_test, m.predict_proba(X_test)),\
                  val_a_loss, val_b_loss))

        return m, X_train, X_test, y_train, self.y_test
          
