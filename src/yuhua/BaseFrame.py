# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split

class BaseFrame:
    """
    分解训练, 测试, 验证集

    Usage
    -----------
    kfold训练, 测试, 验证
    frame = BaseFrame(df, 0.05)
    frame.kflod_validation_seq(10, train_test_callback, None, True)

    划分训练测试集
    frame.split_validation(train_test_callback)

    根据日期划分训练测试集, 并且验证kfold过程与验证集是否同增同减
    frame.kfold_by_date('date', train_test_callback)

    """
    def __init__(self,data,rate=0.05,base_on_which_col=None,asc=None,del_sort_col=True):
        self.base_on_which_col = base_on_which_col
        self.asc = asc
        self.data = pd.DataFrame(data)
        self.rate = rate
        self.del_sort_col = del_sort_col
    
    def sampling(self):
        if self.base_on_which_col != None and type(self.asc) == bool :
            self.data = self.data.sort_values(by=self.base_on_which_col,ascending=self.asc)
            if self.del_sort_col:
                del self.data[self.base_on_which_col]
        train, valid = train_test_split( self.data, test_size=self.rate, shuffle=False)

        
        return train, valid
    
    def kfold_by_date(self, dateCol, callback):
        """
        根据按日分区的列, 进行kfold划分
        Parameters
        ------------
        dateCol : 日期列, 必须是 2018-09-18 de的格式
        callback : 验证测试回调, 接收三个参数, 返回三元组, e.g.:
            def callback(train_df, test_df, valid_df):
                return 1, 2, 3

        Returns
        ------------
        print result
        """
        df = self.data
        dates = df.groupby(dateCol).count().sort_index(ascending=True)
        start_date = dates.index[0]
        last_date = dates.index[-1]
        F = '%Y-%m-%d'
        d0 = datetime.strptime(start_date, F)
        dt = datetime.strptime(last_date, F)
        valid_df = df[(df[dateCol] == dt.strftime(F))].copy()
        del valid_df[dateCol]

        it = 0
        pre_train_ll , pre_test_ll, pre_valid_ll = 0.0, 0.0, 0.0
        for i, row in dates.iterrows():

            d1 = d0 + timedelta(days=it)
            d2 = d1 + timedelta(days=1)
            if d2 not in dates.index and d2 == dt:
                continue

            print("%s to %s is training set, %s is test set, %s is valid set, start training ... " % \
                (d0.strftime(F), d1.strftime(F), d2.strftime(F), dt.strftime(F)))

            train_df = df[(df[dateCol] >= d0.strftime(F)) & (df[dateCol] <= d1.strftime(F))].copy()
            test_df = df[(df[dateCol] == d2.strftime(F))].copy()
            del train_df[dateCol]
            del test_df[dateCol]

            train_ll, test_ll, valid_ll = callback(train_df, test_df, valid_df)

            print('%s done, logloss train = %s (compare to previous: %s), \n\
                    \ttest = %s (compare to previous: %s), \n\
                    \tvalid = %s (compare to previous: %s)\n' % \
                (it, train_ll, (train_ll - pre_train_ll), test_ll, (test_ll - pre_test_ll), 
                    valid_ll, (valid_ll - pre_valid_ll)))
            pre_train_ll, pre_test_ll, pre_valid_ll = train_ll, test_ll, valid_ll
            it += 1
        

    def kflod_validation_seq(self,k,train_and_test,cut=None):
        if k <= 2:
            raise Exception("k必须>=2")

        train, valid = self.sampling()
        arr = np.array_split(train, k)
        
        s_train_lls = 0.0
        s_test_lls = 0.0

        max_iter = 0

        print( "training and testing ...")
        for i in range(k):
            if cut != None and i >= cut:
                break
            train_tmp = pd.concat(arr[:i+1]).copy()
            if i+1 == k:
                print("avg logloss train = %s, test = %s\n"%(s_train_lls/max_iter, s_test_lls/max_iter))
                print("last round using validset as testset")
                test_tmp = valid.copy()
            else:
                test_tmp = arr[i+1].copy()


            print( "Doing",0,i+1," train_len=%s test_len=%s" % (len(train_tmp), len(test_tmp)))

            train_lls, test_lls = train_and_test(train_tmp, test_tmp)

            s_train_lls += train_lls
            s_test_lls += test_lls
            print("logloss train = %s, test = %s\n"%(train_lls, test_lls))
            max_iter += 1
        
    def split_validation(self,train_and_test):
        train, test = self.sampling()
        train_and_test(train, test)



        


# tat = BaseFrame(train_df, 0.95,'context_timestamp', True)
# ta, ts = tat.sampling()
# tat.kflod_validation(10,None)
# help(ta)
# print( ta.iloc[:,0].size)
# print( ts.iloc[:,0].size)