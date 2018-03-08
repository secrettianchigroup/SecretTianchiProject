# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class BaseFrame:
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
    
    def kflod_validation_seq(self,k,train_and_test,cut=None):
        if k <= 2:
            raise Exception("k必须>=2")
        train, valid = self.sampling()
        arr = np.array_split(train, k)
        
        s_train_lls = 0.0
        s_test_lls = 0.0

        max_iter = 0

        print "training and testing ..."
        for i in range(k):
            if cut != None and i >= cut:
                break
            train_tmp = pd.concat(arr[:i+1]).copy()
            if i+1 == k:
                print "avg logloss train = %s, test = %s\n"%(s_train_lls/max_iter, s_test_lls/max_iter)
                print "last round using validset as testset"
                test_tmp = valid.copy()
            else:
                test_tmp = arr[i+1].copy()


            print "Doing",0,i+1," train_len=%s test_len=%s" % (len(train_tmp), len(test_tmp))

            train_lls, test_lls = train_and_test(train_tmp, test_tmp)

            s_train_lls += train_lls
            s_test_lls += test_lls
            print "logloss train = %s, test = %s\n"%(train_lls, test_lls)
            max_iter += 1
        
    def split_validation(self,train_and_test):
        train, test = self.sampling()
        train_and_test(train, test)



        


# tat = BaseFrame(train_df, 0.95,'context_timestamp', True)
# ta, ts = tat.sampling()
# tat.kflod_validation(10,None)
# help(ta)
# print ta.iloc[:,0].size
# print ts.iloc[:,0].size