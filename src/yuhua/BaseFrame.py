# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class BaseFrame:
    def __init__(self,data,rate=0.95,base_on_which_col=None,asc=None,del_sort_col=True):
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
        train, test = train_test_split( self.data, test_size=self.rate, shuffle=False)
        return train, test
    
    def kflod_validation_seq(self,k,train_and_test):
        if k <= 2:
            raise Exception("k必须>=2")
        train, test = self.sampling()
        arr = np.array_split(train, k)
        
        s = 0.0
        max_iter = 0
        for i in range(k):
            # print "Doing",0,i+1,"..."
            s += train_and_test(pd.concat(arr[:i+1]).copy(), test.copy())
            max_iter += 1
        
        return s / max_iter
    
    def split_validation(self,train_and_test):
        train, test = self.sampling()
        train_and_test(train, test)


# tat = BaseFrame(train_df, 0.95,'context_timestamp', True)
# ta, ts = tat.sampling()
# tat.kflod_validation(10,None)
# help(ta)
# print ta.iloc[:,0].size
# print ts.iloc[:,0].size