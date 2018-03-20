# coding: utf-8

import numpy as np 
import pandas as pd 
# from ffm import FFMData
import pickle
import os
import numpy as np
import pandas as pd

class FeatureProcess:
    def __init__(self, target=None, categorical=None, numerical=None, listype=None):
        self.field_index = {}
        self.feature_index = {}
        self.target = target
        self.categorical = categorical
        self.numerical = numerical
        self.listype = listype

    def norm(self, df):
        #对数值列归一化

        for num in self.numerical:
            d = df[[num]].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            df = df.drop(num, axis=1)
            df[num] = d

        return df

    def fillempty(self, df, val):
        for num in self.numerical:
            df[num] = df[num].map(lambda x: val if x < 0 else x )

        return df

    def mov2pos(self, df):
        for num in self.numerical:
            if df[num].min() < 0:
                df[num] += df[num].min()

        return df

    def balance_pos_neg_sample(self, df, factor = [1,10]):
        factor = float(factor[1]) / float(factor[0])
        print factor, int(len(df.loc[df[self.target] > 0])*factor)
        df = df.loc[df[self.target] == 1].append(df.loc[df[self.target] == 0].sample(n=int(len(df.loc[df[self.target] == 1])*factor), random_state=333)).sample(frac=1,random_state=666)
        return df


    def fit(self, df):
        '''
        df: Pandas DataFrame
        target: label column, str
        categorical: categorical columns, list
        numerical: numerical columns, list
        '''
        #收集categorical的field
        feature_code = 0
        field_index = 0
        for col in self.categorical:
            self.field_index[col] = field_index
            field_index += 1

            #收集categorical的Onehot特征
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}={}'.format(col, self.fixVals(val))
                if self.feature_index.get(name, -1) == -1:
                    self.feature_index[name] = feature_code
                    feature_code += 1

        #收集listype的field
        for col in self.listype:
            self.field_index[col] = field_index
            field_index += 1

        #收集listype的Onehot特征
        for _, row in df.iterrows():
            for col in self.listype:
                arr = row[col]
                if arr == None:
                    continue
                for item in arr:
                    name = '{}={}'.format(col, self.fixVals(item))
                    if self.feature_index.get(name, -1) == -1:
                        self.feature_index[name] = feature_code
                        feature_code += 1

        for field_index, col in enumerate(self.numerical, start=len(self.categorical)):
            self.field_index[col] = field_index
            self.feature_index[col] = feature_code
            feature_code += 1

        # for k,v in self.feature_index.items():
        #     print k,v

        return self


    # def toOneHotList(self, df):

    def toOneHotList(self, df):
        print( "toOneHotList ...")
        # listype = ['item_category_list']
        
        muniq = {}
        
        total = len(df)
        
        idx = 0
        
        for _, row in df.iterrows():
            for col in self.listype:
                arr = row[col]
                if arr == None:
                    continue
                for item in arr:
                    new_col = "*ONEHOT*_"+str(col)+"="+str(item)
                    if muniq.get(new_col) == None:
                        muniq[new_col] = [0] * total
                    muniq[new_col][idx] = 1
            idx += 1
                    
                        
        muniq = pd.DataFrame(muniq)
        
    	df[list(muniq.columns)] = muniq[list(muniq.columns)]
        return df

    def toOneHot(self, df):
        print( "toOneHot ...")
        new_prefixs = []
        for c in self.categorical:
        	new_prefixs.append("*ONEHOT*_"+c)
        tmp = df[self.categorical]
        ret = pd.get_dummies(df, prefix=new_prefixs, prefix_sep="=", columns=self.categorical)
        ret[self.categorical] = tmp
        return ret
        

    def fixVals(self, val):
        if type(val) != str:
            if float(val) - int(val) < 1e-7:
                return int(val)
        else:
            return val
    def toFFMData(self, df, fpath, mod=0):
        self.fit(df)
        fp = open(fpath, "wb+")
        print("Max field index:%s"%max(self.field_index.values()))
        print("Max feature index:%s"%max(self.feature_index.values()))
        print( "toFFMData ...", fpath)
        for _, row in df.iterrows():

            fp.write(str(row[self.target]))
            fp.write(" ")

            #构建categorical的 field:feature:val组
            for cat in self.categorical:
                if pd.notnull(row[cat]):
                    feature = '{}={}'.format(cat, self.fixVals(row[cat]))
                    fp.write("%s:%s:%s " % (self.field_index[cat], self.feature_index[feature], 1))

            #构建listype的 field:feature:val组
            for col in self.listype:
                arr = row[col]
                if arr == None:
                    continue
                for item in arr:
                    feature = '{}={}'.format(col, self.fixVals(item))
                    fp.write("%s:%s:%s " % (self.field_index[col], self.feature_index[feature], 1))


            for num in self.numerical:
                val = row[num]
                if pd.notnull(row[num]):
                    if val < 0 or val > 1:
                        val = mod
                    fp.write("%s:%s:%s " % (self.field_index[num], self.feature_index[num], val))
            

            fp.write("\n")
        fp.close()
    
    def readyCache(self, func, df, path="./", subfix=".pickle"):
        import md5

        ss = str(df)+str(len(df))

        
        m1 = md5.new()   
        m1.update(ss)
        ss = m1.hexdigest()

        name =  ss + "-" + str(func.__name__) + subfix

        if os.path.exists(path+name):
            print( "From Cache ...")
            return True, name
        else:
            print( "Process ...")
            return False, name
            

    def cacheRun(self, func, df):
        
        is_cached, name = self.readyCache(func, df)

        if is_cached:
            fp = open(name, "rb+")
            return pickle.load(fp)
        else:
            fp = open(name, "wb+")
            df = func(df)
            if type(df) == pd.DataFrame:
                df.to_pickle(fp)
            else:
                pickle.dump(df, fp)
            return df

