# coding: utf-8

import numpy as np 
import pandas as pd 
# from ffm import FFMData
import pickle
import os
import numpy as np
import pandas as pd
import copy
import collections


class filter_on_cols:
    def __init__(self, target, categorical, numerical, listype):
        self.target = target
        self.categorical = categorical
        self.numerical = numerical
        self.listype = listype
        

    def fit(self,df):
        self.df_col = copy.deepcopy(df.columns)
    
    def __getitem__(self,i):
        if i == "target":
            return self.get_raw_target_col()
        elif i == "categorical":
            return self.get_raw_categorical_cols()
        elif i == "numerical":
            return self.get_raw_numerical_cols()
        elif i == "listype":
            return self.get_raw_listype_cols()
        elif i == "onehot_categorical":
            return self.get_onehoted_cols("categorical")
        elif i == "onehot_listype":
            return self.get_onehoted_cols("listype")
        else:
            print( "FUCK you")

    def get_raw_target_col(self):
        return copy.deepcopy(self.target)
    def get_raw_categorical_cols(self):
        return copy.deepcopy(self.categorical)
    
    def get_raw_numerical_cols(self):
        return copy.deepcopy(self.numerical)
    def get_raw_listype_cols(self):
        return copy.deepcopy(self.listype)
    
    def get_onehoted_cols(self, t):
        if t == "listype":
            org_cols = self.listype
        
        if t == "categorical":
            org_cols = self.categorical
            
        ret = []
        for org_col in org_cols:
            for cur_col in list(self.df_col):
                if cur_col.find("*ONEHOT*_") != -1 and cur_col.find(org_col) != -1:
                    ret.append(cur_col)
        
        return ret


class FeatureProcess:
    def __init__(self, target=None, categorical=None, numerical=None, listype=None):
        self.field_index = {}
        self.feature_index = {}
        self.ff_index = {}
        self.rff_index = collections.defaultdict(set)
        self.target = target
        self.categorical = categorical
        self.numerical = numerical
        self.listype = listype
        self.layering_org_keys = {}

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
                df[num] -= df[num].min()

        return df

    def mov2mean(self, df):
        for num in self.numerical:
            if df[num].min() < 0:
                df.loc[df[num] < 0, num] = df[num].mean()

        return df

    def balance_pos_neg_sample(self, df, factor = [1,10]):
        factor = float(factor[1]) / float(factor[0])
        print(factor, int(len(df.loc[df[self.target] > 0])*factor))
        df = df.loc[df[self.target] == 1].append(df.loc[df[self.target] == 0].sample(n=int(len(df.loc[df[self.target] == 1])*factor), random_state=333)).sample(frac=1,random_state=666)
        return df

    def addLayeringOrgKeys(self, layering_org_keys):
        self.layering_org_keys[layering_org_keys] = 1

    def matchLayeringKeys(self, k):
        orgk = k.split("LAY*_")
        if len(orgk) != 2:
            return False
        orgk = orgk[1]
        if orgk in self.layering_org_keys:
            return orgk


    def fit(self, df):
        '''
        df: Pandas DataFrame
        target: label column, str
        categorical: categorical columns, list
        numerical: numerical columns, list
        '''
        #收集categorical的field

        print("fiting ... ")
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

                    self.ff_index[feature_code] = self.field_index[col]
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
                    if str(item).strip() == "-1":
                        continue
                    name = '{}={}'.format(col, self.fixVals(item))
                    if self.feature_index.get(name, -1) == -1:
                        self.feature_index[name] = feature_code

                        self.ff_index[feature_code] = self.field_index[col]
                        feature_code += 1

        for col in self.numerical:
            fd_col = col
            fea_col = col
            orgkey = self.matchLayeringKeys(col)
            

            if orgkey:
                if orgkey not in self.field_index:
                    self.field_index[orgkey] = field_index
                    field_index += 1
                
                self.field_index[fd_col] = self.field_index[orgkey]

            else:
                self.field_index[fd_col] = field_index
                field_index += 1

            self.feature_index[fea_col] = feature_code

            self.ff_index[feature_code] = self.field_index[fd_col]

            feature_code += 1


        for k,v in self.ff_index.items():
            self.rff_index[v].add(k)
        
        print( "fields")
        for k,v in self.field_index.items():
            print( k,v,len(self.rff_index[v]))

        print( "\nfeatures")
        for k,v in self.feature_index.items():
            print( k,v,self.ff_index[v])
        
        print("Total fields = %s" % len(self.field_index))
        print("Total features = %s" % len(self.feature_index))
        # print( self.ff_index)

        return self


    # def toOneHotList(self, df):

    def toOneHotList(self, df):

        print( "toOneHotList ...")
        # listype = ['item_category_list']

        if len(self.listype) == 0:
            print( "Nothing at all")
            return df
        
        muniq = {}
        
        total = len(df)
        
        idx = 0
        
        for _, row in df.iterrows():
            for col in self.listype:
                arr = row[col]
                if arr == None:
                    continue
                for item in arr:
                    if str(item).strip() == "-1":
                        continue
                    new_col = "*ONEHOT*_"+str(col)+"="+str(item)
                    if muniq.get(new_col) == None:
                        muniq[new_col] = [np.nan] * total
                    muniq[new_col][idx] = 1
            idx += 1
                    
                        
        muniq = pd.DataFrame(muniq)
        
        df[list(muniq.columns)] = muniq[list(muniq.columns)]
        df = df.to_sparse()
        return df

    def toOneHot(self, df):
        print( "toOneHot ...")
        new_prefixs = []
        for c in self.categorical:
            new_prefixs.append("*ONEHOT*_"+c)
        tmp = df[self.categorical]
        ret = pd.get_dummies(df, prefix=new_prefixs, prefix_sep="=", columns=self.categorical).replace(0,np.nan)
        ret[self.categorical] = tmp
        ret = ret.to_sparse()
        return ret
        
    def copyFFFIndexs(self):
        return copy.deepcopy(self.ff_index), copy.deepcopy(self.field_index), copy.deepcopy(self.feature_index)


    def fixVals(self, val):
        if type(val) != str:
            if float(val) - int(val) < 1e-7:
                return int(val)
        else:
            return val
    def toFFMData(self, df, fpath, mod=0):
        #绝对不能在此处fit，fit的时候必须用全集数据
        # self.fit(df)
        fp = open(fpath, "wb+")
        print("Max field index:%s"%max(self.field_index.values()))
        print("Max feature index:%s"%max(self.feature_index.values()))
        print("NEW! toFFMData ... %s" % fpath)
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

                m = {}
                for item in arr:
                    if str(item).strip() == "-1":
                        continue
                    feature = '{}={}'.format(col, self.fixVals(item))
                    if feature in m:
                        continue
                    fp.write("%s:%s:%s " % (self.field_index[col], self.feature_index[feature], 1))
                    m[feature] = 1


            for num in self.numerical:
                val = row[num]
                if pd.notnull(row[num]):
                    if val <= 0 or val > 1:
                        continue
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

