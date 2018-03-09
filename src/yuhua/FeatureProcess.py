# coding: utf-8

import numpy as np 
import pandas as pd 
# from ffm import FFMData
import pickle
import os

class FeatureProcess:
    def __init__(self, target=None, categorical=None, numerical=None):
        self.field_index = {}
        self.feature_index = {}
        self.target = target
        self.categorical = categorical
        self.numerical = numerical

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


    def fit(self, df):
        '''
        df: Pandas DataFrame
        target: label column, str
        categorical: categorical columns, list
        numerical: numerical columns, list
        '''

        feature_code = 0
        for field_index, col in enumerate(self.categorical):
            self.field_index[col] = field_index
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}={}'.format(col, val)
                if self.feature_index.get(name, -1) == -1:
                    self.feature_index[name] = feature_code
                    feature_code += 1

        for field_index, col in enumerate(self.numerical, start=len(self.categorical)):
            self.field_index[col] = field_index
            self.feature_index[col] = feature_code
            feature_code += 1

        return self


    def toOneHot(self, df):
        print( "toOneHot ...")
        df = df[self.categorical + self.numerical + [self.target]]
        return pd.get_dummies(df, columns=self.categorical)
        # datas = []
        # #算上target
        # for i in xrange(len(self.feature_index)+1):
        #     cols = []
        #     for j in xrange(len(df)):
        #         cols.append(0)
        #     datas.append(cols)
            
        # print( "toOneHot ...")
        # for i, row in tqdm(df.iterrows(), total=len(df)):
        #     for cat in self.categorical:
        #         name = '{}={}'.format(cat, row[cat])
        #         if pd.notnull(row[cat]):
        #             datas[self.feature_index[name]][i] = 1

        #     for num in self.numerical:
        #         datas[self.feature_index[num]][i] = row[num]
            
        #     datas[-1][i] = row[self.target]
        
        # data_map = {}
        # for k,i in self.feature_index.items():
        #     data_map[k] = datas[i]

        # data_map[self.target] = datas[-1]

        # return pd.DataFrame(data_map)

    def toFFMData(self, df, fpath):
        self.fit(df)
        fp = open(fpath, "wb+")
        print( "toFFMData ...")
        for _, row in df.iterrows():

            fp.write(str(row[self.target]))
            fp.write(" ")
            for cat in self.categorical:
                if pd.notnull(row[cat]):
                    feature = '{}={}'.format(cat, row[cat])
                    fp.write("%s:%s:%s " % (self.field_index[cat], self.feature_index[feature], 1))
            for num in self.numerical:
                if pd.notnull(row[num]):
                    fp.write("%s:%s:%s " % (self.field_index[num], self.feature_index[num], row[num]))
            

            fp.write("\n")
        fp.close()
        
    def cacheRun(self, func, df):
        import md5


        ss = ""
        i = 0

        len_df = len(df)
        while True:
            for col in df.columns:
                try:
                    ss += str(df[col][i]) + "_"
                except:
                    ss += "None_"
                i += 1
                if i >= len(df):
                    break
            m1 = md5.new()   
            m1.update(ss)
            ss = m1.hexdigest()
            if i >= len(df):
                break

        
        m1 = md5.new()   
        m1.update(ss)
        ss = m1.hexdigest()

        name =  ss + "-" + str(func.__name__) + ".pickle"
        fp = None
        if os.path.exists(name):
            fp = open(name, "rb+")
        if fp != None:
            print( "From Cache ...")
            return pickle.load(fp)
        else:
            print( "Process ...")
            fp = open(name, "wb+")
            df = func(df)
            if type(df) == pd.DataFrame:
                df.to_pickle(fp)
            else:
                pickle.dump(df, fp)
            return df

    # def transform_convert(self, df):
    #     X, y = self.transform(df)
    #     return FFMData(X, y)




# if __name__ == "__main__":

#     train_df = pd.read_table('/Users/yuhua/先验CTR模型线上观察/Alimama比赛/round1_ijcai_18_train_20180301.txt',sep=' ')
#     featProc = FeatureProcess(   target="is_trade", 

#                             categorical=[
#                                             'shop_id',
#                                             'item_brand_id',
#                                             'item_city_id',
#                                             'item_price_level',
#                                             'item_sales_level',
#                                             'item_collected_level',
#                                             'item_pv_level',
#                                             'user_gender_id',
#                                             'user_age_level',
#                                             'user_occupation_id',
#                                             'user_star_level',
#                                             'context_page_id',
#                                             'shop_review_num_level',
#                                             'shop_star_level'], 

#                             numerical=[     'shop_review_positive_rate',
#                                             'shop_score_service',
#                                             'shop_score_delivery',
#                                             'shop_score_description']
#                                 )

#     # print train_df[train_df['shop_review_positive_rate']<0]['shop_review_positive_rate']

#     # train_df = train_df[0:50000]
#     train_df = featProc.fillempty(train_df, -0.000000000001)
#     # print train_df[train_df['shop_review_positive_rate']<0]['shop_review_positive_rate']

#     # print "norming ..."
#     # train_df = featProc.norm(train_df)
#     # print train_df

#     # ffmData = featProc.toFFMData(train_df)
#     # pickle.dump(ffmData, open("/Users/yuhua/先验CTR模型线上观察/Alimama比赛/train-20180301.ffm.pickle", "wb+"))
#     # print "write ffm ok!"


#     df = featProc.toOneHot(train_df)
#     print type(df),"size=",df.size
#     df.to_pickle(open("p1.pickle", "wb+"), compression="gzip")

#     # print "total columns:", len(df.columns)
#     # df.to_pickle(open("/Users/yuhua/先验CTR模型线上观察/Alimama比赛/train-20180301.df.pickle", "wb+"))
#     # print "write one hot ok!"

