from datetime import datetime, timedelta
import pandas as pd
from sklearn import preprocessing
import numpy as np


def setTradeRateByDate(tmp, cols):
    add_count = False
    # window = 2

    exp = "exp_d_"
    cnt = "cnt_d_"

    #此处应该处理按天为梯度的数据
    for k in cols:
        exp_k = exp+k
        cnt_k = cnt+k

        for day in range(0, 8):

            cal_day = max(day - 1, 0)
            set_day = day

            
            #start_d - day(不含day)用于计算，结果赋值到day上
            days1 = np.logical_and(tmp.day.values <= cal_day, tmp.day.values > (cal_day - 7))
            # days1 = (tmp.day.values == cal_day)
            # days1 = np.logical_and(tmp.day.values >= 0, tmp.day.values <= 5)
            days2 = (tmp.day.values == set_day)
            print("column %s trade_rate cal_day %s set to day %s" % (k, tmp[days1]["day"].unique(), set_day))
            ret = calcTVTransform(tmp, k, 'is_trade', days1, days2, smoothing = 200, mean0 = 0.05)
                
            tmp.loc[tmp.day.values == day, exp_k] = ret["exp"]
            tmp.loc[tmp.day.values == day, cnt_k] = ret["cnt"]


def calcTVTransform(df, key, key_y, filter_src, filter_dst, smoothing = 10, mean0=None):
    """
    calcTVTransform(total_df, 'item_city_id', 'is_trade', (total_df['day'] == 5), (total_df['day'] == 6))
    """
    if mean0 is None:
        #计算目标的平均值做平缓用
        mean0 = df.ix[filter_src, key_y].mean()
        print("mean0:", mean0)
    
    #取出key的所有值
    df['_key1'] = df[key].astype('category').values.codes
    
    
    #取出用于计算的源（后面聚合掉就没有顺序可言了）
    df_key1_y = df.ix[filter_src, ['_key1', key_y]]
    
    #根据key的取值去聚合key_y的总数和总和，用户计算rate和count
    grp1 = df_key1_y.groupby(['_key1'])
    sum1 = grp1[key_y].aggregate(np.sum)
    cnt1 = grp1[key_y].aggregate(np.size)
    
    vn_sum = 'sum_' + key
    vn_cnt = 'cnt_' + key
    
    #取出dst（带序列）的所有key
    v_codes = df.ix[filter_dst, '_key1']
    
    #得到_sum,_cnt，按dst的序列
    _sum = sum1[v_codes].values
    _cnt = cnt1[v_codes].values
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    
    r = {}
    r['exp'] = (_sum + smoothing * mean0)/(_cnt + smoothing)
    # r['exp'] = (_sum)/(_cnt)
    r['cnt'] = _cnt
    return r


def getColTradeRate(df, idCol):
    rateCol = idCol + '_tr'
    pvCol = idCol + '_pv'
    try:
        del df[rateCol]
        del df[pvCol]
    except:
        pass
    a = df.groupby([idCol]).agg({'is_trade':'sum'})
    b = df.groupby([idCol]).agg({'is_trade':'size'})
    c = a.join(b, lsuffix="_sum", rsuffix="_size")
    c[rateCol] = c['is_trade_sum'] / c['is_trade_size']
    c[pvCol] = c['is_trade_size']
    return df.join(c[[rateCol, pvCol]], on=idCol)


def getColDupByDate(df, dateCol, col, gap):
    """
    生成新特征, 根据日期, 把特定列的去重状态排列出来

    Parameters
    ----------------
    df : data
    dateCol : 日期列, 'date'
    col : 指定列
    gap : 相隔多少天

    offline_df = getColDupByDate(offline_df, 'date', 'item_id', 3)
    print(offline_df.item_id_dup_g_3)

    Returns
    ----------------
    new DataFrame with new features
    """
    # 新特征列
    newCol = col + '_dup_g_' + str(gap)  
    
    if newCol in df:
        return df
    
    out = df
    le = preprocessing.LabelEncoder()
    dates = df.groupby(dateCol).count().sort_index(ascending=True)
    start_date = dates.index[0]

    F = '%Y-%m-%d'
    d0 = datetime.strptime(start_date, F)
    app = []
    for i in range(len(dates)):

        # d1 is today
        d1 = d0 + timedelta(days=i)
        
        # d2 is peekback day
        d2 = d1 - timedelta(days=gap)

        # 筛选出最近gap天的数据, 筛选出去重目标列
        days_df = df[(df[dateCol] <= d1.strftime(F)) & (df[dateCol] >= d2.strftime(F))][[dateCol, col]]

        days_df[newCol] = days_df[col].duplicated()

        new_df = days_df[days_df[dateCol] == d1.strftime(F)][newCol]

        # 把是否重复的信息附加到原列, 并生成True, False, nodata三种状态
        app.append(new_df)

    ne = pd.DataFrame(pd.concat(app))
    out = pd.concat([df, ne], axis=1)

    # 把三种状态转换成0, 1, 2
    out[newCol] = le.fit_transform(out[newCol].fillna('nodata').astype('str'))

    return out


def getColUvByDate(df, start_date, time_col, col, gap, ext_cond=True):
    '''
    时间序列分析
    获取一个dataframe按照time_col列日期的排列的每个日期的col列聚合

    Parameters
    ----------
    df:
      input data

    start_date:
      e.g.: '2018-09-18'

    time_col:
      e.g.: 'date'

    col:
      e.g.: 'item_id'

    gap:
      rollback days, e.g.: 2

    Returns
    ----------
    DataFrame with two columns: 
    time , uv
    '''
    out = {}
    F = '%Y-%m-%d'
    d0 = datetime.strptime(start_date, F)
    total_days = df.agg({time_col: lambda x:x.nunique()}).values[0]
    # print(total_days)
    for i in range(total_days):
        # d1 is today
        d1 = d0 + timedelta(days=i)
        
        # d2 is peekback day
        d2 = d1 - timedelta(days=gap)
        
        days_df = df[(df[time_col] <= d1.strftime(F)) & (df[time_col] >= d2.strftime(F)) & (ext_cond)]
        cond_today_uv = (days_df[time_col] == d1.strftime(F)) & (days_df[col].duplicated() == False)
        spec_uv = len(days_df[cond_today_uv])
        out[d1.strftime(F)] = spec_uv
        
        #print(spec_uv)
    r_df = pd.Series(out).reset_index()
    r_df.columns = ['date', 'uv']
    return r_df

