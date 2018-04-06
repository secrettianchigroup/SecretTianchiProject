import sys
sys.path.append('..')
from datetime import datetime, timedelta
import pandas as pd
from sklearn import preprocessing
import numpy as np
import util.bayesian_smoothing as bs



def generateTradeRateByDate(tmp, cols, gap = 7, colSmoothing=None, verbose=True, glbSmoothing=10, glbMean0=0.5):
    add_count = False

    total_days = 8

    exp = "exp_d_"
    cnt = "cnt_d_"

    #此处应该处理按天为梯度的数据
    for k in cols:
        exp_k = exp+k+str(gap)
        cnt_k = cnt+k+str(gap)
        
        alpha, beta = 0, 0
        if colSmoothing and k in colSmoothing:
            alpha = colSmoothing[k][0]
            beta = colSmoothing[k][1]
            if verbose:
                print('activating smoothing: alpha %s, beta %s' % (alpha, beta))

        for day in range(0, 8):

            cal_day = max(day - 1, 0)
            set_day = day

            
            #start_d - day(不含day)用于计算，结果赋值到day上
            days1 = np.logical_and(tmp.day.values <= cal_day, tmp.day.values > (cal_day - gap))
            days2 = (tmp.day.values == set_day)
            ret = calcTVTransform(tmp, k, 'is_trade', days1, days2, smoothing=glbSmoothing, mean0=glbMean0)
             
            # 内建平滑
            tmp.loc[tmp.day.values == day, exp_k] = ret['exp'] 

            # 外部平滑
            # tmp.loc[tmp.day.values == day, exp_k] = (ret['sum']+alpha)/(ret['cnt']+alpha+beta)


            tmp.loc[tmp.day.values == day, cnt_k] = ret["cnt"]
            if verbose:
                # print(" %s trade_rate cal_day %s set to day %s" % (k, ))
                print('(%s -> %s) spawn cols by %s' % (tmp[days1]["day"].unique(), set_day, k))



def calcTVTransform(df, key, key_y, filter_src, filter_dst, smoothing = -1, mean0=None):
    """
    Examples:
    -----------
    is_trade是是否交易列, 0为无交易, 1为有交易, 对item_city_id计算交易信息
    r = calcTVTransform(total_df, 'item_city_id', 'is_trade', (total_df['day'] == 5), (total_df['day'] == 6))

    Returns:
    -----------
    r
    r['exp'] # item_city_id列的is_trade概率
    r['sum'] # item_city_id列的is_trade和
    r['cnt'] # item_city_id列的is_trade总数
    """
    if mean0 is None:
        #计算目标的平均值做平缓用
        mean0 = df.ix[filter_src, key_y].mean()
        # print("mean0:", mean0)
    
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
    if smoothing > 0:
        r['exp'] = (_sum + smoothing * mean0)/(_cnt + smoothing)
    else:
        r['exp'] = (_sum)/(_cnt)
    r['sum'] = _sum
    r['cnt'] = _cnt
    return r