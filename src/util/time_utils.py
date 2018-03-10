from datetime import datetime, timedelta
import pandas as pd
from sklearn import preprocessing

start_date = '2018-09-18'


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

