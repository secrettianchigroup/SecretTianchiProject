from datetime import datetime, timedelta
import pandas as pd

start_date = '2018-09-18'


def getColUvByDate(df, start_date, time_col, col, gap, ext_cond=True):
	'''
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

