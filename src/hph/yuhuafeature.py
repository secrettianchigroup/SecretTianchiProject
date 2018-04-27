# -*- coding: utf-8 -*-
import collections
m = collections.defaultdict(dict)

def get_user_t_series(x):
    m[x["user_id"]][x["instance_id"]] = x["context_timestamp"]
    return 0

def get_last_access_sec(x):
    arr = m[x["user_id"]]
    max_idx = len(arr)-1
    target_idx = -1
    for idx,v in enumerate(arr):
        if v[0] == x["instance_id"]:
            target_idx = idx
    
    if target_idx == max_idx:
        return -1
    else:
        return int((arr[target_idx][1] - arr[target_idx+1][1]))

def get_next_access_sec(x):
    arr = m[x["user_id"]]
    max_idx = len(arr)-1
    target_idx = -1
    for idx,v in enumerate(arr):
        if v[0] == x["instance_id"]:
            target_idx = idx
    
    if target_idx - 1 < 0:
        return -1
    else:
        return int((arr[target_idx-1][1] - arr[target_idx][1]))

def future_feature(train_df):
    train_df[["user_id", "instance_id", "context_timestamp"]].apply(get_user_t_series, axis=1)
    
    for k,v in m.items():
        m[k] = sorted(v.items(),key = lambda x:x[1], reverse = True)

    train_df["next_access_sec"] = train_df[["user_id", "instance_id"]].apply(get_next_access_sec, axis=1)
    train_df["last_access_sec"] = train_df[["user_id", "instance_id"]].apply(get_last_access_sec, axis=1)
    train_df["next_access_min"] = train_df["next_access_sec"]/60
    train_df["last_access_min"] = train_df["last_access_sec"]/60
    return train_df