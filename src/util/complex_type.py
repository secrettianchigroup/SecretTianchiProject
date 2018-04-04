import pandas as pd
import numpy as np 
import collections
import numpy as np


class set_review_cnt:
    """
    

    Usage:
    -----------------
    f = set_review_cnt(key1, key2, countdown_mapping)
    df['newCol'] = df.apply(f, axis=1)


    Example:
    -----------------
    cnt_user_item_review = raw_df[["user_id", "item_id", "instance_id"]].groupby(["user_id", "item_id"])['instance_id'].count().to_dict() 
    cnt_user_cate_review = raw_df[["user_id", "item_category_1", "instance_id"]].groupby(["user_id", "item_category_1"])['instance_id'].count().to_dict()  

    f1 = set_review_cnt("user_id", "item_id", cnt_user_item_review)
    f2 = set_review_cnt("user_id", "item_category_1", cnt_user_cate_review)

    tmp = raw_df.sort_values(by="context_timestamp")
    tmp["item_review_cnt"] = tmp[["user_id", "item_id"]].apply(f1, axis=1)
    tmp["cate_review_cnt"] = tmp[["user_id", "item_category_1"]].apply(f2, axis=1)
    raw_df = tmp.sort_index()
    """

    def __init__(self, key1, key2, cnt_k1_k2_review):
        self.key1 = key1
        self.key2 = key2
        self.tmp = collections.defaultdict(int)
        self.cnt_k1_k2_review = cnt_k1_k2_review
    def __call__(self, x):
        val1,val2 = x[self.key1], x[self.key2]
        vk = (val1, val2)

        cnt = self.cnt_k1_k2_review[vk]
        if cnt == 0:
            return 0
        else:
            
            k = "%s_%s" % (val1, val2)
            ret = self.tmp[k]
            self.cnt_k1_k2_review[vk] -= 1
            self.tmp[k] += 1
            return ret
        


#简化list等复杂类型的结构
#
#
#predict_category_property会计算跟item prop和cate的余弦相似度
#具体是把两个list的数据拼成cate:-1和cate:prop两种方式拼成一个字符串再跟predict_category_property的数据计算相似度
def get_icl_map(df):
    """
    item_category_list全展开
    每一层的category是全局唯一的

    Returns
    ----------------
    形成<category:colIndex>对
    {'1968056100269760729': '2',
     '2011981573061447208': '2',
     '22731265849056483': '2'}
    """
    print("get_icl_map ... ")
    dfX = df.copy()
    dfX = dfX['item_category_list'].str.split(';', expand=True)


    m = {}
    for i in dfX[0].unique():
        if i == None:
            continue
        m[i] = "1"
    
    for i in dfX[1].unique():
        if i == None:
            continue
        m[i] = "2"
    
    for i in dfX[2].unique():
        if i == None:
            continue
        m[i] = "3"
    return m

def get_ipl_map(df):
    """
    item_property_list全展开

    """
    print("get_ipl_map ... ")
    df1 = df.copy()
    dfX = df1.copy()['item_property_list'].str.split(';')
    dfX = pd.DataFrame(dfX)
    
    m = collections.defaultdict(float)
    idx = 0
    for _, row in dfX.iterrows():
        for i in row[0]:
            m[i] += 1
    
    ll = len(dfX)
    for k,v in m.items():
        m[k] = v / ll
    return m

def cos_sim(a,b):
    if type(a) == type(b) == list:
        am,bm = {},{}
        for i in a:
            am[i] = 1
        for i in b:
            bm[i] = 1
        
        a = am
        b = bm
    
    up = 0.
    down = 0.
    for k in a:
        if k in b:
            up += (a[k]*b[k])
    down = np.sqrt(len(a))*np.sqrt(len(b))
    
    return up/down

def process_complex_types(dfX, icl_map, ipl_map):
    def filter_unless_cate(arr):
        ret = []
        for i in arr:
            if i in icl_map:
                ret.append(i)
        if len(ret) == 0:
            return None
        else:
            return ret
    
    def filter_unless_prop(arr):
        ret = []
        for i in arr:
            freq = ipl_map.get(i, 0.)
            if freq > 0.05:
                ret.append(i)
            else:
                ret.append(1)
        if len(ret) == 0:
            return None
        else:
            return unique_list(ret)
    
    def unique_list(arr):
        return list(set(arr))
    
    #{cate}:-1命中则为1分
    #{cate}:{prop}命中则为2分
    #后期优化权重
    def inner_product_recall_items(line):
        line = line.split("|")
        item_category_list = unique_list(line[0].split(";"))
        item_property_list = unique_list(line[1].split(";"))
        #删掉-1
        if "-1" in item_property_list:
            del item_property_list[item_property_list.index("-1")]
        
        
        #抽出预测的cate_list和prop_list
        line[2] = line[2].split(";")
        
        pitem_category_list_prop = {}
        pitem_category_list = []
        pitem_property_list = []
        for l in line[2]:
            l = l.split(":")
            
            if l[0] != -1:
                pitem_category_list.append(l[0])
            if len(l) >= 2 and l[1] != -1:
                l[1] = l[1].split(",")
                pitem_property_list.append(l[1])
                pitem_category_list_prop[l[0]] = l[1]
            if len(l) >= 3:
                print( "FUCK?")

        
        #计算预测的cate相似度+prop相似度
        csim = cos_sim(item_category_list, pitem_category_list)
        psim = 0.
        if len(pitem_property_list) > 0:
            for i in pitem_property_list:
                psim += cos_sim(item_property_list, i)
            psim /= len(pitem_property_list)
        
        #统计category命中率
        hit_cate_rate = 0.
        hit_cate_sim = 1.
        if len(item_category_list) > 1:
            if len(item_category_list) == 2 and item_category_list[1] in pitem_category_list_prop:
                hit_cate_rate += 1
                hit_cate_sim *= (1+cos_sim(pitem_category_list_prop[item_category_list[1]], item_property_list))
            if len(item_category_list) == 3 and item_category_list[2] in pitem_category_list_prop:
                hit_cate_rate += 1
                hit_cate_sim *= (1+cos_sim(pitem_category_list_prop[item_category_list[2]], item_property_list))
            
            hit_cate_rate /= (len(item_category_list) - 1)
            
            
        
        
        predict_richness = len(set(pitem_category_list))
        item_property_richness = len(set(item_property_list))
        return [csim, psim, predict_richness, item_property_richness, hit_cate_rate, hit_cate_sim]
            
        
            
        
        
        
    print("processing predict_category_property ...")
#     dfX['predict_category_property'] = dfX['predict_category_property'].str.split(';').map(lambda x: [i.split(":")[0] for i in x]).map(filter_unless_cate)
    
    dfX['tmp'] = dfX['item_category_list']+"|"+dfX['item_property_list']+"|"+dfX['predict_category_property']
    dfX['tmp'] = dfX['tmp'].map(inner_product_recall_items)
    
    dfX['category_sim'] = dfX['tmp'].map(lambda x: x[0])
    dfX['property_sim'] = dfX['tmp'].map(lambda x: x[1])
    dfX['predict_richness'] =  dfX['tmp'].map(lambda x: x[2])
    dfX['item_property_richness'] = dfX['tmp'].map(lambda x: x[3])
    dfX['hit_cate_cnt'] = dfX['tmp'].map(lambda x: x[4])
    dfX['hit_cate_sim'] = dfX['tmp'].map(lambda x: x[5])
    dfX.drop("tmp", axis=1)
    
    print("processing item_property_list ...")
    dfX['item_property_list'] = dfX['item_property_list'].str.split(';').map(filter_unless_prop)
    
    print("processing item_category_list ...")
    dfX['item_category_list'] = dfX['item_category_list'].str.split(';')
    
    print( "generating item_category_1, item_category_2 ...")
#     dfX['item_category_list01'] = dfX['item_category_list'].map(lambda x:x[0] if x != None and len(x) > 0 else None)
    dfX['item_category_1'] = dfX['item_category_list'].map(lambda x:x[1] if x != None and len(x) > 1 else '0')
    dfX['item_category_2'] = dfX['item_category_list'].map(lambda x:x[2] if x != None and len(x) > 2 else '0')
    
    return dfX