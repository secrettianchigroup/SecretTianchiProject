import pandas as pd
import numpy as np 
import collections
import numpy as np


class set_review_cnt:
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
        
        whole_combines = {}
        for cate in item_category_list:
            tmp = cate+":"+"-1"
            whole_combines[tmp] = 1
            for prop in item_property_list:
                tmp = cate+":"+prop
                whole_combines[tmp] = 2
        
        
                
        predict_category_property = unique_list(line[2].split(";"))
        product = 0.
        item_vec_len = np.sqrt(len(whole_combines))
        user_vec_len = np.sqrt(len(predict_category_property))
        for item in predict_category_property:
            #x1 == 1
            #y1 == 1
            #x1*y1 == 1
            #x2 == 0
            #y2 == 1
            #x2*y2 == 0
            #所以product由x决定 += 1/0
            product += whole_combines.get(item, 0)
        
        return product/(item_vec_len*user_vec_len)
 
    print("processing predict_category_property ...")
#     dfX['predict_category_property'] = dfX['predict_category_property'].str.split(';').map(lambda x: [i.split(":")[0] for i in x]).map(filter_unless_cate)
    
    dfX['predict_richness'] =  dfX['predict_category_property'].map(lambda x: 0 if len(x.strip()) == 0 else len(x.split(";")))
    dfX['predict_category_property'] = dfX['item_category_list']+"|"+dfX['item_property_list']+"|"+dfX['predict_category_property']
    dfX['predict_category_property'] = dfX['predict_category_property'].map(inner_product_recall_items)
    
    print("processing item_property_list ...")
    dfX['item_property_richness'] = dfX['item_property_list'].map(lambda x: 0 if len(x.strip()) == 0 else len(x.split(";")))
    dfX['item_property_list'] = dfX['item_property_list'].str.split(';').map(filter_unless_prop)
    
    print("processing item_category_list ...")
    dfX['item_category_list'] = dfX['item_category_list'].str.split(';')
    
    print( "generating item_category_1, item_category_2 ...")
#     dfX['item_category_list01'] = dfX['item_category_list'].map(lambda x:x[0] if x != None and len(x) > 0 else None)
    dfX['item_category_1'] = dfX['item_category_list'].map(lambda x:x[1] if x != None and len(x) > 1 else 0)
    dfX['item_category_2'] = dfX['item_category_list'].map(lambda x:x[2] if x != None and len(x) > 2 else 0)
    
    return dfX

