import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

train_raw0 = pd.read_csv('train.csv')

def to_dummy(i):
    d = {'Y':1, 'y':1, 'N':0,'n':0}
    try:
        return d[i]
    except:
        return i

def trans_obj(df):
    isObjCol = df.dtypes=='object'
    isObjCol = isObjCol[isObjCol==True].index
    if len(isObjCol) >0:
        df[isObjCol] = df.select_dtypes(['object']).applymap(lambda x:to_dummy(x))
    return df

train_raw = trans_obj(train_raw0)
# set some paramater to used for every notebook

N, N_f = train_raw.shape #1521787, 23

# fraud / non_fraud ind
fraud = train_raw.fraud_ind == 1
non_fraud = train_raw.fraud_ind == 0

# the txn_amt transfer col
para_f =plt.hist(train_raw['conam'][fraud],bins=50)
para_nf =plt.hist(train_raw['conam'][non_fraud].sample(int(N/74)), bins=50)
plt.close()
ratio = list(map(lambda x: 0 if x == np.Inf else x,(para_f[0]/para_nf[0])))+[0]
txn_amt_cvt = np.array([para_f[1], np.array(ratio)])

stocn_ratio= train_raw.groupby(['stocn']).agg({'fraud_ind':['count','mean']})
stocn_ratio.columns=['fraud_cnt', 'fraud_mean']

def toOneHot(df, col_name):
    if not col_name:
        raise 'select the col u want to do one_hot encoding'
    OHE = OneHotEncoder(handle_unknown='ignore')
    OHE.fit(df[[col_name]])
    stscd_OHE = pd.DataFrame(OHE.transform(df[[col_name]]).toarray(),\
        columns = ['%s_%s'%(col_name, i) for i in range(len(df[col_name].unique()))],\
        index = df.index)
    return stscd_OHE

# transfer amt to ratio
def txn_amt_prob(df, col_name= 'conam'):
    def cvt_txn(amt):
        if amt >txn_amt_cvt[0,-1] or amt <txn_amt_cvt[0,0]:
            return 0

        start = 0
        end = txn_amt_cvt.shape[1]-1
        flg = True
        while flg:
            mid = int(round((start + end)/2,0))
            if amt < txn_amt_cvt[0,mid]:
                end = mid
            else:
                start = mid

            while abs(start-end) == 1:
                flg= False
                return txn_amt_cvt[1,start]
    amt_cvt = df[col_name].apply(lambda x:cvt_txn(x))
    amt_cvt.name = col_name + '_cvt'
    return amt_cvt

# transfer datetime -> 7 x 24
def to_timeseg(df, col_name = ['locdt','loctm']):
    week_day = df[col_name[0]]%7
    time_hh = round(df[col_name[1]]/100000).astype(int)
    week_day.name = col_name[0] + '_cvt'
    time_hh.name = col_name[1] + '_cvt'
    return week_day.rename({'col_name':'col_name'+"_cvt"}), time_hh

def stocn_cvt(df, col_name = 'stocn'):
    a = df[[col_name]].join(stocn_ratio['fraud_mean'], how='left', on = 'stocn')
    a.name = col_name[1] + '_cvt'
    return a
