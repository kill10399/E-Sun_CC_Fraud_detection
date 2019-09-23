import numpy as np
import pandas as pd

train_raw0 = pd.read_csv('train.csv')
N, N_f = train_raw0.shape #1521787, 23

def to_dummy(i):
    d = {'Y':1, 'y':1, 'N':0,'n':0}
    try:
        return d[i]
    except:
        return i


isObjCol = train_raw0.dtypes=='object'
isObjCol = isObjCol[isObjCol==True].index
if len(isObjCol) >0:
    train_raw0[isObjCol] = train_raw0.select_dtypes(['object']).applymap(lambda x:to_dummy(x))
train_raw = train_raw0
