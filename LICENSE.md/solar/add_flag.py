import math
import pandas as pd
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
# import datetime
import time
from datetime import datetime
from datetime import timedelta

def set_flag(number):
    if number > 0:
        return 1

    else:
        return 0


def extract_time (df,name):
    dt =  df[name]
    dt = dt.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df["year"] = dt.map(lambda x: x.year)
    df["month"] = dt.map(lambda x: x.month)
    df["day"] = dt.map(lambda x: x.day)
    df["hour"] = dt.map(lambda x: x.hour)    
    return df    

def gen_lag(df,name,num):
    df['p'+str(num)] =  df[name].shift(-num)
    return df
 
#get start and end time data
def data_pack(id):

    df = pd.read_csv("solar"+str(id)+".csv")  

    df['p_flag'] = df['power'].map(set_flag)

    df = extract_time (df,'time')
    df.pop('time')
    df = gen_lag(df,'power',-1)
    for i in range(0,3):
        df = gen_lag(df,'power',-23-24*i)
        df = gen_lag(df,'power',-24-24*i)
        df = gen_lag(df,'power',-25-24*i)    
        
    
    print(df.head(2))
    # d = datetime.strptime('Sep-21-09 16:34','%b-%d-%y %H:%M')

    return df
    # df = data.set_index('time')  

    
    
def main():
    df  = data_pack(1)
    
    df.to_csv("so-pred"+str(1)+".csv",index = None)
if __name__ == "__main__":
    main()
    
    
    
    
