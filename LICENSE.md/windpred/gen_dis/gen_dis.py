import math
import pandas as pd
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
import timeit


def get_data():
    df = pd.read_csv("data/tt_1.csv")
    for i in range(2,36+1):
        df2 =  pd.read_csv("data/tt_"+str(i)+".csv")
        df = pd.concat([df,df2],axis = 1)
    print(df.head())
    df.to_csv("data/cc.csv",index =0)
def get_real():
    path1 = "../../data/"
    df = pd.read_csv(path1+"ppge12_2010.csv")
    df2 = pd.read_csv(path1+"ppmit12_2010.csv")
    
    df["sum"] = df["l0"]+ df2["l0"]
    # print(df.head())
    return df["sum"].values
    
def sign(x):
    if(x>0):
        return 1
    else:
        return 0

def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def deal(start,end):
    df = pd.read_csv("data/cc.csv")
    df2 = get_real()
    res = 0
    for i in range(start,end):
        temp = df.iloc[i].values
        # print(temp.shape)
        # print(temp)
        
        # print(df2[:5])
        dt = df2[i]
        # print(dt)
        ecdf = sm.distributions.ECDF(temp)
        x = np.linspace(0, max(temp),1000)
        y = ecdf(x)
        # print(x)
        
        for v in range(302):
        # v = 1
            ind =  find_nearest(x,v)
            # print(ind)

            prob = y[ind]
            H = sign(v - dt)
            res += (prob - H)*(prob - H)
    res /= end - start
    # df.shape[0]
    # print(res)
    return res
    
def batch():
    ll = []
    for i in range(73):
        ll.append(deal(720*i,720*(i+1)))
    # print(ll)
    print("All events "+str(np.mean(ll)))
    
def gene_index(option):

    tr = pd.read_csv("../../data/ppmit12_2009.csv")    
    test = pd.read_csv("../../data/ppmit12_2010.csv")   
    max_y = 221

       
    
    if option == 1:
        tr = tr.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]
        test = test.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]
        valp = 7
        valn = 5
        
    elif option == 2:
        tr = tr.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]
        test = test.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]
        valp = 7
        valn = 7
        
    elif option == 3:
        tr = tr.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        test = test.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        # valp = 5
        # valn = 10
        valp = 10
        valn = 10
    elif option == 4:
        tr =     tr.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        test = test.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        valp = 7
        valn = 7
        
    elif option == 5:
        tr =    tr.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        test = test.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        valp = 10
        valn = 10
        
    elif option == 6:
        tr = tr.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        test = test.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        valp = 5
        valn = 10    

    # print(tr.iloc[:, [-5, -2]])
    tr['diff'] = tr.iloc[:, -5]-tr.iloc[:, -2]
    test['diff'] = test.iloc[:, -5]-test.iloc[:,-2]
    # test['diff'] = test['l4']-test['l1']
    



    df1 = test[(test['diff']>0.01*valp*max_y)&(test['diff']<0.15*max_y)]
    df2 = test[(test['diff']<-0.01*valn*max_y)&(test['diff']>-0.15*max_y)]
    df3 = test[(test['diff']>0.15*max_y)|(test['diff']<-0.15*max_y)|(test['diff']>-0.01*valn*max_y)&(test['diff']<0.01*valp*max_y)]
    
    # print(df1.shape)
    
    # print(df1.head())
    # print(df2.shape)
    # print(df2.tail())
    

    df = pd.concat([df1, df2])


    # print(df.head())
    # print(df.shape)
    # print(df.tail())   
    
    idx = df.index.tolist()
    
    return idx

def deal_ramp(option):
    df = pd.read_csv("data/cc.csv")
    df2 = get_real()
    res = 0
    ind = gene_index(option)
    df = df.iloc[ind]
    df2 = df2[ind]
    df= df.reset_index(drop=True)
    
    for i in range(df.shape[0]):
        temp = df.iloc[i].values
        # print(temp.shape)
        # print(temp)
        
        # print(df2[:5])
        dt = df2[i]
        # print(dt)
        ecdf = sm.distributions.ECDF(temp)
        x = np.linspace(0, max(temp),1000)
        y = ecdf(x)
        # print(x)
        
        for v in range(302):
        # v = 1
            ind =  find_nearest(x,v)
            # print(ind)

            prob = y[ind]
            H = sign(v - dt)
            res += (prob - H)*(prob - H)
    res /= df.shape[0]
    # df.shape[0]
    print("Ramp " + str(res))
    # return res

def run(option):
    get_data()
    batch()
    deal_ramp(option)
    
def main():



    s1 = timeit.default_timer()  
    
    s2 = timeit.default_timer()  
    print ('Runing time is secs:',round((s2 -s1),2))
    
""" 
"""

if __name__ == "__main__":
    main()


 