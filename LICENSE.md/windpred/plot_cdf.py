import math
import pandas as pd
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt



def data_pack(turb,option,season):

    
    if turb == 'mit':
        tr = pd.read_csv("../../data/ppmit12_2009.csv")    
        test = pd.read_csv("../../data/ppmit12_2010.csv")   
        max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv("../../data/ppge12_2009.csv")    
        test = pd.read_csv("../../data/ppge12_2010.csv")  
        max_y = 53*1.5

        
    if season == 2:
        ind1 = 60*144
        ind2 = 13140 + 60*144
        path = 'gene1/'+str(turb)+'-'
    elif season == 3:
        ind1 = 13140+ 60*144
        ind2 = 13140*2 + 60*144
        path = 'gene2/'+str(turb)+'-'        
    elif season == 4:    
        ind1 = 13140*2 + 60*144
        ind2 = 13140*3+ 60*144
        path = 'gene3/'+str(turb)+'-'   
    elif season == 1:
        ind1 = 13140*3+ 60*144
        ind2 = 13140*4 
        
        ind3 = 0
        ind4 = 60*144        
        
        path = 'gene4/'+str(turb)+'-'   
    
    if option == 1:
        tr = tr.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]
        test = test.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]

    elif option == 2:
        tr = tr.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]
        test = test.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]

    elif option == 3:
        tr = tr.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        test = test.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        
    elif option == 4:
        tr =     tr.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        test = test.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]

    elif option == 5:
        tr =    tr.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        test = test.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]

    elif option == 6:
        tr = tr.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        test = test.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
    
    
    if season == 1:
        tr2 = tr.iloc[ind3:ind4]
        test2 = test.iloc[ind3:ind4]    
        # print(tr2.head())
        # print(tr2.tail())
        tr = tr.iloc[ind1:ind2]
        test = test.iloc[ind1:ind2]
        tr = pd.concat([tr2,tr])
        test = pd.concat([test2,test])
        # print(tr.head())
        # print(tr.tail())
    else:    
        tr = tr.iloc[ind1:ind2]
        test = test.iloc[ind1:ind2]
    

    
    tr['diff'] = tr['l4']-tr['l1']
    test['diff'] = test['l4']-test['l1']
    
    tr1 = tr[(tr['diff']>0.05*max_y)&(tr['diff']<0.15*max_y)]
    tr2 = tr[(tr['diff']<-0.05*max_y)&(tr['diff']>-0.15*max_y)]
    tr3 = tr[(tr['diff']>0.15*max_y)|(test['diff']<-0.15*max_y)|(test['diff']>-0.05*max_y)&(test['diff']<0.05*max_y)]

    df1 = test[(test['diff']>0.05*max_y)&(test['diff']<0.15*max_y)]
    df2 = test[(test['diff']<-0.05*max_y)&(test['diff']>-0.15*max_y)]
    df3 = test[(test['diff']>0.15*max_y)|(test['diff']<-0.15*max_y)|(test['diff']>-0.05*max_y)&(test['diff']<0.05*max_y)]
    
    df1.pop('diff')
    df2.pop('diff')
    df3.pop('diff')
    
    # print(df1.shape)
    # print(df2.shape)
    # print(df3.shape)
    
    idx1 = df1.index.tolist()
    idx2 = df2.index.tolist()
    idx3 = df3.index.tolist()
    
    # print(df1.head())
    # print(df2.head())
    # print(df3.head())



    y1 = df1.pop('l0')
    y2 = df2.pop('l0')
    y3 = df3.pop('l0')
    y4 = tr3.pop('l0')
    # i = 1
    # a = str(option)+'-'
    
    df1 = df1.values.tolist()
    df2 = df2.values.tolist()
    df3 = df3.values.tolist()
    
    y1 = y1.values.tolist()
    y2 = y2.values.tolist()
    y3 = y3.values.tolist()
    y4 = y4.values.tolist()
    # y_test = y_test.values.tolist() 
    return y4,y3

def data_pack2(turb,option,season,val):

    
    if turb == 'mit':
        tr = pd.read_csv("../../data/ppmit12_2009.csv")    
        test = pd.read_csv("../../data/ppmit12_2010.csv")   
        max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv("../../data/ppge12_2009.csv")    
        test = pd.read_csv("../../data/ppge12_2010.csv")  
        max_y = 53*1.5

        
    if season == 2:
        ind1 = 60*144
        ind2 = 13140 + 60*144
        path = 'gene1/'+str(turb)+'-'
    elif season == 3:
        ind1 = 13140+ 60*144
        ind2 = 13140*2 + 60*144
        path = 'gene2/'+str(turb)+'-'        
    elif season == 4:    
        ind1 = 13140*2 + 60*144
        ind2 = 13140*3+ 60*144
        path = 'gene3/'+str(turb)+'-'   
    elif season == 1:
        ind1 = 13140*3+ 60*144
        ind2 = 13140*4 
        
        ind3 = 0
        ind4 = 60*144        
        
        path = 'gene4/'+str(turb)+'-'   
    
    if option == 1:
        tr = tr.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]
        test = test.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]

    elif option == 2:
        tr = tr.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]
        test = test.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]

    elif option == 3:
        tr = tr.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        test = test.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        
    elif option == 4:
        tr =     tr.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        test = test.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]

    elif option == 5:
        tr =    tr.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        test = test.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]

    elif option == 6:
        tr = tr.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        test = test.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
    
    
    if season == 1:
        tr2 = tr.iloc[ind3:ind4]
        test2 = test.iloc[ind3:ind4]    
        # print(tr2.head())
        # print(tr2.tail())
        tr = tr.iloc[ind1:ind2]
        test = test.iloc[ind1:ind2]
        tr = pd.concat([tr2,tr])
        test = pd.concat([test2,test])
        # print(tr.head())
        # print(tr.tail())
    else:    
        tr = tr.iloc[ind1:ind2]
        test = test.iloc[ind1:ind2]
    

    
    tr['diff'] = tr['l4']-tr['l1']
    test['diff'] = test['l4']-test['l1']
    
    tr1 = tr[(tr['diff']>0.01*val*max_y)&(tr['diff']<0.15*max_y)]
    tr2 = tr[(tr['diff']<-0.01*val*max_y)&(tr['diff']>-0.15*max_y)]
    tr3 = tr[(tr['diff']>0.15*max_y)|(test['diff']<-0.15*max_y)|(test['diff']>-0.01*val*max_y)&(test['diff']<0.01*val*max_y)]

    df1 = test[(test['diff']>0.01*val*max_y)&(test['diff']<0.15*max_y)]
    df2 = test[(test['diff']<-0.01*val*max_y)&(test['diff']>-0.15*max_y)]
    df3 = test[(test['diff']>0.15*max_y)|(test['diff']<-0.15*max_y)|(test['diff']>-0.01*val*max_y)&(test['diff']<0.01*val*max_y)]
    

    y1 = tr1.pop('l0')
    y2 = tr2.pop('l0')
    
    y3 = df1.pop('l0')
    y4 = df2.pop('l0')

    y1 = y1.values.tolist()
    y2 = y2.values.tolist()
    y3 = y3.values.tolist()
    y4 = y4.values.tolist()

    return y1,y2,y3,y4
    
def plot(turb,year):
    for i in range (1,4+1):
        s1,s2= data_pack(turb,1,i)
        if(year):
            sample = s1
            years = 2009
        else:
            sample = s2
            years = 2010
            
        sample = np.squeeze(sample)
        # print(sample.shape)
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y, linewidth = '1',label='season_'+str(i))
# x = np.linspace(min(sample), max(sample))
    

    plt.xlabel('Aggreated Wind power of '+str(turb))
    plt.ylabel('CDF')
    plt.title('CDF of Wind power in year of '+str(years))



    plt.legend(loc='upper left')
    plt.show()

def plot_ramp(turb,year):
    list = ['y1-','c2-','g+','.-','r-3','b-4']
    r1,r2,r3,r4 = data_pack2(turb,1,1,7)
    plt.figure(num=1, figsize=(4,4))
    for i in range (1,4+1):
        s1,s2= data_pack(turb,1,i)
        if(year==2009):
            sample = s1

        else:
            sample = s2

            
        sample = np.squeeze(sample)
        # print(sample.shape)
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y,str(list[i-1]), linewidth = '1',label='season_'+str(i))
        
        
        
        
    if(year==2009):
        # years = 2009
        sample = r1
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y,str(list[4]), linewidth = '0.1',label='Ramp-up')
    

        sample = r2
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y,str(list[5]), linewidth = '0.1',label='Ramp-down')
        print('year = 1')
    else:
        # years = 2010
        sample = r3
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y, str(list[4]),linewidth = '0.1',label='Ramp-up')
        

        sample = r4
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y,str(list[5]), linewidth = '0.1',label='Ramp-down')
        print('year = esle')
    
# x = np.linspace(min(sample), max(sample))
    plt.xlabel('Aggreated Wind power of Mitsubishi turbines',fontsize=11)
    plt.ylabel('CDF',fontsize=9)
    # plt.title('CDF of Wind power in year of '+str(year))
    
    plt.xlim(0, 222)
    plt.ylim(0, 1)
# +str(years)
    plt.legend(loc='lower right',prop={'size':11})
    plt.grid(ls='--')
    plt.show()

    
def main():
    turb = 'mit'

    plot_ramp(turb,2009)
""" 
"""

if __name__ == "__main__":
    main()
    
    
    
    
