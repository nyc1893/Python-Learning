# After meeting with Dr. Hanif make changes:
#from 
#noise_var = 0.02 * np.nanmedian(dt)
#to
#noise_var = 0.02 * np.nanmedian(d_i)
# He also mentioned we can use the rocof as input


import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import math
from scipy.stats.distributions import chi2
import os

k = 1
path1 = '/scratch/share/extracted_data/8weeks_'+str(k)+'/'
path2 = 'det'+str(k)+'/'
# fname = '2016_Feb_01_0_Line Outage'
# df=pq.read_table(path1+fname).to_pandas()



    
def deal(df,fname):
    # try:
    #
    PMU_list = pd.read_csv('../../../../PMU_reporting_rate.csv')
    
    # PMU_list = pd.read_csv('/PMU_reporting_rate.csv')
    PMU_list_1 = PMU_list['60fps']

    n_PMU = PMU_list_1.size - PMU_list_1.isna().sum() - 1


    """
    parameters to adjust for event detection
    IMPORTANT NOTE: depending on the type of event and location of PMU, the parameters below are required to be updated.
    Suggestion: Fix the parameters and then create a voting mechanism among all PMUs. 

    noise_var # % of the nominal voltage. This is used in the normlaization stage of line 65
    Chi_window = 60 # size of window after a tentative event is identified. This is to confirm that event was an actual one
    Thr = 100 # Threshold of the high gain filter in the CUSUM algorithm

    """
    point_N_4 = 50


    Chi_window = 30 # size of window"""
    Chi_Thr = chi2.ppf(0.9 , Chi_window-1) # confidence level for event, you change it to 85% or 95% based on sensitivity of time series data to event"""
    Thr = 30 #Threshold of the high gain filter in CUSUM algorithm"""

    
    # print (df.head())
    # print (df.columns .tolist())
    # print (df['id'].value_counts())
    p2 = []
    t2 = []
    df['id'].value_counts().to_csv(path2+'ck.csv')
    for i_PMU in range(n_PMU):
        p = []
        t = []
        # target_PMU = 'B'+str(math.floor(PMU_list_1[i_PMU]))
        target_PMU = PMU_list_1[i_PMU]
        # target_PMU = 'B161'
        # print('target_PMU=',target_PMU)
        target_PMU = target_PMU.strip()
        # print(type(target_PMU))
        subset_df = df[(df['id'] == target_PMU)]
        n = subset_df.shape[0]
        # print('subset_df.shape',subset_df.shape)
        if n != 0:
        # if n != 0:
            start = 0
            end = n
            df2 = subset_df.reset_index()['ip_m'].iloc[start:n].interpolate(method='polynomial', order=2)
            df2 = pd.DataFrame(df2)
            df2.columns = ['a']
            # print('I am here 1')
            df2.to_csv(path2+'cc.csv',index = None)
            
            df2 =  pd.read_csv(path2+'cc.csv')
            # df = df['vp_a'].astype(str).astype(int)
            df2['a'] = df2['a'].astype("float")
            dt = np.squeeze(df2.values)
            # print(dt[0:10])
            d_i = np.gradient(dt)
            v_utc = subset_df.reset_index()['utc']
            # print(v_utc)
            filter_input = d_i # you can change the input, I suggest you use di_pm or ROCOF
            noise_var = 0.02 * np.nanmedian(d_i)
            noise_std = np.sqrt(noise_var)
            # print('I am here 2')
            # CUSUM Algorithm 
            ave = 0
            r_memo = np.zeros((n,1))
            r_N = np.zeros((n,1))
            g1_k_1 = 0 #for CUSUM
            g2_k_1 = 0 # for CUSUM """
            g1_k = 0
            g2_k = 0
            d = 0 
            TAT_counter = -1
            TAT_ind = np.zeros((n,1))
            Chi_Flag = 0
            Chi_sum = 0
            Chi_counter = 0
            # print('I am here 3')
            for kk in range(n):
                r_memo[kk] = filter_input[kk]**2 
                if (Chi_Flag == 0) and (kk > point_N_4):
                    ave  = (ave*(kk - point_N_4 - 1) + r_memo[kk]) / (kk - point_N_4) #weighted everage of residulal between two consecutive sample""" 
                    r_N[kk] = (r_memo[kk] - ave) / noise_std #normalzied residual wrt to ave and std of noise"""
                    g1_k = max(g1_k_1 + r_N[kk] - d, 0) # CUSUM filter output 1"""
                    g2_k = max(g2_k_1 - r_N[kk] - d, 0) #"CUSUM filter output 2"""
                    if (g1_k > Thr) or (g2_k > Thr): #Tentative Event Time"""
                        TAT_counter = TAT_counter + 1
                        if (abs(r_N[kk]) > Thr):
                            TAT_ind[TAT_counter] = kk - 1
                        else:
                            TAT_ind[TAT_counter] = kk
                    
                        Chi_Flag = 1
                        g1_k = 0
                        g2_k = 0
                        Chi_sum = 0
            
                g1_k_1=g1_k
                g2_k_1=g2_k
            
                if Chi_Flag == 1: # Chi-squared confiormation """
                    Chi_counter = Chi_counter + 1
                    Chi_sum = Chi_sum + r_N[kk]**2
                    if Chi_sum > Chi_Thr: 
                        # Event is detected, you can save event time for each PMU id in the created excel file
                        # print('PMU = ',PMU_list_1[i_PMU])
                        # print(PMU_list_1[i_PMU].shape)
                        p.append(target_PMU)
                        temp = v_utc[start + TAT_ind[TAT_counter]].values
                        # print('Time = ', temp )
                        # print(v_utc[start + TAT_ind[TAT_counter]].shape)
                        t.append(temp)
                        break
                    
                    if (Chi_counter > Chi_window):
                        Chi_Flag = 0;
                        Chi_sum = 0;
                        Chi_counter = 0
                        g1_k = 0
                        g2_k = 0
                        g1_k_1 = 0
                        g2_k_1 = 0
                    
                        
            if p:
                # print('I am here 4')
                p2.append(p)    
                t2.append(t)  
            else:
                print(fname + ' '+str(target_PMU)+' not detected.') 
            # print('I am here 4')
    if  p2:
        df = pd.DataFrame(p2)
        dt = pd.DataFrame(t2)
        print(df.head())
        print(dt.head())
        dt = dt.astype("str")
        # print(dt.dtypes)
        df['time'] = dt
        df['start_time'] = v_utc[start]
        df.columns = ['pmu','time','start_time']
        # print(df.head()) 
        # print('pmu shape',df.shape)   
        # df.to_csv(fname+'.csv',index =None)
        df.to_excel(path2+fname+'.xls',sheet_name='a',index =None)
        print(fname + ' save done!')
    else:
        f = open(path2+'EventDetlog.txt','a')
        f.write('\n ' + fname + ' no detected.')
        f.close()
        
            # print(fname + ' '+str('B161')+' not detected.') 
    # except ValueError or KeyError:
        # print('some errors!')
        
    # else:    

    # print('starting time:',v_utc[start])
    
def check(df):
    # print(df.count())
    df.to_csv('ck.csv',index =None)
def main():    

    fileList=os.listdir(path1)
    for f in fileList:

        x = f.split(".", 1)
        print(x[0])
        if x[0]:
            fname = x[0]
            df=pq.read_table(path1+fname+'.parquet').to_pandas()
            deal(df,fname)
        else:
            continue
        
def test():
    fname = '2016_Jan_27_1_Frequency Event'
    df=pq.read_table(path1+fname+'.parquet').to_pandas()  
    deal(df,fname)
    # check(df)
if __name__ == '__main__':  

    main()
    # test()    

    
    
    
    # nohup python -u EventDet.py > test.log 2>&1 &    
