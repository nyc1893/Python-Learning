# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:47:16 2020

@author: Yunchuan

This code extract the event information based on Event and related PMU from Eventdet Alg.
"""
##calculating running time


## importing libariries
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime
# import cv2
import timeit
def get_time(String):
    year = String.split("_")[0]
    day = String.split("_")[2]
    num = int(String.split("_")[3])

    if String.split("_")[1] == 'Jan':
        month = 1
    elif String.split("_")[1] == 'Feb':
        month = 2
    elif String.split("_")[1] == 'Mar':
        month = 3
    elif String.split("_")[1] == 'Apr':
        month = 4
    elif String.split("_")[1] == 'May':
        month = 5 
    elif String.split("_")[1] == 'Jun':
        month = 6
    elif String.split("_")[1] == 'Jul':
        month = 7  
        
    elif String.split("_")[1] == 'Aug':
        month = 8
    elif String.split("_")[1] == 'Sep':
        month = 9
    elif String.split("_")[1] == 'Oct':
        month = 10
    elif String.split("_")[1] == 'Nov':
        month = 11 
    elif String.split("_")[1] == 'Dec':
        month = 12
           
    # print(year,month,day,num)
    String = str(month) +'/'+day+'/'+ year
    return num,String

def deal_label(y_test):
    path1 = 'y_label/'
    y_test = pd.DataFrame(y_test)
    # print(y_test.head())
    y_test.columns = ['event',	'label']
    df = y_test.pop('event')
    # print(df.head())
    df = df.str.split('_')
    df2 = pd.DataFrame(df)
    y_test['year'] = df2['event'].str[0]
    y_test['month'] = df2['event'].str[1]
    y_test['day'] = df2['event'].str[2].astype(int)
    y_test['no'] = df2['event'].str[3]   

    y_test['new'] = y_test['year'].astype(str).str.cat(y_test['month'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['day'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['no'].astype(str),sep = '_')    

    y_test = y_test[['new','label']]
    
    

    df2 = pd.read_csv(path1+'trans2.csv')
    list = df2['0'].tolist()
    y_test['label'] = y_test['label'].astype("int")
    ind1 = y_test['new'].isin(list).tolist()
    # print(ind1)
    df2 = pd.read_csv(path1+'feq2.csv')
    list = df2['0'].tolist()
    ind2 = y_test['new'].isin(list).tolist()
    
    y_test.loc[ind1, 'label'] = 6
    y_test.loc[ind2, 'label'] = 7
    # y_test['label'].iloc[ind2] = 7
    
    
    y_test.pop('new')
    
    # print(y_test.loc[40:55])
    # y_test.to_csv('kankan.csv',index =None)
    return y_test.values
    
    
    
def get_class(file_name,String):    

    data = pd.read_csv(file_name)
    data['Start'] = pd.to_datetime(data['Start'])
    data = data.set_index('Start')
    num, str =get_time(String)
    dt = data[str]
    # print(dt.iloc[num,2])
    return dt.iloc[num,2]
start = timeit.default_timer()


list_pmu = [
'B126',
'B161',
'B176',
'B193',
'B232',
'B326',
'B328',
'B450',
'B457',
'B462',
'B500',
'B513',
'B623',
'B641',
'B703',
'B750',
'B780',
'B789',
'B885',
'B890',
'B904',
'B968',
'B992']
    

event_log = '../../B_T.csv'
def check():
    
    path1 = '../parasearch6/Chi_window=100/det'+str(1)+'/'
    path2 = '../../../processed/8weeks_'+str(1)+'_inter/'
    fname2 = '2016_Feb_01_1_Line Outage.xls'
    
    df = pd.read_excel(path1 + fname2)
    p_list =[]
    # print(df.head())
    for i in range(0,df.shape[0]):
        p_list.append(df['pmu'][i])    
    
    x = fname2.split(".")
    fname2 = x[0]+'_'+p_list[0]
    df2=pq.read_table(path2+fname2+'.parquet').to_pandas()


    df2.head().to_csv('haha.csv')
 
def genrate(num):
    #for 2nd
    path1 = '../parasearch7/Chi_window=2/det'+str(num)+'/'
    #for 1st
    # path1 = '../parasearch2/Chi_window=10/det'+str(num)+'/'
    #for 3rd
    # path1 = '../parasearch6/Chi_window=100/det'+str(num)+'/'
    path2 = '../../../processed/8weeks_'+str(num)+'_inter/'   
    st=[]
    labels =[] 
    ename = []

    rootpath = os.listdir(path1)
    # rootpath.sort(key= lambda x:str(x))
    path2 = 'index/'
    filename = open(path2+'a'+str(num)+'.txt','w')
    for value in rootpath:
        if ".xls" in value:
            filename.write(str(value)+'\n')
    filename.close()

    
def save_splitdata(num):
    nn= test(num)

    a = np.arange(0,nn)
    tr,val = train_test_split(a,test_size=0.2)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)    
    
    
def test(num):
    genrate(num)
    #for 2nd
    path1 = '../parasearch7/Chi_window=2/det'+str(num)+'/'
    # for 1st
    # path1 = '../parasearch2/Chi_window=10/det'+str(num)+'/'
    #for 3rd
    # path1 = '../parasearch6/Chi_window=100/det'+str(num)+'/'
    path2 = '../../../processed/8weeks_'+str(num)+'_inter/'   
    st=[]
    labels =[] 
    rootpath = []
    path3 = 'index/'
    f = open(path3+"a"+str(num)+".txt")               # 返回一个文件对象 
    line = f.readline()               # 调用文件的 readline()方法 
    while line: 
        line = f.readline() 
        rootpath.append(line.strip('\n'))
    f.close() 
    rootpath = rootpath[:-1]
    print(rootpath)
    return len(rootpath)
    # print('len(rootpath)',len(rootpath))

    
def cc(num,flag):
    path1 = '../parasearch7/Chi_window=2/det'+str(num)+'/'
    # for 1st
    # path1 = '../parasearch2/Chi_window=10/det'+str(num)+'/'
    #for 3rd
    # path1 = '../parasearch6/Chi_window=100/det'+str(num)+'/'
    path2 = '../../../processed/8weeks_'+str(num)+'_inter/'   
    st=[]
    st2=[]
    labels =[] 
    rootpath = []
    path3 = 'index/'
    f = open(path3+"a"+str(num)+".txt")               # 返回一个文件对象 
    line = f.readline()               # 调用文件的 readline()方法 
    while line: 
        line = f.readline() 
        rootpath.append(line.strip('\n'))
    f.close() 
    rootpath = rootpath[:-1]
    path3 = 'index/'
    tr=np.load(path3 +'tr_' +str(num)+'.npy')
    val=np.load(path3 +'val_' +str(num)+'.npy')        
    rootpath = np.array(rootpath)
    tr = tr.astype(int)
    val= val.astype(int)
    if(flag == 0):
        rootpath = rootpath[tr]
    elif(flag ==1):
        rootpath = rootpath[val]
    nn = rootpath.shape[0]
    planned_event=['Planned Operations','Planned Service', 'Planned Testing']
    for j in range(0,nn):
        if '.xls' in rootpath[j]:
            filename = path1 + rootpath[j]

            if os.path.exists(filename) ==1:
                # fname = '2016_Feb_01_1_Line Outage.xls'
                df = pd.read_excel(filename)
                # df['time'] = pd.DataFrame([str(line).strip('[').strip(']').strip('\'').strip('\'').strip('"') for line in df['time']])
                # df['time'] = pd.to_datetime(df['time'])
                # df['start_time'] = pd.to_datetime(df['start_time'])
                # df['dif2'] = (df['time'] - df['start_time']).dt.total_seconds()
                p_list =[]
                time_list =[]
                # print(df.head())
                for i in range(0,df.shape[0]):
                    p_list.append(df['pmu'][i])
                    # time_list.append(df['dif2'][i])
                # print(p_list)
                x = rootpath[j].split(".")

                # for i in range(0,1):
                list2 = []
                point = np.zeros(300) 
               
                for i in range(0,df.shape[0]):
                    fname2 = x[0]+'_'+p_list[i]
                    # print(fname2)
                    if os.path.exists(path2+fname2+'.parquet') ==1:
                        df2=pq.read_table(path2+fname2+'.parquet').to_pandas()
                        xx1 = df2['vp_m'].values
                        xx2 = df2['df'].values
                        xx1 = xx1[18000-3600: 18000+7200]
                        xx2 = xx2[18000-3600: 18000+7200]
                        # print(xx.shape)
                        zz= Modified_Z(xx1)
                        ind= np.argmin(zz)
                        list2.append(ind)
                        
                for ii in range (0, 89):
                    list1 = range(ii*120,(ii+1)*120)
                    if (diff(list1,list2)):
                        point[ii] +=1
                        # print(ii)
                p =0    
                ind =0
                for ii in range (0, 89):    
                    if(p<=point[ii]):
                        p = point[ii]
                        ind = ii
                        
                for i in range(0,df.shape[0]):
                    fname2 = x[0]+'_'+p_list[i]
                    # print(fname2)
                    if os.path.exists(path2+fname2+'.parquet') ==1:
                        df2=pq.read_table(path2+fname2+'.parquet').to_pandas()
                        xx1 = df2['vp_m'].values
                        xx2 = df2['df'].values
                        xx1 = xx1[18000-3600: 18000+7200]
                        xx2 = xx2[18000-3600: 18000+7200]
                        if(ind>0 and ind<90):
                            xx1 = xx1[ind*120:(ind+1)*120]
                            xx2 = xx2[ind*120:(ind+1)*120]
                            if(isenough(xx1) and isenough(xx2)):
                                st.append(xx1)
                                st2.append(xx2)
                                event_name = x[0]
                                                            # get the y label
                                if 'Line' in event_name:
                                    # temp=  get_class(event_log,event_name)
                                    # if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
                                       # labels.append(4)
                                    # else:
                                    labels.append(0)
                                            
                                elif 'XFMR' in event_name:
                                    # temp=  get_class(event_log,event_name)
                                    # if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
                                       # labels.append(5)
                                    # else:
                                    labels.append(1)
                                          
                                elif 'Frequency' in event_name: 
                                    labels.append(2)
                                elif 'Oscillation' in event_name: 
                                    labels.append(3)   
                            
    st = np.array(st)
    st2 = np.array(st)
    labels = np.array(labels)
    return st,st2,labels
 
def isenough(a):
    if (a.max() - a.min()> (5e-4)*a.max()):
        return True
    else: 
        return False
    
def datapack(num):
    X_train,X_train2,  y_train = cc(num,0)
    X_val,X_val2,  y_val = cc(num,1)
    
    
    
    # print("X_train",X_train.shape)
    # print("y_train",y_train.shape)
    print("X_val",X_val.shape)
    print("y_val",y_val.shape)    

    data = (X_train,X_train2,y_train,X_val,X_val2,y_val)
    file = open('set/2f_'+str(num), 'wb')
    pickle.dump(data, file)
    file.close()
    print("pickle save done!")  
    
def get_class(file_name,String):    

    data = pd.read_csv(file_name)
    data['Start'] = pd.to_datetime(data['Start'])
    data = data.set_index('Start')
    num, str =get_time(String)
    dt = data[str]
    # print(dt.iloc[num,2])
    return dt.iloc[num,2]
    
def Modified_Z(data):
    c = 1.4826
    median = np.median(data)
    # print(median)
    # print("median.shape",median.shape)
    dev_med = np.array(data) -median
    # print("dev_med.shape",dev_med.shape)
    mad = np.median(np.abs(dev_med))
    z_score = dev_med/(mad*mad)
    return z_score    
    
def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]
    # retB = list(set(listA).intersection(set(listB)))            
    if retA:
        # print(retA)
        return True 
    else:
        return False    
        
def load_data(num):
    path4 = 'set/'

    p1 = open(path4+'X1_1sec_df2.pickle',"rb")
    X_train= pickle.load(p1)
    X_train =pd.DataFrame(X_train)
    print(X_train.isnull().any().sum())


    
def load_data2(num):
    p1 = open('X'+str(num)+'_rof_msee.pickle',"rb")
    X_train= pickle.load(p1)
    print(X_train.shape)

    p2 = open('y'+str(num)+'_rof_msee.pickle',"rb")
    y= pickle.load(p2)
    y2 = pd.DataFrame(y)
    
    y2.columns = ['A']
    y2['A'] = pd.to_numeric(y2['A'])
    # print(y2.shape)
    # print(y2.dtypes)
    
    # y2.columns = ['A','B']
    df = y2[y2['A'] ==0]

    # print(df.shape)
    ind1 = df.index.tolist()
    p3 = X_train[ind1]
    # p4 = df.values

    
    # print(p3[0:5])
    # print(p4[0][0:5])

    return p3
    
def draw(p3):
    x = range(p3[0].shape[0])

    plt.figure( figsize=(3*10,10))
    for j in range(0,10):
        for i in range(1+10*j,11+10*j):
            plt.subplot(10, 10, i)
            plt.plot(x,p3[i])
       
        
    plt.tight_layout()
    plt.savefig('f1')     
    


def main(jj):
    s1 = timeit.default_timer()  
    list = ['df','ip_m','ip_a','vp_a','vp_m']

   
    # for m in range(1,7):
        # save_splitdata(m)
        
    # cc(jj,0,list[4])
    for kk in range(1,7):
        datapack(kk)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
    
if __name__ == '__main__':  

    jj = int(sys.argv[1])
    main(jj)





   


    
