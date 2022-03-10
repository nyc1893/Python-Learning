
import timeit
import sys, os
import pandas as pd
import numpy as np
import timeit
import pyarrow.parquet as pq
from datetime import datetime, timedelta

def get_feature(flag,mm,day):
    dt = pd.read_csv("icc.csv").values
    # dt.sort(key= lambda x:str(x))
    # print(dt[0][0])
    
    if(flag == 0):  
        fname= "vpm"
        feature = "vp_m"
    elif(flag == 1):
        fname= "ipm"
        feature = "ip_m"
    elif(flag == 2):  
        fname= "feq"
        feature = "f"
    elif(flag == 3):  
        fname= "rof"
        feature = "df"
    for k in range(dt.shape[0]):
        pmu = dt[k][0]
        path1 = "/hndata/ic_c/theYear=2017/theMonth="+str(mm)+"/theDay="+str(day)+"/id="+pmu+"/"
        if os.path.exists(path1) ==1:
            df = get_one(feature,path1,day)
            if(df.shape[0]==108000*24):
                df= pd.DataFrame(df)
                df.columns =  [pmu]
                print(pmu)
                df.to_parquet("../2017C/"+fname+"/theMonth="+str(mm)+"/d="+str(day)+"_"+pmu+".parquet") 
                # break
                # dt.shape[0]
    # for j in range(k+1,dt.shape[0]):
        
        # pmu = dt[j][0]
        # print(pmu)
        # path1 = "../2017C/theYear=2017/theMonth=1/theDay="+str(day)+"/id="+pmu+"/"
        # if os.path.exists(path1) ==1:
            # df2 = get_one(feature,path1,day)
            # if(df2.shape[0]==108000):
                # df[pmu] =  df2
                # print(pmu)
    # print(df.head())
    # print(df.shape) 
    
                
def get_one(feature,path1,day):

    rootpath =os.listdir(path1)
    rootpath.sort(key= lambda x:str(x))
    nn = len(rootpath)
    # print("YY")
    df = pq.read_table(path1+rootpath[0]).to_pandas()
    for i in range(1,nn):
        df2 = pq.read_table(path1+rootpath[i]).to_pandas()
        df = pd.concat([df,df2])
    # df.head(100).to_csv("hh.csv")
    # df.tail(100).to_csv("tt.csv")
    
    df['time'] = pd.to_datetime(df['utc'],format = '%Y/%m/%d %H:%M:%S')
    # datetime_start = datetime.strptime("0:30", '%H:%M') 
    # Start_time_event=datetime_start-timedelta(minutes = 30)
    # End_time_event=datetime_start+timedelta(minutes = 30)

    # df = df[(df['time']>=datetime.strftime(Start_time_event, '%H:%M')) & (df['time']<datetime.strftime(End_time_event, '%H:%M'))]
    df.sort_values('time', inplace=True) 
    res = df[ feature].values
    
    return res

        
def cc():
    df = pd.read_csv("/hndata/icc.csv")
    print(df.head())
def main():
    s1 = timeit.default_timer()
    flag = int(sys.argv[1])
    mm = int(sys.argv[2])
    day = int(sys.argv[3])
    
    get_feature(flag,mm,day)

    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == "__main__":
    main()
