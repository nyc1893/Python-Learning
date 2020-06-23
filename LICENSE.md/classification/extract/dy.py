# -*- coding: utf-8 -*-
"""
Created on 6/18/2020 2020
This is the main content to extract the dataset from cluster.
By using 'python ./dy.py 2 26 30' as input for month, starting date, end date.
@author: Barron
"""

import multiprocessing
import pandas as pd
import numpy as np
from functools import partial
import pyarrow.parquet as pq
import glob
from datetime import datetime, timedelta
import time

            
'''Define task function'''
def Extract_event(filename,df_it,event_num):
        Event_data_dict={}
        start_time= df_it['Start']

        datetime_start = datetime.strptime(start_time, '%m/%d/%Y %H:%M') 
        Start_time_event=datetime_start-timedelta(minutes = 5)
        End_time_event=datetime_start+timedelta(minutes = 5)
        
        Event_data = []        
        print('running the process '+str(event_num))
        table2 = pq.read_table(filename)
        batch_tables=table2.to_batches(10000)
        for batch_index in range(len(batch_tables)):
        
            df=batch_tables[batch_index].to_pandas()
#            df.index.names=['sample_num','utc','vp_m','va_m','vb_m','vc_m','vp_a','va_a','vb_a','vc_a','ip_m','ia_m','ib_m','ic_m','ip_a','ia_a','ib_a','ic_a','f','df','status','id']
#            df=df.reset_index()
            sorted_df10= df.sort_values(by=['id','utc'])
            subset_df10 = sorted_df10[(sorted_df10['utc']>=datetime.strftime(Start_time_event, '%Y-%m-%d %H:%M:%S:%f')) & (sorted_df10['utc']<=datetime.strftime(End_time_event, '%Y-%m-%d %H:%M:%S:%f'))]
            Event_data.append(subset_df10)
        subset_df = pd.concat(Event_data, axis=0, ignore_index=True)
#        print(subset_df)
        if event_num in Event_data_dict:
            Event_data_dict[event_num].append(subset_df)
        else:
            Event_data_dict.update({event_num:subset_df})   

        return Event_data_dict


def main():
    multiprocessing.freeze_support()   # required to use multiprocessing
# Protected main functio
    start = time.time()
                
    user = 'ycliu'
    '''
    set up parameters required by the task
    '''
    path1 = '/pnnl1/ic_b/theYear=2016/theMonth='+str(int(sys.argv[1]))+'/theDay='

    
    
    file = '/home/'+str(user)+'/B_Training.csv'
    df_itt = pd.read_csv(file)
#    df_iter = df_iter[5:7]
    st_time= df_itt['Start'] 
    dd=[]
    for i_time in range(len(st_time)):
        d_time=datetime.strptime(st_time[i_time], '%m/%d/%Y %H:%M') 
        dd.append(d_time)
       
    
    for Days in range(int(sys.argv[2]),int(sys.argv[3])):
        path1 = r'/pnnl1/ic_b/theYear=2016/theMonth='+str(int(sys.argv[1]))+'/theDay='+str(Days)

        all_files = glob.glob(path1 + "/*.parquet")
        df_it=[]
        Event_data_list=[]
        for days in range(len(dd)):
            if (dd[days].day==Days) & (dd[days].month==int(sys.argv[1])) & (dd[days].year==2016):
                df_it.append(days)        
        if not len(df_it)==0:
            
            df_iter=df_itt[df_it[0]:df_it[len(df_it)-1]+1]
#####################################################################################################
            if len(df_iter)>10:
                ev_num=0
                for batch_number, batch_df in df_iter.groupby(np.arange(len(df_iter))//7):
                    print(batch_df)
                    Event_data_list=[]
                    results={}
                    for filename in all_files:
                        pool = multiprocessing.Pool(len(batch_df))
                        for ind in range(batch_df.shape[0]):
                            results[ind] = pool.apply_async(Extract_event, args=(filename,batch_df.iloc[ind],ind)) 
                        pool.close()
                        pool.join()
#        print(results)
                        for it in range(batch_df.shape[0]):
                            subset_data_p=results[it].get()
                            Event_data_list.append(subset_data_p)      

                    Final_event_data={}  
                    df=[]
                    for j in range(batch_df.shape[0]):
                        for i in range(len(Event_data_list)):
                            if list(Event_data_list[i]) == [j]:
                                df.append(Event_data_list[i][j])
                        fr = pd.concat(df, axis=0, ignore_index=True)
                        Final_event_data.update({j:fr})
                        df=[]
            
        
#    print(Final_event_data)
                    for i in range(len(Final_event_data)):

                        sorted_frame= Final_event_data[i].sort_values(by=['id','utc'])
                        sorted_frame.columns = sorted_frame.columns.astype(str)
                        start_time= batch_df['Start'].iloc[i]
                        datetime_start = datetime.strptime(start_time, '%m/%d/%Y %H:%M')                                                           
                        sorted_frame.to_parquet(str(datetime_start.year)+'_'+datetime_start.strftime("%b")+'_'+datetime_start.strftime("%d")+'_'+str(ev_num)+'_'+batch_df['Event'].iloc[i]+'.parquet')
                        ev_num=ev_num+1
                    print("Mutiprocessing time: {}mins\n".format((time.time()-start)/60))
                    print("Mutiprocessing time: {}secs\n".format((time.time()-start)))

#####################################################################################################               
                
            else:
                Event_data_list=[]
                results={}
#    func = partial(Extract_event, Event_data_dict)   
                for filename in all_files:
                    pool = multiprocessing.Pool(len(df_iter))
                    for ind in range(df_iter.shape[0]):
                        results[ind] = pool.apply_async(Extract_event, args=(filename,df_iter.iloc[ind],ind)) 
                    pool.close()
                    pool.join()
#        print(results)
                    for it in range(df_iter.shape[0]):
                        subset_data_p=results[it].get()
                        Event_data_list.append(subset_data_p)      

                Final_event_data={}  
                df=[]
                for j in range(df_iter.shape[0]):
                    for i in range(len(Event_data_list)):
                        if list(Event_data_list[i]) == [j]:
                            df.append(Event_data_list[i][j])
                    fr = pd.concat(df, axis=0, ignore_index=True)
                    Final_event_data.update({j:fr})
                    df=[]
            
        
#    print(Final_event_data)
                for i in range(len(Final_event_data)):

                    sorted_frame= Final_event_data[i].sort_values(by=['id','utc'])
                    sorted_frame.columns = sorted_frame.columns.astype(str)
                    start_time= df_iter['Start'].iloc[i]
                    datetime_start = datetime.strptime(start_time, '%m/%d/%Y %H:%M')                                                            
                    sorted_frame.to_parquet(str(datetime_start.year)+'_'+datetime_start.strftime("%b")+'_'+datetime_start.strftime("%d")+'_'+str(i)+'_'+df_iter['Event'].iloc[i]+'.parquet')

                print("Mutiprocessing time: {}mins\n".format((time.time()-start)/60))
                print("Mutiprocessing time: {}secs\n".format((time.time()-start)))
        
if __name__ == "__main__":
    main()
