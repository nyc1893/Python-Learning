#batch copy  https://zhidao.baidu.com/question/1641671577076787860.html

# Need to install
# python -m pip install -U xlrd

#This one will get rid of Prefix like P1_...
#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np
#p1 is the folder name tat contain all the file
model = pd.DataFrame({'degree': (np.arange(360*5)+1)*0.2})
print(model.head(20))
print(model.tail())

id_num = 1

# locals()['df'+str(i)] = pd.read_csv('data2/2019-5-8-11-0 less vehicle (Frame 2100).csv')
# locals()['df'+str(i)] = locals()['df'+str(i)][['laser_id','azimuth','distance_m']]
# print(df1.head())
# print(df1.shape)

path = "./data2/"
path2 = "./data2"
fileList=os.listdir(path2)
i = 0
for f in fileList:
    if os.path.splitext(f)[1] == '.csv':
        i = i+1
        locals()['df'+str(i)] = pd.read_csv('data2/'+str(f))
        # locals()['df'+str(i)] = pd.read_excel('data/1.xlsx')
        locals()['df'+str(i)] = locals()['df'+str(i)][['laser_id','azimuth','distance_m']]
        locals()['df'+str(i)].columns = ['id','ang','dist_file'+str(i)]
        locals()['df'+str(i)]  = locals()['df'+str(i)][locals()['df'+str(i)]['id']==id_num]
        locals()['df'+str(i)] ['degree'] = locals()['df'+str(i)].ang.apply(lambda x: 0.2*(x//20+1))

        locals()['df'+str(i)].pop('id')
        locals()['df'+str(i)].pop('ang')
        locals()['df'+str(i)] = pd.merge(locals()['df'+str(i)], model, how='outer', on='degree')
        locals()['df'+str(i)] = locals()['df'+str(i)].drop_duplicates(['degree'])
        locals()['df'+str(i)] = locals()['df'+str(i)].sort_values(by="degree" , ascending=True)
        
        locals()['df'+str(i)] = locals()['df'+str(i)][['degree','dist_file'+str(i)]]
        # print(locals()['df'+str(i)].head())
        # print(locals()['df'+str(i)].shape)
# df = pd.concat([df1, df2], ignore_index=True)
# df.columns = ['id','ang','dist']
print('Total number of file in the folder is',i)
df = df1
for ii in range(2,i+1):
    df = pd.merge(df , locals()['df'+str(ii)], how='outer', on='degree')
print(df.head())
print(df.shape)


df=df.fillna('NaN')
df.to_csv('OUTPUT-'+str(id_num)+'.csv',index = None)
print("save done for id =",id_num )

"""
"""
