#!/usr/bin/env python
# encoding: utf-8

"""
@version: 01
@author: 
@license: Apache Licence 
@python_version: python_x86 2.7.11
@site: octowahle@github
@software: PyCharm Community Edition
@file: gen_online_image.py
@time: 2016/11/28 15:15
"""

import os
import sys
import matplotlib.pyplot as plt
import datetime
# from matplotlib.dates import YearLocator, MonthLocator, DayLocator
# from matplotlib.dates import drange, DateLocator, DateFormatter
# from matplotlib.dates import HourLocator, MinuteLocator, SecondLocator
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def get_real(turb,option):  
    if turb == 'mit':
        tr = pd.read_csv("../../data/ppmit12_2009.csv")    
        test = pd.read_csv("../../data/ppmit12_2010.csv")   
        max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv("../../data/ppge12_2009.csv")    
        test = pd.read_csv("../../data/ppge12_2010.csv")  
        max_y = 53*1.5

    y3 = test.pop('l0').values
    return max_y,y3    


def gen_image_2(l,ind1,ind2,df):
    # 格式化刻度单位
    # years=mdates.YearLocator()
    # months=mdates.MonthLocator()
    # days=mdates.DayLocator()
    hours = mdates.HourLocator()
    minutes = mdates.MinuteLocator()
    seconds = mdates.SecondLocator()

    # dateFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    # dateFmt = mdates.DateFormatter('%Y-%m-%d')
    dateFmt = mdates.DateFormatter('%H:%M')  # 显示格式化后的结果

    if len(l) != 3:
        return False

    # x = np.array(l[0])
    x = l[0]
    y1 = np.array(l[1])
    y2 = np.array(l[2])
    # std = sqrt(mean(abs(x - x.mean())**2)).
    tr = pd.read_csv("output.csv")  
    tr = np.squeeze(tr.values)
    tr = tr[ind1:ind2+1]
    print(y2.shape)
    print(tr.shape)
    val = 3.5
    y3 = y1 + val*tr
    y4 = y1 - val*tr 
    

    fig, ax = plt.subplots()
    # format the ticks
    ax.xaxis.set_major_locator(hours)  # 设置主要刻度
    # ax.xaxis.set_minor_locator(minutes)  # 设置次要刻度
    ax.xaxis.set_major_formatter(dateFmt)  # 刻度标志格式
    # ax.plot(x,y1, '', marker='.', label='Point forecast')
    ax.plot(x,y1, '-', marker='.',  color='b',label='Actual' )
   
    ax.plot(x,y2, '--', marker='.', color='r', label='Point forecast')
    
    ax.fill_between(x, y3, y4, color='grey', alpha='0.5')

    # 
    legend_elements = [Line2D([0], [0],linestyle='-',marker='.',color='b', lw=1, label='Actual'),
                   Line2D([0], [0], linestyle='--',marker='.', color='r', label='Point forecast',
                           ),
                   Patch(facecolor='gray', edgecolor='gray',
                         label='95% prediction interval')]
    
    
    
    plt.xlabel('Time of day')
    plt.ylabel('Wind Farm Generation (MW)')
    plt.grid(ls='--')
    # plt.xticks(rotation=0) 
    # fig.autofmt_xdate()  # 自动格式化显示方式
    handles=legend_elements
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.xlim(datetime.datetime(int(df[ind1][0]), int(df[ind1][1]), int(df[ind1][2]), int(df[ind1][3]), int(df[ind1][4]), 0),
    datetime.datetime(int(df[ind2][0]), int(df[ind2][1]), int(df[ind2][2]), int(df[ind2][3]), int(df[ind2][4]), 0))


    # plt.ylim(0, 46)
    plt.show()  # 显示图片
    # plt.savefig('filename.png')  # 保存图片


def main():
    pass


if __name__ == '__main__':
    # main()


    turb1 = 'mit'  
    turb2 = 'ge'

    # season  =2 

    option = 1

    _,p2 = get_real(turb1,option)
    _,p4 = get_real(turb2,option)  
    p5 = p2+p4 
    # df  = pd.DataFrame(p5)
    df2 = pd.read_csv("season-best10m.csv")
    # THis dataframe just for getting a date time
    df = pd.read_csv("G:\py_file\windpaper\code\Mits_kW_2010.csv")
  
    
    yhat = p5
    df = df.values
    df2 = df2['0'].values
    

    #possible: 2432,  5534 , 2120
    
    time_list = []
    pred_list =[]
    real_list =[]
    ii = 0
    ind1 = 5534 + ii*24
    ind2 = ind1+24
    print(ind1)
    for i in range(ind1,ind2+1):
        time_list.append(datetime.datetime(int(df[i][0]), int(df[i][1]), int(df[i][2]), int(df[i][3]), int(df[i][4]), 0))
        pred_list.append(df2[i])
        real_list.append(yhat[i])
        

    result_list = [
                    time_list,pred_list,real_list
                    ]
    gen_image_2(result_list,ind1,ind2,df)
