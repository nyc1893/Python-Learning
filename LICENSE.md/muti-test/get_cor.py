# By using the heatmap to get the Correlation



import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt
# 2-input XOR inputs and expected outputs.

def heatMap(df):
    #Create Correlation df
    corr = df.corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(6, 5))
    #Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()



turb = 'mit'

tr   = pd.read_csv('pp'+str(turb)+'12_2009.csv')
test = pd.read_csv('pp'+str(turb)+'12_2010.csv')

tr = tr.iloc[:,[0,1,7,8,9,10,11,12,13,14]]
test= test.iloc[:,[0,1,7,8,9,10,11,12,13,14]]

# heatMap(df1)
heatMap(tr)
heatMap(test)


