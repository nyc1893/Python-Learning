

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd


import pickle
import datapick
import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# python getlabel.py 2>&1 | tee b1.log
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time  
from sklearn import metrics  
import pickle as pickle  


import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd
from pandas._testing import assert_frame_equal


def run():
    s1 = timeit.default_timer()  
    df1 = pd.read_csv("te.csv")
    df2 = pd.read_csv("../py2/te2.csv")
    print(df1.dtypes)
    # df.loc[df.Q1 == 8]
    ind1 = df1[df1["real"] == 1].index
    ind2 = df1[df1["RF"] == 1].index
    
    print(len(ind1))
    df2.real[ind1] = 2
    df2.RF[ind2] = 2
    
    y_test = df2["real"].values
    predict = df2["RF"].values
    matrix=confusion_matrix(y_test, predict)
    print(matrix)
    class_report=classification_report(y_test, predict)
    print(class_report)      
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))


    
    
def main():
    s1 = timeit.default_timer()  

    run()

    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

