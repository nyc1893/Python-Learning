

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

import datetime


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
from sklearn import metrics  
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import sys
import pandas as pd


      
def main(k):

    list = ['real',	'KNN',	'LR','RF','DT','SVM','GBDT','SNO']
    df = pd.read_csv('Ltest_34.csv')
    yreal = df[str(list[0])]
    yhat = df[str(list[k])]
    # print(df.head())
    matrix=confusion_matrix(yreal,yhat)
    
    print(matrix)
    class_report=classification_report(yreal,yhat)
    print(class_report)
    print(list[k])

    
if __name__ == '__main__':  
    k = int(sys.argv[1])
    main(k)

