#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:32:48 2020

@author: imanniazazari
"""




import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers,regularizers

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

import pickle


#loading training set
pickle_in = open("hp.pickle","rb")
hp= pickle.load(pickle_in)

pickle_in = open("validation_set.pickle","rb")
X_val, y_val= pickle.load(pickle_in)

model = load_model('best_model_so_far.h5')    
#model.load_weights("best_model_so_far.hdf5")
#model.load_weights("best_model.hdf5")

y_pred_percentage=model.predict(X_val)
y_pred=model.predict_classes(X_val) 
matrix=confusion_matrix(y_val, y_pred)
print(matrix)
class_report=classification_report(y_val, y_pred)
print(class_report) 


# scores = model.evaluate(X_val, y_val, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




