#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN to train and test with separate PMU 
for example X_train =(281, 23, 36000, 1) 
after separate PMU : X_train =(281 x 23 = 6463, 36000, 1)
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers,regularizers

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import timeit
import pickle

start = timeit.default_timer()

    


def removePlanned(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X_train[i,:,:,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X_train[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X_train[i,:,:,:])
        
        elif y_train[i]==3:
            y_new.append(3)
            X_new.append(X_train[i,:,:,:])
        

    return  np.array(X_new), np.array(y_new)


def separatePMUs(X,y):
    
    """
    This function separates features and their corresponding labels for each PMU
    to make more events 
    """
    
    num_case=X.shape[0]
    num_pmu=X.shape[1]
    num_sample=X.shape[2]
    X=X.reshape(num_case*num_pmu,num_sample)
    y_new=[]
    for i in range(len(y)):
        if y[i]==0:
            for j in range(num_pmu):
                y_new.append(0)
                
        if y[i]==1:
            for j in range(num_pmu):
                y_new.append(1)
                
        if y[i]==2:
            for j in range(num_pmu):
                y_new.append(2)
                
        if y[i]==3:
            for j in range(num_pmu):
                y_new.append(3)
        
    
    return X,np.array(y_new)


'''
num_filter=[2,5,10,20,50]
h_filter=[2,5,10,20]
w_filter=[2, 10,100, 1000]
'''


#conv_layers = [1]

#pool_hs=[1]
#pool_ws=[100]
#pool_hs=[1,8,12,24]
#pool_ws=[12,24,100, 360,1000,3000]


#loading training set
p1 =open("X_train2_rocof_6.pickle","rb")
# pickle_in = open("X_train_6.pickle","rb")
p2 = open("X_train_6_planned.pickle","rb")


pk1 = pickle.load(p1)
pk2 = pickle.load(p2)

print('pk1.shape',pk1.shape)
print('pk2.shape',pk2.shape)
X_train=np.concatenate((pk1, pk2), axis=0)
# X_train = pk2
#X_train=X_train*X_train

print('X_train.shape',X_train.shape)

p1 = open("y_train2_rocof_6.pickle","rb")
# pickle_in = open("y_train_6.pickle","rb")

p2 = open("y_train_6_planned.pickle","rb")

pk1 = pickle.load(p1)
pk2 = pickle.load(p2)
print('pk1.shape',pk1.shape)
print('pk2.shape',pk2.shape)
y_train=np.concatenate((pk1, pk2), axis=0)

print('y_train.shape',y_train.shape)

num_pmu=X_train.shape[1]

#cropping input
fps=60
start_crop=int(fps*60*4)
stop_crop=int(fps*60*8)

X_train=X_train[:,:,start_crop:stop_crop,:]

#Removing the planned events
X_train,y_train=removePlanned(X_train,y_train)


###Separate Testing
#X_train, X_test, y_train, y_test=train_test_split(X_train, y_train, test_size=0.3)

##Train/Validation Split
X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=0.25)

#separeting PMUs to make more Training and Validation sets
X_train, y_train= separatePMUs(X_train,y_train)
X_train, y_train = shuffle(X_train, y_train)
X_val, y_val= separatePMUs(X_val,y_val)
X_val, y_val = shuffle(X_val, y_val)
                    

#number of classes
num_classes=len(np.unique(y_train))








#make samples positive
#X_train=abs(X_train)

#normalization
'''
scaler=MinMaxScaler()
X_train_tr=scaler.fit_transform(np.transpose(X_train))
X_train=np.transpose(X_train_tr)
'''




'''
#converting integer labels to one-hot encoding labels
num_classes=4
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
'''

##Constants

EPOCHS=50

all_accuracy=[] #list for reporting all accuracy

iteration=0

dense_layers = [4]
layer_sizes= [200]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
                    
                    iteration+=1
                    print(iteration)
        
                    model = Sequential()
        
                    model.add(Dense(layer_size, input_shape=X_train.shape[1:], kernel_regularizer=regularizers.l2(0.0001)))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.5))


                    for _ in range(dense_layer):
                        model.add(Dense(layer_size))
                        model.add(Activation('relu'))
                        model.add(Dropout(0.5))
                        
                    
                    #fully connected and classifer
                    model.add(Dense(num_classes, activation='softmax'))
                    #model.add(Activation("softmax"))
                    
                    adam = optimizers.Adam(lr=0.001)
        
                    model.compile(loss="sparse_categorical_crossentropy",
                                  optimizer=adam,
                                  metrics=['accuracy'])
                    print(model.summary())
                    
                    #earlyStopping =EarlyStopping(monitor='val_loss', patience=10)
                    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
                    
                    #model.fit(X_train, y_train, batch_size=16, epochs=200, validation_split=0, validation_data=(X_val, y_val))
                    #model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0, validation_data=(X_val, y_val),callbacks=[earlyStopping,mcp_save])
                    model.fit(X_train, y_train, batch_size=16, epochs=EPOCHS, validation_split=0, validation_data=(X_val, y_val),callbacks=[mcp_save])
                    #history=model.fit(X_train, y_train, batch_size=16, epochs=10, validation_split=0, validation_data=(X_val, y_val),callbacks=[earlyStopping])
                    loss_history=pd.DataFrame(model.history.history)
                    loss_history.plot();plt.show();

                    ## get the weights and biases
                    #weight=model.layers[0].get_weights()[0]
                    #bias=model.layers[0].get_weights()[1]
                    
                    # convert the history.history dict to a pandas DataFrame:     
                    #hist_df = pd.DataFrame(history.history) 
                    # get the accuracy
                    #val_acc=hist_df['val_acc'].iloc[0]
                    #print(hist_df['val_acc'].iloc[0])
                    
                    #prediction on validation set
                    model.load_weights(filepath = '.mdl_wts.hdf5')

                    
                    y_pred_percentage=model.predict(X_val)
                    y_pred=model.predict_classes(X_val) 
                    matrix=confusion_matrix(y_val, y_pred)
                    print(matrix)
                    class_report=classification_report(y_val, y_pred)
                    print(class_report)
                    
                    
                    
                    ##prediction on test set

                    '''
                    #loading training set
                    pickle_in = open("X_test_rocof.pickle","rb")
                    X_test = pickle.load(pickle_in)
                    
                    num_pmu=X_test.shape[1]
                    
                    pickle_in = open("y_test_rocof.pickle","rb")
                    y_test= pickle.load(pickle_in)
                    y_test_original=y_test
                    
                    num_test=len(y_test)
                    '''
                    
                    '''
                    y_test_original=y_test
                    num_test=len(X_test)
                    
                    X_test, y_test= separatePMUs(X_test,y_test)
                    
                    ##with separating                    
                    y_pred_percentage=model.predict(X_test)
                    y_pred=model.predict_classes(X_test) 
                    matrix=confusion_matrix(y_test, y_pred)
                    print(matrix)
                    class_report=classification_report(y_test, y_pred)
                    print(class_report)
                    
                    
                    ##Without separation
                    
                    #normalization
                    
                    # scaler=MinMaxScaler()
                    # X_test_tr=scaler.fit_transform(np.transpose(X_test))
                    # X_test=np.transpose(X_test_tr)
                    
                                
                    #acc_history=[]
                    y_pred_vote=[]
                    occurrences_hist=[]
                    #prediction 
                    for i in range(num_test):
                        occurrences=np.zeros([4])
                        y_pred_percentage=model.predict(X_test[i*num_pmu:(i+1)*num_pmu,:])
                        y_pred=model.predict_classes(X_test[i*num_pmu:(i+1)*num_pmu,:])
                        
                        occurrences[0] = np.count_nonzero(y_pred == 0)
                        occurrences[1] = np.count_nonzero(y_pred == 1)
                        occurrences[2] = np.count_nonzero(y_pred == 2)
                        occurrences[3] = np.count_nonzero(y_pred == 3)
                        vote=occurrences.argmax()
                        y_pred_vote.append(vote)
                        #occurrences_hist.append(occurrences)
                        
                        #acc=accuracy_score(y_test[i*num_pmu:(i+1)*num_pmu],y_pred)
                        #acc_history.append(acc)
                    #matrix=confusion_matrix(y_test[i*num_pmu:(i+1)*num_pmu], y_pred)
                    #print(matrix)
                    # occurrences_hist=np.array(occurrences_hist)
                    y_pred_vote=np.array(y_pred_vote)
                    matrix=confusion_matrix(y_test_original, y_pred_vote)
                    print(matrix)
                    acc=accuracy_score(y_test_original, y_pred_vote)
                    print('Accuracy is: '+ str(round(acc,2)))
                    #class_report=classification_report(y_test_original, y_pred_vote)
                    #print(class_report)
                    
                    

                    
                    #matrix_percentage=matrix/len(y_test)*100
                    #con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
                    
                    #get the predictions lables
                    #predictions = model.predict(dataset)
                    #all_accuracy.append('accuracy_'+str(round(acc, 2))+'_num_filter_'+str(layer_size)+'_num_convL_'+str(conv_layer)+'_num_denseL_'+str(dense_layer))
                    '''
                    stop = timeit.default_timer()
                    #running time
                    #print('Time: ', stop - start)





'''
## save  cofusion matrix
with open('cnf_mtrx', 'wb') as f:
    pickle.dump(con_mat, f)
'''

'''
## save  all acuuracies
with open('all_accuracy', 'wb') as f:
    pickle.dump(all_accuracy, f)



pickle_in = open("cnf_mtrx.pickle","rb")
zz = pickle.load(pickle_in)
'''
#model.save('dense_model.h5')
#later_model=load_model('dense_model.h5')
