
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:17:10 2020

THis code classifies the events


@author: iniazazari
"""




from pyts.image import MarkovTransitionField
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.optimizers import Adam


from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# import pyarrow.parquet as pq
import timeit
import pickle

start = timeit.default_timer()

def rm2(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X[i,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:])
            
        elif y[i]==4:
            y_new.append(0)
            X_new.append(X[i,:])
        
        elif y[i]==5:
            y_new.append(1)
            X_new.append(X[i,:])
        

    return  np.array(X_new), np.array(y_new)
    

  
def rd1(num):
    path1 = 'gen/'
    p1 = open(path1 +'pack_'+str(num),"rb")
    vp,rof,label,vp2,rof2,label2 = pickle.load(p1)
    
    # print('vp.shape',vp.shape) 
    # print('rof.shape',rof.shape) 
    # print('label.shape',label.shape) 
    # print('vp2.shape',vp2.shape) 
    # print('rof2.shape',rof2.shape) 
    # print('label2.shape',label2.shape)     
    
    return vp,rof,label,vp2,rof2,label2

    
def data_pack():
    X_train,X_trainn,y_train,X_val,X_vall,y_val = rd1(1)
    for i in range(2,3+1):
        X_train2,X_trainn2,y_train2,X_val2,X_vall2,y_val2= rd1(i)
        X_train = np.concatenate((X_train,X_train2))
        X_trainn = np.concatenate((X_trainn,X_trainn2))
        y_train = np.concatenate((y_train,y_train2))
        X_val = np.concatenate((X_val,X_val2))
        X_vall = np.concatenate((X_vall,X_vall2))
        y_val = np.concatenate((y_val,y_val2))
        
    X_train = X_train[:,:,:,np.newaxis] 
    X_trainn = X_trainn[:,:,:,np.newaxis] 
    X_val = X_val[:,:,:,np.newaxis] 
    X_vall = X_vall[:,:,:,np.newaxis] 
    
    X_train =  np.concatenate((X_train,X_trainn),axis = 3)  
    X_val = np.concatenate((X_val,X_vall),axis = 3)  
    print('X_train.shape',X_train.shape) 
    # print('X_trainn.shape',X_trainn.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    # print('X_vall.shape',X_vall.shape) 
    print('y_val.shape',y_val.shape) 
    # return  X_train,X_trainn,y_train,X_val,X_vall,y_val 
    return  X_train,y_train,X_val,y_val 



    
def main():    
    X_train, y_train, X_test, y_test=data_pack()

    validation_set = (X_test,y_test)
    ##Constants

    EPOCHS=30
    BATCH_SIZE=16
    
    

    
    #number of classes
    num_classes=len(np.unique(y_train))

    #print(X.shape[0:])
    #hyperparameters
    learning_rate=0.01
    num_conv_filters=12
    size_conv_filters= 4
    num_dense_layers=2
    num_dense_nodes=100
    drop_out_rate_input=0.5
    drop_out_rate_hidden=0.5

    
    channel_num = X_train.shape[3]
    s_x = X_train.shape[1]
    s_y = X_train.shape[2]
    
    all_accuracy=[] #list for reporting all accuracy



    model = Sequential()
    
    model.add(Conv2D(32, (1, 1), use_bias=True,input_shape= (s_x, s_y, channel_num)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(32, (1, 1), use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(64, (1, 1), use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(64, (1, 1), use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_out_rate_input))
    
    model.add(Conv2D(128, (1, 1), use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))   
    
    model.add(Conv2D(128, (1, 1), use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_out_rate_hidden))

    
    model.add(Flatten())

    model.add(Dense(num_dense_nodes,
                    activation='relu'))
    model.add(Dropout(drop_out_rate_hidden))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))

    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)

    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
                            
    #earlyStopping =EarlyStopping(monitor='val_loss', patience=10)
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

    #model.fit(X_train, y_train, batch_size=16, epochs=200, validation_split=0, validation_data=(X_val, y_val))
    #model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0, validation_data=(X_val, y_val),callbacks=[earlyStopping,mcp_save])
    model.fit(X_train, y_train, batch_size=16, epochs=EPOCHS, validation_split=0, validation_data=(X_test, y_test))
    #history=model.fit(X_train, y_train, batch_size=16, epochs=10, validation_split=0, validation_data=(X_val, y_val),callbacks=[earlyStopping])
    loss_history=pd.DataFrame(model.history.history)
    loss_history.plot();

    ## get the weights and biases


    # y_pred_percentage=model.predict(X_val)
    y_pred=model.predict_classes(X_test) 
    matrix=confusion_matrix(y_test, y_pred)
    print(matrix)
    class_report=classification_report(y_test, y_pred)
    print(class_report)
                    
                    


##prediction on test set



        

if __name__ == '__main__':  

    s1 = timeit.default_timer()
    # rd1(1)
    # data_pack()
    # datamiddle(50)
    # datamtop()
    main()
    
    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
        
        
