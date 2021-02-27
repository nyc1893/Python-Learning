
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
import heapq
import datapick

# import pyarrow.parquet as pq
import timeit
import pickle

start = timeit.default_timer()


def get_data():
    X_train,y_train,X_val,y_val = datapick.load_data2(1)
    for k in range(2,14):
        X_train2,y_train2,X_val2,y_val2 = datapick.load_data2(k)
        X_train = np.concatenate((X_train, X_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        
        X_val = np.concatenate((X_val, X_val2), axis=0)
        y_val = np.concatenate((y_val, y_val2), axis=0)       

    return X_train,y_train,X_val,y_val
    
def loadmodel():    
    from tensorflow.keras.models import load_model
    X_train, y_train, X_test, y_test= get_data()
    validation_set = (X_test,y_test)
    model = load_model('bm.h5')    
    # y_pred_percentage=model.predict(X_val)
    y_pred=model.predict_classes(X_test) 
    matrix=confusion_matrix(y_test, y_pred)
    print(matrix)
    class_report=classification_report(y_test, y_pred)
    print(class_report)
    df = pd.DataFrame(y_test)
    df['y_pred'] = y_pred

    df.to_csv("df.csv",index = None)
                             
def main():    
    X_train, y_train, X_test, y_test= get_data()
    # X_train = select(X_train)
    # X_test = select(X_test)
    validation_set = (X_test,y_test)
    ##Constants

    EPOCHS=20
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
    drop_out_rate_hidden=0.7

    
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
    # model.add(Dropout(drop_out_rate_hidden))

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
    model.save("bm.h5")
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

                    


        

if __name__ == '__main__':  

    s1 = timeit.default_timer()
    # rd1(1)
    # data_pack()
    # datamiddle(50)
    # datamtop()
    main()
    # loadmodel()
    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
        
        
