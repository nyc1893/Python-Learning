
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:17:10 2020

THis code classifies the events


@author: iniazazari
"""






from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
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
            X_new.append(X[i,:,:,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:,:,:])
        

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
    y2=[]
    for i in range(len(y)):
        if y[i]==0:
            for j in range(num_pmu):
                y2.append(0)
                
        elif y[i]==1:
            for j in range(num_pmu):
                y2.append(1)
                
        elif y[i]==2:
            for j in range(num_pmu):
                y2.append(2)
                
        elif y[i]==3:
            for j in range(num_pmu):
                y2.append(3)
        elif y[i]==4:
            for j in range(num_pmu):
                y2.append(4)       
        elif y[i]==5:
            for j in range(num_pmu):
                y2.append(5)    
    return X,np.array(y2)



    
    
def rd1(sav,k):
    path1 = '../pickleset/'

    list = ['rocof','v_grad','i_grad', 'vp_a_diff_grad', 'ip_a_diff_grad','f_grad']
    
    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[0])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    # len(list)
    # for i in range(1,len(list)):
        # p2 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
        # pk2 = pickle.load(p2)    
    
        # pk1=np.concatenate((pk1, pk2), axis=3)
        
    fps=1
    start_crop=int(fps*2*1)
    stop_crop=int(fps*2*1)

    pk1=pk1[:,:,start_crop:stop_crop,:]
    
    p3 = open(path1 + 'y_S'+str(k)+'_rocof_6.pickle',"rb")
    pk3 = pickle.load(p3)        
    # print(pk1.shape) 
    # print(pk3.shape)        
    
    X_train,y_train=removePlanned(pk1,pk3)
    num = y_train.shape[0]
    
    a = np.arange(0,num)
    tr,val = train_test_split(a,test_size=0.25)    
    path2 = 'index/'
    np.save(path2+'tr_'+str(k)+'.npy',tr) 
    np.save(path2+'val_'+str(k)+'.npy',val) 
    
def recover(df):
    df = pd.DataFrame(df)
    df.columns = ['L1','L2','L3','L4']
    dd = np.zeros(df.shape[0])
    dd = dd.astype("int8") 
    rr = df[df['L2']>0].index.tolist()
    dd[rr] = 1
    
    rr = df[df['L3']>0].index.tolist()
    dd[rr] = 2    


    rr = df[df['L4']>0].index.tolist()
    dd[rr] = 3    
    
    df['rev'] = dd
    return df['rev'].values 
    
def rd2(k):
    path1 = '../pickleset/'

    list = ['rocof','v_grad','i_grad', 'vp_a_diff_grad', 'ip_a_diff_grad','f_grad']
    
    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[0])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    # len(list)
    for i in range(1,len(list)):
        p2 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
        pk2 = pickle.load(p2)    
    
        pk1=np.concatenate((pk1, pk2), axis=3)
        
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)

    pk1=pk1[:,:,start_crop:stop_crop,:]
    
    p3 = open(path1 + 'y_S'+str(k)+'_rocof_6.pickle',"rb")
    pk3 = pickle.load(p3)        
    
    path2 = 'index/'
    tr=np.load(path2 +'tr_' +str(k)+'.npy')
    val=np.load(path2 +'val_' +str(k)+'.npy')
    tr=tr.tolist()  
    val=val.tolist() 
    # b = a[c]
    
    pk1,pk3=removePlanned(pk1,pk3)
    X_train = pk1[tr]
    y_train = pk3[tr]
    X_val = pk1[val]
    y_val = pk3[val]
    
    # print(X_train.shape) 
    # print(X_val.shape) 
    # print(y_train.shape) 
    # print(y_val.shape) 
    


    
    return X_train, X_val, y_train, y_val    
    

  
  
  
  
  
def read_data():    
    # X_train=np.concatenate((pk1, pk2), axis=0)
    X_train, X_val, y_train, y_val = rd2(1)
    for i in range(2,6+1):
        X_train1, X_val1, y_train1, y_val1 = rd2(i)
        X_train = np.concatenate((X_train, X_train1), axis=0)
        y_train = np.concatenate((y_train, y_train1), axis=0)
        X_val = np.concatenate((X_val, X_val1), axis=0)
        y_val = np.concatenate((y_val, y_val1), axis=0)        
        
    print(X_train.shape) 
    print(X_val.shape) 
    print(y_train.shape) 
    print(y_val.shape) 
    print(type(X_train))
    return X_train, y_train, X_val, y_val

# def load_data():    



def main():
    # s1 = timeit.default_timer()
    X_train,y_train,X_test,y_test = read_data()
    validation_set = (X_test,y_test)
    ##Constants

    EPOCHS=30
    BATCH_SIZE=16
    
    

    
    #number of classes
    num_classes=len(np.unique(y_train))

    #print(X.shape[0:])
    #hyperparameters
    learning_rate=0.0035
    num_conv_filters=4
    size_conv_filters= X_train.shape[0]/100
    num_dense_layers=1
    num_dense_nodes=100
    drop_out_rate_input=0.9
    drop_out_rate_hidden=0.2

    
    channel_num = X_train.shape[3]
    s_x = X_train.shape[1]
    s_y = X_train.shape[2]
    
    all_accuracy=[] #list for reporting all accuracy



    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(s_x, s_y, channel_num)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dense(4*3597, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))



    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=0.001)

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
    #weight=model.layers[0].get_weights()[0]
    #bias=model.layers[0].get_weights()[1]

    # convert the history.history dict to a pandas DataFrame:     
    #hist_df = pd.DataFrame(history.history) 
    # get the accuracy
    #val_acc=hist_df['val_acc'].iloc[0]
    #print(hist_df['val_acc'].iloc[0])

    #prediction on validation set



    # y_pred_percentage=model.predict(X_val)
    y_pred=model.predict_classes(X_test) 
    matrix=confusion_matrix(y_test, y_pred)
    print(matrix)
    class_report=classification_report(y_test, y_pred)
    print(class_report)
                    
                    


##prediction on test set


def gen_data():
    for i in range(4,6+1):
        rd1(1,i)
        # rd2(i)


def read22():
    for i in range(1,4):
        # rd1(1,i)
        rd2(i)
        

if __name__ == '__main__':  

    s1 = timeit.default_timer()
    main()
    # read_data()
    # gen_data()
    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
        
        
