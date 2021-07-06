import pandas as pd
import numpy as np

from random import random
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.optimizers import Adam
from keras.layers import Activation, Dense, BatchNormalization, TimeDistributed
from keras import regularizers


    
def rd_non(ii):

    path2 = 'index2/'
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    path1 ="../stat/"
    df1 = pd.read_csv(path1+"non_"+str(ii)+'.csv')
    y = np.zeros(df1.shape[0])
    X = df1.values
    # y = y.values
    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]

    return X_train,y_train,X_val,y_val      
    
def rd_eve(ii):
    path1 ="../stat/"
    path2 = 'index2/'
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    df1 = pd.read_csv(path1+"com_"+str(ii)+'.csv')
    y = df1.pop("label")
    X = df1.values
    y = y.values
    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]

    return X_train,y_train,X_val,y_val    
    
def rd_eve2(ii):
    path1 ="../stat/"
    path2 = 'index2/'
    
    path3 ="../dif_scal/30/"
    
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    df1 = pd.read_csv(path1+"com_"+str(ii)+'.csv')
    dy1 = pd.read_csv(path3 + "y_S"+str(ii)+".csv")
    df1.pop("label")
    y = dy1["label"]
    y = y.values
    y[y!=1] = 0
    
    X = df1.values
    
    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]

    return X_train,y_train,X_val,y_val  
    
def get_non():
    ii =1

    X_train,y_train,X_val,y_val = rd_non(1)

    for ii in range(2,8):
        X_train2,y_train2,X_val2,y_val2 = rd_non(ii)
        X_train = np.concatenate((X_train, X_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        
        X_val = np.concatenate((X_val, X_val2), axis=0)
        y_val = np.concatenate((y_val, y_val2), axis=0)  

    return X_train,y_train,X_val,y_val    
    
def get_eve():
    ii =1

    X_train,y_train,X_val,y_val = rd_eve(1)

    for ii in range(2,14):
        X_train2,y_train2,X_val2,y_val2 = rd_eve(ii)
        X_train = np.concatenate((X_train, X_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        
        X_val = np.concatenate((X_val, X_val2), axis=0)
        y_val = np.concatenate((y_val, y_val2), axis=0)  

    return X_train,y_train,X_val,y_val
    
def get_eve2():
    ii =1

    X_train,y_train,X_val,y_val = rd_eve2(1)

    for ii in range(2,14):
        X_train2,y_train2,X_val2,y_val2 = rd_eve2(ii)
        X_train = np.concatenate((X_train, X_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        
        X_val = np.concatenate((X_val, X_val2), axis=0)
        y_val = np.concatenate((y_val, y_val2), axis=0)  

    return X_train,y_train,X_val,y_val
    

def top2():
    X_train,y_train,X_val,y_val = get_eve()
    X_train2,y_train2,X_val2,y_val2 = get_non()
    
    X_train = np.concatenate((X_train, X_train2), axis=0)
    y_train = np.concatenate((y_train, y_train2), axis=0)
    
    X_val = np.concatenate((X_val, X_val2), axis=0)
    y_val = np.concatenate((y_val, y_val2), axis=0)      
    
    print(X_train.shape)
    print(y_train.shape)
    return X_train,y_train,X_val,y_val
    
def data():

    dt1 = get_nor()
    dt2,dt3 = get_eve2()
    x =[]
    real = []
    for i in range(dt1.shape[0]):
        x.append(dt1[i])
        real.append(0)

    
    for i in range(dt2.shape[0]):
        # temp = dt2[i]
        x.append(dt2[i])
        real.append(dt3[i])

    x = np.array(x)
    real = np.array(real)
    return  x,real
    
def data_pack(max_time,epochs_num,a1,a2,a3):
    
    X_train,y_train,X_val,y_val = top2()
    X_train = X_train[:,np.newaxis,:]
    X_val = X_val[:,np.newaxis,:]
    
    model = Sequential()
    model.add(LSTM(a1, input_shape=(X_train.shape[1], X_train.shape[2]),kernel_regularizer=regularizers.l1(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(a2))
    model.add(Activation('relu'))
    model.add(Dense(a3))
    model.add(Activation('tanh'))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # model.fit(X_train, y_train, batch_size=64, validation_split = 0.2, epochs=50, shuffle=False, verbose=1)

    history = model.fit(X_train,y_train, epochs=epochs_num, batch_size=32, validation_data=(X_val,y_val), verbose=1, shuffle=True)

    yhat = model.predict(X_val)

    value = accuracy_score(yhat,y_val)
    f_name ='model/'+str(a1)+'-'+str(a2)+'-'+str(a3)+'.h5'
    model.save(f_name)
    
    for  i in range(2,max_time+1):


        model = Sequential()
        model.add(LSTM(a1, input_shape=(X_train.shape[1], X_train.shape[2]),kernel_regularizer=regularizers.l1(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(a2))
        model.add(Activation('relu'))
        model.add(Dense(a3))
        model.add(Activation('tanh'))
        model.add(Dense(1, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train,y_train, epochs=epochs_num, batch_size=32, validation_data=(X_val,y_val), verbose=1, shuffle=True)

        yhat = model.predict(X_val)
        # .round()
        acc = accuracy_score(yhat,y_val)
        # f_name ='model/'+str(a1)+'-'+str(a2)+'-'+str(a3)+'.h5'
    
        if value< acc:
            value  = acc
            model.save(f_name)


    return value
        
    
from bayes_opt import BayesianOptimization    
import timeit    



def black_box_function(aa1,aa2,aa3):

    a1 = int(round(aa1))
    a2 = int(round(aa2))    
    a3 = int(round(aa3))

    max_time =4
    epochs_num =100
    return data_pack(max_time,epochs_num,a1,a2,a3)



def main():

    # data_pack(70,70,200,200,'mit')

    s1 = timeit.default_timer()  
    # Bounded region of parameter space
    pbounds = {'aa1': (30, 100), 'aa2': (20, 100), 'aa3': (20, 100)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=40
    )

    # print(optimizer.max)  


    tt = str(optimizer.max)
    with open('lstm_osc.txt','a+') as f:    #设置文件对象
        f.write(tt+'\n')                 #将字符串写入文件中
    print(tt)    
    
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
""" 
"""

if __name__ == "__main__":
    main()


