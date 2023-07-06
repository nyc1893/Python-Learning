from pandas import concat
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error ,mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import numpy as np
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import timeit

# prepare data for lstm
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    
def prepare_data():
     # load dataset
    dataset = read_csv('solar_training.csv', header=0, index_col=0)
    cols = ["VAR78","VAR79","VAR134","VAR157","VAR164","VAR165","VAR166","VAR167","VAR169","VAR175","VAR178","VAR228","POWER"]
    values = dataset.loc[:,cols].values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 24)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[13,14,15,16,17,18,19,20,21,22,23,24]], axis=1, inplace=True)
    print(reframed.head())
    values_ = reframed.values

    # test new data
    dataset_test = read_csv('solar_test.csv', header=0, index_col=0)
    cols = ["VAR78","VAR79","VAR134","VAR157","VAR164","VAR165","VAR166","VAR167","VAR169","VAR175","VAR178","VAR228","POWER"]
    values_test = dataset_test.loc[:,cols].values
    # ensure all data is float
    values_test = values_test.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = scaler.fit_transform(values_test)
    # frame as supervised learning
    reframed_test = series_to_supervised(scaled_test, 1, 24)
    # drop columns we don't want to predict
    temp = reframed_test.columns[[13,14,15,16,17,18,19,20,21,22,23,24]]
    print("temp")
    print(temp[:20])
    reframed_test.drop(reframed_test.columns[[13,14,15,16,17,18,19,20,21,22,23,24]], axis=1, inplace=True)
    # reframed_test.to_csv("train.csv",index= None)
    print(reframed_test.head())

    # split into train and test sets
    values_test_ = reframed_test.values    


    #solar power 1
    # split into train and test sets
    n_train_sample1 = int(len(values)/3)
    train1 = values_[:n_train_sample1, :]
    val1 = values_test_[:int(len(values_test)/3), :]
    # split into input and outputs
    train_X_1, train_y_1 = train1[:, :-1], train1[:, -1]
    val_X_1, val_y_1 = val1[:, :-1], val1[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X_1 = train_X_1.reshape((train_X_1.shape[0], 1, train_X_1.shape[1]))
    val_X_1 = val_X_1.reshape((val_X_1.shape[0], 1, val_X_1.shape[1]))
    print(train_X_1.shape, train_y_1.shape, val_X_1.shape, val_y_1.shape)
    
    return train_X_1, train_y_1, val_X_1, val_y_1
    
def tran_model(epochs_num):
    # design network
    train_X_1, train_y_1, val_X_1, val_y_1 = prepare_data()
    model1 = Sequential()
    model1.add(LSTM(200,activation='softmax', input_shape=(train_X_1.shape[1], train_X_1.shape[2])))
    model1.add(Dropout(0.1))
    model1.add(Dense(1, activation='relu'))
    model1.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    mcp = ModelCheckpoint(os.path.join("best_model.h5"), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')#, save_freq = 'epoch')
    model1.summary()
    # fit network
    history1 = model1.fit(train_X_1, train_y_1, epochs=epochs_num, batch_size=72, validation_data=(val_X_1, val_y_1), verbose=2, shuffle=False,callbacks=[callback, mcp])
    # plot history
    pyplot.plot(history1.history['loss'], label='train')
    pyplot.plot(history1.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def test_model():
# make a prediction
    train_X_1, train_y_1, val_X_1, val_y_1 = prepare_data()
    print(train_y_1[:5])
    
    
def gg():
    saved_model1 = tf.keras.models.load_model('best_model.h5')
    yhat_1 = saved_model1.predict(val_X_1)
    val_X_1 = val_X_1.reshape((val_X_1.shape[0], val_X_1.shape[2]))
    # invert scaling for forecast
    inv_yhat_1 = concatenate((yhat_1, val_X_1[:, 1:]), axis=1)
    #inv_yhat_1 = scaler.inverse_transform(inv_yhat_1)
    inv_yhat_1 = inv_yhat_1[:,0]
    # invert scaling for actual
    val_y_1 = val_y_1.reshape((len(val_y_1), 1))
    inv_y_1 = concatenate((val_y_1, val_X_1[:, 1:]), axis=1)
    #inv_y_1 = scaler.inverse_transform(inv_y_1)
    inv_y_1 = inv_y_1[:,0]
    
    print(np.max(inv_y_1))
    print(np.max(inv_yhat_1))
    
    # calculate RMSE and MAE
    # rmse_val_1 = sqrt(mean_squared_error(inv_y_1, inv_yhat_1))
    # print('Val RMSE: %.3f' % rmse_val_1)
    # mae_val_1 = mean_absolute_error(inv_y_1, inv_yhat_1)
    # print('Val MAE: %.3f' % mae_val_1)
    max_y = 1
    print(" NMAE=",100*mae_val_1/max_y)
    print(" NRMSE=",100*rmse_val_1/max_y)  


def main():
    s1 = timeit.default_timer()  
    test_model()
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
""" 
"""

if __name__ == "__main__":
    main()
