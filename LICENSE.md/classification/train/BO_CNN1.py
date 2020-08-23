
#For Rocof data after wavelet

import keras
import np_utils
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
from sklearn.utils import shuffle
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import pickle
import timeit
import datetime
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# import pyarrow.parquet as pq
import timeit
import pickle

start = timeit.default_timer()

sampling_rate = 1800
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


def rd2(k):
    path1 = '../pickleset/'

    list = ['rocof','v_grad','i_grad', 'vp_a_diff_grad', 'ip_a_diff_grad','f_grad']
    
    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[0])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    # len(list)
    # for i in range(1,len(list)):
        # p2 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
        # pk2 = pickle.load(p2)    
    
        # pk1=np.concatenate((pk1, pk2), axis=3)
        
    fps=60
    start_crop=int(fps*60*4.95)
    stop_crop=int(fps*60*5.05)

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
    
    print(X_train.shape) 
    print(X_val.shape) 
    print(y_train.shape) 
    print(y_val.shape) 
    

    return X_train, X_val, y_train, y_val    
    
def Cwt(X_train):    
    wavename = 'morl'
    pca_2 = PCA(n_components=100)
    # totalscal = 11
    # fc = pywt.central_frequency(wavename)
    # cparam = 2 * fc * totalscal
    # print(data.shape)
    scales = np.arange(1, 101)
    # scales = cparam / np.arange(totalscal, 1, -1)
    # data = X_train
    # t = range(sampling_rate)
    wavelet = []
    for i in range(0,X_train.shape[0]):
        data = X_train[i].flatten()
        
        # print('data.shape',data.shape)   
        # print('t.shape',len(t)) 
        [coeff1, freqs1] = pywt.cwt(data, scales, wavename)
        
        

        wavelet.append(pca_2.fit_transform(coeff1))
    wavelet = np.array(wavelet)
    print(wavelet.shape)
    return wavelet
        # print('frequencies.shape',frequencies.shape)
        # print('cwtmatr.shape',cwtmatr.shape)
        # print('cwtmatr[0].shape',cwtmatr[0].shape)      

def data_pack():
    X_train, X_val, y_train, y_val = rd2(1)
    X_train,  y_train = separatePMUs (X_train,  y_train)
    X_val,  y_val = separatePMUs (X_val,  y_val)
    
    X_train =Cwt(X_train)
    X_val =Cwt(X_val)
    num= X_train.shape[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    p1 = np.concatenate((X_train,X_val))
    # scaler.fit(p1)
    print(np.amax(p1))
    # p2 = scaler.transform(p1)
    # print(np.amax(p2))

    
    X_train = p1[:num]
    X_val = p1[num:]
    
    X_train = X_train[:,:,:,np.newaxis]
    X_val = X_val[:,:,:,np.newaxis]    

    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    print('y_val.shape',y_val.shape) 
    return  X_train,y_train, X_val, y_val
    


# def load_data():    
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
        
def fake():    
    s_x = 10
    s_y = 3600
    channel_num = 1
    c_num =4
    tr_num =  20
    test_num = 5
    X_train = np.random.random((tr_num, s_x, s_y,channel_num))
    y_train = keras.utils.to_categorical(np.random.randint(c_num, size=(tr_num, 1)), num_classes=c_num)
    # y_train = (np.random.randint(c_num, size=(100, 1)), num_classes=c_num)
    X_test = np.random.random((test_num, s_x, s_y,channel_num))
    y_test = keras.utils.to_categorical(np.random.randint(c_num, size=(test_num, 1)), num_classes=c_num)
    y_test = recover(y_test)
    y_train = recover(y_train)
    print(X_test.shape)
    print(y_test.shape)
    
    
    return X_train,y_train,X_test, y_test
best_accuracy = 0.0   

def main():

    X_train,y_train,X_test,y_test = read_data()
    validation_set = (X_test,y_test)
    ##Constants

    EPOCHS=2
    BATCH_SIZE=16
    
    

    #number of classes
    num_classes=len(np.unique(y_train))

    #print(X.shape[0:])
    #hyperparameters
    learning_rate=0.0035
    num_conv_filters=4

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
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))




    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model.summary()
    model.fit(X_train, y_train, batch_size=32, epochs=2)


    y_pred=model.predict_classes(X_test) 




    # print(y_test.shape)
    print(y_pred.shape)

    matrix=confusion_matrix(y_test, y_pred)
    print(matrix)
    class_report=classification_report(y_test, y_pred)
    print(class_report)

##prediction on test set



    
  
def main2():
    X_train, y_train, X_test, y_test=   data_pack()
    # data_pack()

    validation_set = (X_test, y_test)
    num_classes=len(np.unique(y_train))


    
    ck = '2016cnn-'
    dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                             name='learning_rate')
    dim_num_conv_filters = Integer(low=4, high=32, name='num_conv_filters')
    dim_size_conv_filters=Integer(low=4, high=20, name='size_conv_filters')
    dim_num_dense_layers = Integer(low=1, high=4, name='num_dense_layers')
    dim_num_dense_nodes = Integer(low=10, high=80, name='num_dense_nodes')
    dim_drop_out_input = Real(low=0.4, high=0.9, name='drop_out_rate_input')
    dim_drop_out_hidden = Real(low=0.2, high=0.7, name='drop_out_rate_hidden')

    dimensions = [dim_learning_rate,
                  dim_num_conv_filters,
                  dim_size_conv_filters,
                  dim_num_dense_layers,
                  dim_num_dense_nodes,
                  dim_drop_out_input,
                  dim_drop_out_hidden]

                  
    def log_dir_name(   learning_rate,num_conv_filters,
                        size_conv_filters, num_dense_layers,
                        num_dense_nodes,
                        drop_out_rate_input, drop_out_rate_hidden):

        # The dir-name for the TensorBoard log-dir.
        # s = "./19_logs/lr_{0:.0e}_conv_filters_{1}_conv_filter_size_{2}_dense_layers_{3}_nodes_{4}_dropout_input_{5}_dropout_hidden_{6}_activation_{7}/"
        s = "./19_logs/lr_{0:.0e}_conv_filters_{1}_conv_filter_size_{2}_dense_layers_{3}_nodes_{4}_dropout_input_{5}_dropout_hidden_{6}/"
        # Insert all the hyper-parameters in the dir-name.
        log_dir = s.format(learning_rate,
                           num_conv_filters,
                           size_conv_filters,
                           num_dense_layers,
                           num_dense_nodes,
                           drop_out_rate_input,
                           drop_out_rate_hidden
                            )

        return log_dir
        
        
        
    def create_model(
    
                    num_conv_filters,
                    size_conv_filters,
                    learning_rate,
                    num_dense_layers,
                    num_dense_nodes,
                    drop_out_rate_input,
                    drop_out_rate_hidden
                    ):



        channel_num = X_train.shape[3]
        s_x = X_train.shape[1]
        s_y = X_train.shape[2]
        
        all_accuracy=[] #list for reporting all accuracy



        model = Sequential()
        
        model.add(Conv2D(num_conv_filters, (3, size_conv_filters), activation='relu', input_shape=(s_x, s_y, channel_num)))
        # model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(drop_out_rate_input))

        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        
        model.add(Flatten())
        for i in range(num_dense_layers):
            # Name of the layer. This is not really necessary
            # because Keras should give them unique names.
            name = 'layer_dense_{0}'.format(i+1)

            model.add(Dense(num_dense_nodes,
                            activation='relu',
                            name=name))
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
        
        return model

        
        
    path_best_model = ck + 'best_model.hdf5'
    best_accuracy = 0.0

    @use_named_args(dimensions=dimensions)
    def fitness(learning_rate,num_conv_filters,
                           size_conv_filters, num_dense_layers,
                num_dense_nodes, drop_out_rate_input,
                           drop_out_rate_hidden):
    
        # Print the hyper-parameters.
        print('*************************** NEW ITERATION IS STARTED****************************')
        print('learning rate: {0:.1e}'.format(learning_rate))
        print('num_conv_filters:', num_conv_filters)
        print('size_filter:', size_conv_filters)
        print('num_dense_layers:', num_dense_layers)
        print('num_dense_nodes:', num_dense_nodes)
        print('drop_out_rate_input:', drop_out_rate_input)
        print('drop_out_rate_hidden:', drop_out_rate_hidden)
        # print('activation:', activation)
        print()
        
        # Create the neural network with these hyper-parameters.
        model = create_model(learning_rate=learning_rate,
                             num_conv_filters=num_conv_filters,
                             size_conv_filters=size_conv_filters,
                             num_dense_layers=num_dense_layers,
                             num_dense_nodes=num_dense_nodes,
                             drop_out_rate_input=drop_out_rate_input,
                             drop_out_rate_hidden=drop_out_rate_hidden
                             )

        # Dir-name for the TensorBoard log-files.
        log_dir = log_dir_name(learning_rate,num_conv_filters,
                           size_conv_filters, num_dense_layers,
                               num_dense_nodes,drop_out_rate_input,
                           drop_out_rate_hidden)    
                           
        callback_log = TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            write_grads=False,
            write_images=False)    
            
        file_name=ck+ "best_model_weights.hdf5"

        mcp_save = ModelCheckpoint(file_name, save_best_only=True, monitor='val_accuracy', mode='max')
        EPOCHS=2
        BATCH_SIZE=16                           
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=2,
                            validation_data=validation_set,
                            callbacks=[callback_log,mcp_save])

        # Get the classification accuracy on the validation-set
        # after the last training-epoch.
        #accuracy = history.history['val_accuracy'][-1] #last epoch accuracy in each call
        accuracy = max(history.history['val_accuracy']) #best accuracy among all epochs in each call
        print()
        print("Accuracy: {0:.2%}".format(accuracy))
        print()
        
        #y_pred_percentage=model.predict(X_test)
        # X_test,y_test 
        y_pred=model.predict_classes(X_test) 
        matrix=confusion_matrix(y_test, y_pred)
        print(matrix)
        class_report=classification_report(y_test, y_pred)
        print(class_report)

        # Save the model if it improves on the best-found performance.
        # We use the global keyword so we update the variable outside
        # of this function.
        global best_accuracy
        # If the classification accuracy of the saved model is improved ...
        if accuracy > best_accuracy:
            # Save the new model to harddisk.
            
            model.load_weights(ck + "best_model_weights.hdf5")# save best model's weights
            model.save(ck + "best_model_so_far.h5") # save the entire model configurartion and weights
            print('**** THE BEST ACCURACY SO FAR IS ACHIEVED AND THE MODEL IS SAVED **** \n')
            
            # save the hyperparameters
            pickle_out = open(ck + "hp.pickle","wb")
            pickle.dump([learning_rate, num_conv_filters,
                           size_conv_filters,num_dense_layers,num_dense_nodes,drop_out_rate_input,
                           drop_out_rate_hidden], pickle_out, protocol=2)
            pickle_out.close() 

            # Update the classification accuracy.
            best_accuracy = accuracy
              

        # Delete the Keras model with these hyper-parameters from memory.
        del model
        
        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()
        
        # NOTE: Scikit-optimize does minimization so it tries to
        # find a set of hyper-parameters with the LOWEST fitness-value.
        # Because we are interested in the HIGHEST classification
        # accuracy, we need to negate this number so it can be minimized.
        return -accuracy                           
 
    ###Run the Hyper-Parameter Optimization####
    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=10)

    stop = timeit.default_timer()
    #running time
    print('Overall Time:(mins)', (stop - start)/60)

    plot_convergence(search_result)
    plt.savefig(ck+'-plot1.png', format='png')

if __name__ == '__main__':  

    s1 = timeit.default_timer()
    main2()
    # rd2(1)
    # data_pack()
    s2 = timeit.default_timer()
    #running time
    print('Time(mins): ', (s2 - s1)/60 )
        
        
