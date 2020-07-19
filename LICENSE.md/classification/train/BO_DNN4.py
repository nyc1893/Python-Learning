
# This is cross_valiation code
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers,regularizers
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection  import StratifiedKFold
from sklearn.model_selection  import cross_val_score

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

import pickle
import timeit
import datetime

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



#default_parameters = [1e-5, 1, 16, 'relu']

def log_dir_name(learning_rate, num_dense_layers,
                drop_out_rate_input, drop_out_rate_hidden,
                num_dense_nodes, activation):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_dropout_input_{3}_dropout_hidden_{4}_activation_{5}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       drop_out_rate_input,
                       drop_out_rate_hidden,
                       activation)

    return log_dir





    
def read_data(num):  

    # num = 1
    path = '2016-'
    p2 = open(path+ "tr_set.pickle","rb")
    X_train, y_train= pickle.load(p2)


    p2 = open(path+ "va_set.pickle","rb")
    X_test, y_test= pickle.load(p2)
    y_train = y_train[:,l]
    
    y_test = y_test[:,l]    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    


    # only shuffle X_train part 
    # X_train, y_train =removePlanned(X_train, y_train)
    X_train, y_train=separatePMUs(X_train, y_train)
    # X_train, y_train = shuffle(X_train, y_train)   

    # X_test, y_test=removePlanned(X_test, y_test)
    X_test, y_test= separatePMUs(X_test,y_test)
 
    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    
    
    
    return X_train,y_train,X_test,y_test    
    
    
    
    
def check_y_test(y_test):
    # X_test,y_test = read_data()
    i = 0
    num = y_test.shape[0]/23
    # num = 2
    # j = 0
    flag = 0
    for i in range(1,int(num)+1):
        ind1 = 23*(i-1)
        ind2 = 23*i
        
        data =  y_test[ind1:ind2]
        # print(data)
        for j in range(1,23):
            if(data[0] != data[j]):
                flag = 1
        if flag == 1:
            print('Not equal in ', i)   
    return flag
  
def vote_result(y_test):

    i = 0
    list = []
    
    num = y_test.shape[0]/23
    # num = 2
    # j = 0
    flag = 0
    for i in range(1,int(num)+1):
        ind1 = 23*(i-1)
        ind2 = 23*i
        
        data =  y_test[ind1:ind2]
        b = np.argmax(np.bincount(data))
        # print(data)
        for j in range(1,23+1):
            list.append(b)

    df = pd.DataFrame(list) 
    if(check_y_test(df.values)==0):
        print('label is formated!')   
        
    return df.values    
best_accuracy = 0.0    

def main():
    # list = ['real','SN','KNN',	'LR',	'RF',	'DT',	'SVM',	'GBDT']
    
    # ck = list[i]
    ck = 'S13-'

    ####Train and Evaluate the Model###
    path_best_model = ck + '-best_model.hdf5'
    X_train,y_train,X_test,y_test = read_data()

    
    num_classes=len(np.unique(y_train))
    EPOCHS=25
    BATCH_SIZE=16
    ##SET UP the RANGE of HPs
    dim_learning_rate = Real(low=1e-5, high=1e-1, prior='log-uniform',
                             name='learning_rate')
    dim_num_dense_layers = Integer(low=1, high=6, name='num_dense_layers')
    dim_num_dense_nodes = Integer(low=20, high=500, name='num_dense_nodes')
    dim_drop_out_input = Real(low=0.4, high=0.9, name='drop_out_rate_input')
    dim_drop_out_hidden = Real(low=0.2, high=0.7, name='drop_out_rate_hidden')
    dim_activation = Categorical(categories=['relu', 'sigmoid'],
                                 name='activation')
    dimensions = [dim_learning_rate,
                  dim_num_dense_layers,
                  dim_num_dense_nodes,
                  dim_drop_out_input,
                  dim_drop_out_hidden,
                  dim_activation]
    
    @use_named_args(dimensions=dimensions)
    def fitness(learning_rate, num_dense_layers,
                num_dense_nodes, drop_out_rate_input,
                           drop_out_rate_hidden, activation):
        """
        Hyper-parameters:
        learning_rate:          Learning-rate for the optimizer.
        num_dense_layers:       Number of dense layers.
        num_dense_nodes:        Number of nodes in each dense layer.
        drop_out_rate_input:    Rate of dropout in input layer.
        drop_out_rate_hidden:   Rate of dropout in hidden layer.
        activation:             Activation function for all layers.
        """
        
        # Print the hyper-parameters.
        print('*************************** NEW ITERATION IS STARTED****************************')
        print('learning rate: {0:.1e}'.format(learning_rate))
        print('num_dense_layers:', num_dense_layers)
        print('num_dense_nodes:', num_dense_nodes)
        print('drop_out_rate_input:', drop_out_rate_input)
        print('drop_out_rate_hidden:', drop_out_rate_hidden)
        print('activation:', activation)
        print()
        
        

        # Create the neural network with these hyper-parameters.
        
        
        
        model = create_model(learning_rate=learning_rate,
                             num_dense_layers=num_dense_layers,
                             num_dense_nodes=num_dense_nodes,
                             drop_out_rate_input=drop_out_rate_input,
                             drop_out_rate_hidden=drop_out_rate_hidden,
                             activation=activation)
        
        # Dir-name for the TensorBoard log-files.
        log_dir = log_dir_name(learning_rate, num_dense_layers,
                               num_dense_nodes,drop_out_rate_input,
                           drop_out_rate_hidden, activation)
        
        # Create a callback-function for Keras which will be
        # run after each epoch has ended during training.
        # This saves the log-files for TensorBoard.
        # Note that there are complications when histogram_freq=1.
        # It might give strange errors and it also does not properly
        # support Keras data-generators for the validation-set.
        callback_log = TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            write_grads=False,
            write_images=False)
        # path2= 'bodnn-'
        
        
        file_name= ck+ "best_model_weights.hdf5"
        # calls=[EarlyStopping(monitor='acc', patience=10), ModelCheckpoint(file_name, monitor='acc', save_best_only=True, mode='auto', period=1)]

        
        
        mcp_save = ModelCheckpoint(file_name, save_best_only=True, monitor='val_accuracy', mode='max')

        X,y,X_test,y_test = read_data()

        kfold = StratifiedKFold(n_splits=4, shuffle = False)

        kfold.get_n_splits(X, y)
                
        for train_index, test_index in kfold.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
            print(X_train.shape)
            validation_set = (X_val,y_val)
        


            # Use Keras to train the model.
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
            # accuracy = max(history.history['val_accuracy']) #best accuracy among all epochs in each call
            y_pred=model.predict_classes(X_test) 
            # y_pred = vote_result(y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            


        


        # Print the classification accuracy.
        # print()
        # print("Accuracy: {0:.2%}".format(accuracy))
        # print()
        
        #y_pred_percentage=model.predict(X_test)
        # X_test,y_test 
        
        # matrix=confusion_matrix(y_test, y_pred)
        # print(matrix)
        # class_report=classification_report(y_test, y_pred)
        # print(class_report)

        # Save the model if it improves on the best-found performance.
        # We use the global keyword so we update the variable outside
        # of this function.
        global best_accuracy
        

        # If the classification accuracy of the saved model is improved ...
        if accuracy > best_accuracy:
            # Save the new model to harddisk.
            
            model.load_weights(ck+"best_model_weights.hdf5")# save best model's weights
            model.save(ck+"best_model_so_far.h5") # save the entire model configurartion and weights
            print('**** THE BEST ACCURACY SO FAR IS ACHIEVED AND THE MODEL IS SAVED **** \n')
            
            # save the hyperparameters
            pickle_out = open(ck+"hp.pickle","wb")
            pickle.dump([learning_rate, num_dense_layers,num_dense_nodes,drop_out_rate_input,
                           drop_out_rate_hidden, activation], pickle_out, protocol=2)
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
        
        
    def create_model(learning_rate, num_dense_layers,
                     num_dense_nodes, drop_out_rate_input,
                           drop_out_rate_hidden, activation):
        """
        Hyper-parameters:
        learning_rate:          Learning-rate for the optimizer.
        num_dense_layers:       Number of dense layers.
        num_dense_nodes:        Number of nodes in each dense layer.
        drop_out_rate_input:    Rate of dropout in input layer.
        drop_out_rate_hidden:   Rate of dropout in hidden layer.
        activation:             Activation function for all layers.
        """



        model = Sequential()    

        
        model.add(Dense(num_dense_nodes, input_shape=X_train.shape[1:],activation=activation ,kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dropout(drop_out_rate_input))


        # Flatten the 4-rank output of ckthe convolutional layers
        # to 2-rank that can be input to a fully-connected / dense layer.
        model.add(Flatten())

        # Add fully-connected / dense layers.
        # The number of layers is a hyper-parameter we want to optimize.
        for i in range(num_dense_layers):
            # Name of the layer. This is not really necessary
            # because Keras should give them unique names.
            name = 'layer_dense_{0}'.format(i+1)

            # Add the dense / fully-connected layer to the model.
            # This has two hyper-parameters we want to optimize:
            # The number of nodes and the activation function.
            model.add(Dense(num_dense_nodes,
                            activation=activation,
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
            


    # best_accuracy = 0.0 
    

    ##TEST RUN
    #fitness(x=default_parameters)


    ###Run the Hyper-Parameter Optimization####
    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=35)

    stop = timeit.default_timer()
    #running time
    # print('Time: ', stop - start)
    print ('Runing time is (mins):',round((stop - start)/60,2))
    plot_convergence(search_result)
    plt.savefig(ck+'-plot1.png', format='png')




    
    
if __name__ == '__main__':  
    main()
    # global best_accuracy 
    # for i in range(3,8):
        # main(i)   
    # read_data()
