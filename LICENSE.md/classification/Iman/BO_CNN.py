


import matplotlib.pyplot as plt
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
from sklearn.utils import shuffle

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
    X=X.reshape(num_case*num_pmu,1,num_sample,1)

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


#loading training set
pickle_in = open("/Users/imanniazazari/Desktop/DOE project/X_train_rocof_6.pickle","rb")
X_train = pickle.load(pickle_in)
#X_train=X_train*X_train

pickle_in = open("/Users/imanniazazari/Desktop/DOE project/y_train_rocof_6.pickle","rb")
y_train= pickle.load(pickle_in)


#number of classes
num_classes=len(np.unique(y_train))


##Constants

EPOCHS=3
BATCH_SIZE=16

# set the number of time samples (originallly it is 10 mins of data)
fps=60
start_crop=int(fps*60*4)
stop_crop=int(fps*60*8)

X_train=X_train[:,:,start_crop:stop_crop,:]

#Removing the planned events
X_train,y_train=removePlanned(X_train,y_train)

#split the original data into training/valiadation set (the data is suffled and then is split)  
X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=0.25)

#separate PMUs to make more events
X_train, y_train= separatePMUs(X_train,y_train)
X_train, y_train = shuffle(X_train, y_train)


X_val, y_val= separatePMUs(X_val,y_val)
X_val, y_val = shuffle(X_val, y_val)

# saving training and validation sets
training_set=(X_train, y_train)
pickle_out = open("training_set.pickle","wb")
pickle.dump(training_set, pickle_out, protocol=2)
pickle_out.close() 


validation_set = (X_val, y_val)
pickle_out = open("validation_set.pickle","wb")
pickle.dump(validation_set, pickle_out, protocol=2)
pickle_out.close() 

#make samples positive
#X_train=abs(X_train)

#normalization
'''
scaler=MinMaxScaler()
X_train_tr=scaler.fit_transform(np.transpose(X_train))
X_train=np.transpose(X_train_tr)
'''



##SET UP the RANGE of HPs
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_conv_filters = Integer(low=4, high=128, name='num_conv_filters')
dim_size_conv_filters=Integer(low=4, high=1000, name='size_conv_filters')
dim_num_dense_layers = Integer(low=1, high=6, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=20, high=500, name='num_dense_nodes')
dim_drop_out_input = Real(low=0.4, high=0.9, name='drop_out_rate_input')
dim_drop_out_hidden = Real(low=0.2, high=0.7, name='drop_out_rate_hidden')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dimensions = [dim_learning_rate,
              dim_num_conv_filters,
              dim_size_conv_filters,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_drop_out_input,
              dim_drop_out_hidden,
              dim_activation]

#default_parameters = [1e-5, 1, 16, 'relu']

def log_dir_name(learning_rate,num_conv_filters,
                       size_conv_filters, num_dense_layers,num_dense_nodes,
                drop_out_rate_input, drop_out_rate_hidden,
                 activation):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_conv_filters_{1}_conv_filter_size_{2}_dense_layers_{3}_nodes_{4}_dropout_input_{5}_dropout_hidden_{6}_activation_{7}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_conv_filters,
                       size_conv_filters,
                       num_dense_layers,
                       num_dense_nodes,
                       drop_out_rate_input,
                       drop_out_rate_hidden,
                       activation)

    return log_dir



def create_model(learning_rate, num_conv_filters,
                       size_conv_filters,num_dense_layers,
                 num_dense_nodes, drop_out_rate_input,
                       drop_out_rate_hidden, activation):
    """
    Hyper-parameters:
    learning_rate:          Learning-rate for the optimizer.
    num_conv_filters:       Number of conv layer filters
    size_conv_filters:      Size of filter
    num_dense_layers:       Number of dense layers.
    num_dense_nodes:        Number of nodes in each dense layer.
    drop_out_rate_input:    Rate of dropout in input layer.
    drop_out_rate_hidden:   Rate of dropout in hidden layer.
    activation:             Activation function for all layers.
    """

    model = Sequential()   
    model.add(Conv2D(num_conv_filters, (1, size_conv_filters), padding="same", input_shape=X_train.shape[1:],
                     activation=activation, name='layer_conv1',
                     kernel_regularizer=regularizers.l2(0.0001)))
    model.add(MaxPooling2D(pool_size=(1,2), strides=1,padding="same"))
    model.add(Dropout(drop_out_rate_input))
    
    
    model.add(Conv2D(36, (3, 3), padding="same", input_shape=X_train.shape[1:],
                     activation=activation, name='layer_conv2',
                     kernel_regularizer=regularizers.l2(0.0001)))
    model.add(MaxPooling2D(pool_size=2, strides=2,padding="same"))
    model.add(Dropout(drop_out_rate_hidden)) 



    # Flatten the 4-rank output of the convolutional layers
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


####Train and Evaluate the Model###
path_best_model = 'best_model.hdf5'
best_accuracy = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate,num_conv_filters,
                       size_conv_filters, num_dense_layers,
            num_dense_nodes, drop_out_rate_input,
                       drop_out_rate_hidden, activation):
    """
    Hyper-parameters:
    learning_rate:          Learning-rate for the optimizer.
    num_conv_filters:       Number of conv layer filters
    size_conv_filters:      Size of filter
    num_dense_layers:       Number of dense layers.
    num_dense_nodes:        Number of nodes in each dense layer.
    drop_out_rate_input:    Rate of dropout in input layer.
    drop_out_rate_hidden:   Rate of dropout in hidden layer.
    activation:             Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('*************************** NEW ITERATION IS STARTED****************************')
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_conv_filters:', num_conv_filters)
    print('size_filter:', size_conv_filters)
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('drop_out_rate_input:', drop_out_rate_input)
    print('drop_out_rate_hidden:', drop_out_rate_hidden)
    print('activation:', activation)
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_conv_filters=num_conv_filters,
                         size_conv_filters=size_conv_filters,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         drop_out_rate_input=drop_out_rate_input,
                         drop_out_rate_hidden=drop_out_rate_hidden,
                         activation=activation)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate,num_conv_filters,
                       size_conv_filters, num_dense_layers,
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
    
    file_name="best_model_weights.hdf5"

    mcp_save = ModelCheckpoint(file_name, save_best_only=True, monitor='val_accuracy', mode='max')

   
    # Use Keras to train the model.
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=EPOCHS,
                        batch_size=16,
                        verbose=2,
                        validation_data=validation_set,
                        callbacks=[callback_log,mcp_save])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    #accuracy = history.history['val_accuracy'][-1] #last epoch accuracy in each call
    accuracy = max(history.history['val_accuracy']) #best accuracy among all epochs in each call


    

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy


    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        
        model.load_weights("best_model_weights.hdf5")# save best model's weights
        model.save("best_model_so_far.h5") # save the entire model configurartion and weights
        print('**** THE BEST ACCURACY SO FAR IS ACHIEVED AND THE MODEL IS SAVED **** \n')
        
        # save the hyperparameters
        pickle_out = open("hp.pickle","wb")
        pickle.dump([learning_rate, num_conv_filters,
                       size_conv_filters,num_dense_layers,num_dense_nodes,drop_out_rate_input,
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

##TEST RUN
#fitness(x=default_parameters)


###Run the Hyper-Parameter Optimization####
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40)

stop = timeit.default_timer()
#running time
print('Time: ', stop - start)

plot_convergence(search_result)
plt.save('plt.png')
