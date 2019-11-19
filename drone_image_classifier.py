"""
IMAGE CLASSIFIER, DATAFRAME BUILDER & CNN Network
--------------------------------------------------------------------------------------------------------------
DESCRIPTION:

--------------------------------------------------------------------------------------------------------------    
MADE BY: 
    Ivan A. Reyes
    Ernest J. Quant
    Orson Meyreles
    John Quitto-Graham  
    Carlos Valdes
    Catherine Angelini
    Marcial Barrios 
--------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import tensorflow.keras as keras 
from tqdm import tqdm

#Directory variables
DATADIR = r".\\git_repos\\ai-uav-simulator" # The directory to SAFE and UNSAFE PFM/PNG images
STATES_PNG = ["Airsim_SafePNG", "Airsim_UnsafePNG"] #Sub-directories for png files
SAFE_IMG_PNG = STATES_PNG[0] 
UNSAFE_IMG_PNG = STATES_PNG[1]


#function to convert to array for our visulization function
def dataToAR():

    for state in STATES_PNG:
        path_PNG = os.path.join(DATADIR,state)  #iterate through the STATES_PNG to train the model

        for img in os.listdir(path_PNG):  # iterate over each image
            img_ArrayPNG = cv2.imread(os.path.join(path_PNG,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            break  # we just want one for now so break
        break  #...and one more!

    
        ImgAttributes(img_ArrayPNG)

#Helper function to help us visualize image attributes
def ImgAttributes(image_array):
    print(image_array)

    print(image_array.shape)

    IMG_SIZE = 50

    new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()

    new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()

def create_training_data():
    
    training_data = []

    for state in STATES_PNG:
        path = os.path.join(DATADIR,state)  # create path to STATES_PNG[state]
        class_num = STATES_PNG.index(state)  # get the classification  (0 or a 1).

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (64, 64))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    def CNN_trainer(input_df):
        X = input_df.iloc[:,0]
        y = input_df.iloc[:,1]

        pickle_out = open("X.pickle", "wb")
        pickle.dump(X,pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", "wb")
        pickle.dump(y,pickle_out)
        pickle_out.close()

        pickle_in = open("X.pickle", "rb")
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle", "rb")
        y = pickle.load(pickle_in)

        X = X/255.0 #normalize

        dense_layers = [0]
        layer_sizes = [64]
        conv_layers = [1]

        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                    print(NAME)
                    model = Sequential() #feed-forward network
                    model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model.compile(loss='binary_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'],
                                )

                model.fit(X, y,
                            batch_size=32,
                            epochs=10,
                            validation_split=0.1,
                            callbacks=[tensorboard])

        model.save('CNN_tester.model')

def buildDF():
    dataToAR()

    #appends safe & unsafe images to the appropriate lists to later use as labels
    imageSafeList = pd.Series()
    imageUnsafeList = pd.Series()
    images_df = pd.DataFrame(columns=["image_Data", "State"])
    
    
    for file in os.listdir(DATADIR + '\\' + SAFE_IMG_PNG):
        try:
            
            frame = cv2.imread(DATADIR + '\\' + SAFE_IMG_PNG + '\\' + file)
            frame = cv2.resize(frame, (50, 50))
            images_df.loc[file] = [frame, "Safe"]
    
        except Exception as e:
            continue

    for file in os.listdir(DATADIR + '\\' + UNSAFE_IMG_PNG):
        try:
            frame = cv2.imread(DATADIR + '\\' + UNSAFE_IMG_PNG + '\\' + file)
            frame = cv2.resize(frame, (50, 50))
            images_df.loc[file] = [frame, "Unsafe"]

        except Exception as e:
            continue

    return images_df

def main():
    #Here we are building a dataframe of our images and their labels
    images_df = buildDF()
    #invoke the network
    CNN_trainer(images_df)

# Where we'll store weights and biases
PARAMFILE = 'params.pkl'

'''
CNN adapter from:
from https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
by: Harrison Kinsley
'''