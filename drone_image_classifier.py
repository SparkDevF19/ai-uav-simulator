'''This is not our work, we are simpliy modifying the CNN
from https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
by: Harrison Kinsley'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import tensorflow.keras as keras 
from tqdm import tqdm

#Directory variables
DATADIR = "D:\\" # The directory to SAFE and UNSAFE PFM/PNG images
STATES_PNG = ["Airsim_SafePNG", "Airsim_UnsafePNG"] #Sub-directories for png files
STATES_PFM = ["Airsim_SafePFM", "Airsim_UnsafePFM"] #Sub-directories for pfm files
SAFE_IMG_PNG = STATES_PNG[0]
SAFE_IMG_PFM = STATES_PFM[0] 
UNSAFE_IMG_PNG = STATES_PNG[1]
UNSAFE_IMG_PFM = STATES_PFM[1]

#Check if we are taking in a png or a pfm file
def image_Type(file):
    return file.endswith(".png")


#function to convert to array for our visulization function
def dataToAR():

    for state in STATES_PNG:
        path_PNG = os.path.join(DATADIR,state)  #iterate through the STATES_PNG to train the model

        for img in os.listdir(path_PNG):  # iterate over each image
            img_ArrayPNG = cv2.imread(os.path.join(path_PNG,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            '''plt.imshow(img_array, cmap='gray')  # graph it
            plt.show()  # display!'''
            break  # we just want one for now so break
        break  #...and one more!

    for state in STATES_PFM:
        path_PFM = os.path.join(DATADIR, state)

        for img in os.listdir(path_PFM):
            img_ArrayPFM= cv2.imread(os.path.join(path_PFM, img) ,cv2.IMREAD_GRAYSCALE)
            break
        break

    ImgAttributes(img_ArrayPNG)
    ImgAttributes(img_ArrayPFM)

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

    def CNN_trainer(IMAGE):
        pickle_in = open("X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb")
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

def main():
    dataToAR()

    #store safe/unsafe images into a list
    imageSafeList = []
    imageUnsafeList = []

    #appends safe & unsafe images to the appropriate lists to later use as labels
    if (image_Type):
        for file in os.listdir(DATADIR+SAFE_IMG_PNG):
                frame = loadgray(DATADIR + SAFE_IMG_PNG + file)
                imageSafeList.append(file)
        for file in os.listdir(DATADIR+UNSAFE_IMG_PNG):
                frame = loadgray(DATADIR + UNSAFE_IMG_PNG + file)
                imageUnsafeList.append(file)
    elif(not(image_Type)):
        for file in os.listdir(DATADIR+SAFE_IMG_PFM):
            frame = loadgray(DATADIR + SAFE_IMG_PFM+ file)
            imageSafeList.append(file)
        for file in os.listdir(DATADIR + UNSAFE_IMG_PFM):
            frame = loadgray(DATADIR + UNSAFE_IMG_PFM + file)
            imageUnsafeList.append(file)

    #store unsafe images into a list
    

    #final list that stores the images
    images = imageSafeList.append(imageUnsafeList)

    labels = []
    for i in range(len(imageSafeList)):
        labels.append("safe")
    for u in range(len(imageUnsafeList)):
        labels.append("unsafe")

    retPairs = feed_dict = {images, labels}
    print(retPairs)

main()
# Where we'll store weights and biases
PARAMFILE = 'params.pkl'