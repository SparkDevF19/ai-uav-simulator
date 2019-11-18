'''This is not our work, we are simpliy modifying the CNN
from https://pythonprogramming.net/convolutional-neural-network-cnn-machine-learning-tutorial/
by: Harrison Kinsley'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "" #Whater the directory to SAFE and UNSAFE images
STATES = ["AirSim_safe", "Airsim_Unsafe"] #sub directories

#function to convert to array for our visulization function
def dataToAR():

    for state in STATES:
        path = os.path.join(DATADIR,state)  #iterate through the states to train the model
        for img in os.listdir(path):  # iterate over each image
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            '''plt.imshow(img_array, cmap='gray')  # graph it
            plt.show()  # display!'''
            break  # we just want one for now so break
        break  #...and one more!
    ImgAttributes(img_array)

#Helper function to help us visualize image attributes
def ImgAttributes(image):
    print(img_array)
    [[189 189 189 ...  29  29  31]
    [186 186 186 ...  36  35  36]
    [184 185 185 ...  35  33  33]
    ...
    [168 169 170 ...  71  72  72]
    [169 170 171 ...  68  67  67]
    [168 169 170 ...  64  63  62]]

    print(img_array.shape)
    (398, 500)

    IMG_SIZE = 50

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()

    training_data = []

def create_training_data():
    for state in STATES:
        path = os.path.join(DATADIR,state)  # create path to STATES[state]
        class_num = STATES.index(state)  # get the classification  (0 or a 1).

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

def main():
    create_training_data()

    print(len(training_data))

    import random

    random.shuffle(training_data)

    for sample in training_data[:10]:
        print(sample[1])

    X = []
    y = []

    for features,label in training_data:
        X.append(features)
        y.append(label)

    print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    '''pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)'''
