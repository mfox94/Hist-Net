import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
from scipy.misc import imresize, imread
import sklearn
from skimage import color
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, \
    GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from sklearn import datasets, svm, metrics
from skimage import filters
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, \
    AveragePooling2D
from os import listdir
import gc
import time as time
from keras.utils.np_utils import to_categorical
import time as time

# Colab paths...
#  PATH_INPUT = "../Data/"

# Local paths..
PATH_INPUT = "../Preprocessed_Data/"


def printAcc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def ApplyFilter(images, filter):
    if filter == 'GREY':
        Applygrey(images)
    if filter == 'SOBEL':
        Applysobel(images)
    if filter == 'ROBERTS':
        Applysobel(images)
    if filter != 'GREY' | filter != 'GREY' | filter != 'GREY':
        print("ERROR: unrecognized filter")


def Applysobel(images):
    images_temp = np.ndarray(shape=(90000, 50, 50))
    for i in range(images.shape[0]):
        curr_im = images[i].astype('uint8')
        curr_im = filters.sobel(color.rgb2gray(images[i]))
        images_temp[i] = curr_im
    return images_temp


def Applygrey(images):
    images_temp = np.ndarray(shape=(90000, 50, 50))
    for i in range(images.shape[0]):
        curr_im = images[i].astype('uint8')
        curr_im = color.rgb2gray(images[i])
        images_temp[i] = curr_im
    return images_temp


def Applyroberts(images):
    images_temp = np.ndarray(shape=(90000, 50, 50))
    for i in range(images.shape[0]):
        curr_im = images[i].astype('uint8')

        curr_im = filters.roberts(color.rgb2gray(images[i]))
        images_temp[i] = curr_im
    return images_temp


def initialize():
    print("loading images...")
    images = np.load(PATH_INPUT + 'images.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                     encoding='ASCII')
    print("loading labels...")
    labels = np.load(PATH_INPUT + 'labels.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                     encoding='ASCII')
    print("splitting..")
    Images_Train, Images_test, Labels_Train, Labels_Test = train_test_split(images, labels, test_size=0.2)
    print("to categorical Train")
    y_trainHot = to_categorical(Labels_Train, num_classes=2)
    print("to categorical Test")
    y_testHot = to_categorical(Labels_Test, num_classes=2)
    print("splitting completed!")
    return Images_Train, Images_test, y_trainHot, y_testHot
    del images
    del labels


def Hist_Net(num_classes, input_shape, strides):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, strides=strides))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(
        rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    return model, datagen


def Hist_Net_Initialize(Images_Train, Images_test, y_trainHot, y_testHot, n_channel):
    num_classes = 2
    img_rows, img_cols = 50, 50
    input_shape = (img_rows, img_cols, n_channel)
    strides = 2
    model, datagen = Hist_Net(num_classes, input_shape, strides)


def main():
    print("Initialization")
    Images_Train, Images_test, y_trainHot, y_testHot = initialize()
    num_channels = Images_Train.shape[3]  # check the channels of each image
    Hist_Net_Initialize(Images_Train, Images_test, y_trainHot, y_testHot, num_channels)

    print("COMPLETE!_______________")


# print("MODEL HISTNET:")
# batch_size = 1024
# num_classes = 2
# epochs = 8
# img_rows,img_cols=50,50
# input_shape = (img_rows, img_cols, 3)
# #input_shape = (img_rows, img_cols, 1)
# e = 2
#
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape,strides=e))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(128, (5, 5), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# a = Images_Train
# #a = Images_Train.reshape(Images_Train.shape[0], 50, 50, 1)
# b = y_trainHot
# c = Images_test
# #c = Images_test.reshape(Images_test.shape[0], 50, 50, 1)
# d = y_testHot
# epochs = 10
#
# print("Model 1 compiled!")
# time1= time.time()
#
# datagen = ImageDataGenerator(
#         rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=True)  # randomly flip images
#
#
# exec_model = model.fit_generator(datagen.flow(a,b, batch_size=16),
#                         steps_per_epoch=len(a) / 16,
#                               epochs=epochs,validation_data = [c, d])
#
# print("execution succeded in ")
# print(time.time()-time1)
# printAcc(exec_model)

if __name__ == '__main__':
    main()
