import numpy as np
from tkinter import *
import tkinter.ttk as ttk
import numpy as np
import matplotlib.pylab as plt
from skimage import color
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, \
    GridSearchCV
import keras
import cv2
from skimage import filters
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, \
    AveragePooling2D
from keras.utils.np_utils import to_categorical
from PIL import Image, ImageTk
from tkinter import Tk, BOTH
from tkinter.ttk import Frame, Label, Style
from tkinter.filedialog import askopenfilename

MODEL_PATH = '../Models/'
PATH_INPUT = "../Preprocessed_Data/"


def initialize_data():
    print("\tLoading IMAGES")
    images = np.load(PATH_INPUT + 'images_val.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                     encoding='ASCII')
    print("\tLoading LABELS")
    labels = np.load(PATH_INPUT + 'labels_val.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                     encoding='ASCII')
    return images, labels


def Baseline_Net(input_shape):
    num_classes = 2

    strides = 2
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, strides=strides))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
    return model, datagen


def Hist_Net(num_classes, input_shape, strides):
    num_classes = 2
    input_shape = (50, 50)
    strides = 2
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
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True)
    return model, datagen


def load_model(images, cnn, filter):
    if cnn == 'B' and filter == 'N':
        input_shape = (50, 50, 3)
        model, datagen = Baseline_Net(input_shape)
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        model.load_weights(MODEL_PATH + 'BaseLine_Nofilter.h5')
        predictions = model.predict(images.reshape(1,50,50,3), steps=1, verbose=1)
    return predictions


def proc_images(img):
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    full_size_image = cv2.imread(img)
    img = cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    return img


# images, labels = initialize_data()
pathimg = '..\\Data\\8984\\1\\8984_idx5_x2001_y1601_class1.png'
img = proc_images(pathimg)
img=img/255
pred = load_model(img, 'B', 'N')
