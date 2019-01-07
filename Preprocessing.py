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
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from os import listdir
import gc
import time as time


def proc_images(lowerIndex, upperIndex):
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    i = 0
    for img in allImages[lowerIndex:upperIndex]:
        i = i + 1
        if i % 5000 == 0: print("Processing {}th image".format(i))
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x, y


def statistics(a, b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of Negative Images: {}'.format(np.sum(b == 0)))
    print('Number of Positive Images: {}'.format(np.sum(b == 1)))
    print('Percentage of positive images: {:.2f}%'.format(100 * np.mean(b)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))


PATH_INPUT = "../Data/"
PATH_OUTPUT = "../Preprocessed_Data/"

start = time.time()
allImages = glob(PATH_INPUT + '**/*.png', recursive=True)

# class0 := negative
# class1 := positive

negative_class_pattern = '*class0.png'
positive_class_pattern = '*class1.png'
classZero = fnmatch.filter(allImages, negative_class_pattern)
classOne = fnmatch.filter(allImages, positive_class_pattern)
X, Y = proc_images(0, 90000)
X1 = np.array(X)

df = pd.DataFrame()
df["images"] = X
df["labels"] = Y

X2 = df["images"]
Y2 = df["labels"]

X2 = np.array(X2)

imgs0 = []
imgs1 = []
imgs0 = X2[Y2 == 0]  # (0 = no IDC, 1 = IDC)
imgs1 = X2[Y2 == 1]
statistics(X2, Y2)

X = np.array(X)
X = X / 255.0

np.save(PATH_OUTPUT+'images', X, allow_pickle=True, fix_imports=True)
np.save(PATH_OUTPUT+'labels', Y, allow_pickle=True, fix_imports=True)
print('elapsed')
print(round(time.time() - start))
