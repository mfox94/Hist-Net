import pandas as pd
import numpy as np

from glob import glob
from sklearn.model_selection import train_test_split
import fnmatch

import cv2

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

# validation data
_, _, Xval, Yval = train_test_split(X, Y, test_size=.1)
np.save(PATH_OUTPUT + 'images_val', Xval, allow_pickle=True, fix_imports=True)
np.save(PATH_OUTPUT + 'labels_val', Yval, allow_pickle=True, fix_imports=True)
# np.save(PATH_OUTPUT + 'images', X, allow_pickle=True, fix_imports=True)
# np.save(PATH_OUTPUT + 'labels', Y, allow_pickle=True, fix_imports=True)
print('elapsed')
print(round(time.time() - start))
