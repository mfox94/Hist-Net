import numpy as np

import matplotlib.pylab as plt

from skimage import color

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, \
    GridSearchCV

import keras

from skimage import filters
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, \
    AveragePooling2D

from keras.utils.np_utils import to_categorical

# Colab paths...
PATH_INPUT = "gdrive/My Drive/Colab Notebooks/"

# Local paths..

PATH_INPUT = "../Preprocessed_Data/"
epochs = 10
batch_size = 32


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
        images = Applygrey(images)
    if filter == 'SOBEL':
        images = Applysobel(images)
    if filter == 'ROBERTS':
        images = Applysobel(images)
    return images


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


#
# def initialize():
#     print("splitting..")
#     Images_Train, Images_test, Labels_Train, Labels_Test = train_test_split(images, labels, test_size=0.2)
#     print("to categorical Train")
#     y_trainHot = to_categorical(Labels_Train, num_classes=2)
#     print("to categorical Test")
#     y_testHot = to_categorical(Labels_Test, num_classes=2)
#     print("splitting completed!")
#     return Images_Train, Images_test, y_trainHot, y_testHot
#     del images
#     del labels


def Baseline_Net():
    num_classes = 2
    input_shape = (50, 50)
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


def fit_CNN(Images_Train, Images_test, y_trainHot, y_testHot, model, datagen):
    a = Images_Train
    b = y_trainHot
    c = Images_test
    d = y_testHot
    exec_model = model.fit_generator(datagen.flow(a, b, batch_size=batch_size),
                                     steps_per_epoch=len(a) / batch_size,
                                     epochs=epochs, validation_data=[c, d])
    return exec_model


def statistics(a, b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of Negative Images: {}'.format(np.sum(b == 0)))
    print('Number of Positive Images: {}'.format(np.sum(b == 1)))
    print('Percentage of positive images: {:.2f}%'.format(100 * np.mean(b)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))


def Train_Cnn(cnn, filter):
    print("Initialize training")
    images = np.load(PATH_INPUT + 'images.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                     encoding='ASCII')
    labels = np.load(PATH_INPUT + 'labels.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                     encoding='ASCII')
    if filter == 'N':
        images = images
    if filter == 'G':
        images = Applygrey(images)
    if filter == 'S':
        images = Applysobel(images)

    n_folds = 5
    skf = StratifiedKFold(n_folds=5, shuffle=True)
    skf.get_n_splits(images, labels)
    for i, (train, test) in enumerate(skf):
        print("Running Fold" + (i + 1) + "/" + n_folds)

        # CREATE MODEL
        Images_Train, Images_Test = images[train], images[test]
        Labels_Train, Labels_Test = labels[train], labels[test]

        model, datagen = None
        if cnn == 'B':
            model, datagen = Baseline_Net()
        if cnn == 'H':
            model, datagen = Hist_Net()


    # model = fit_CNN(Images_Train, Images_Test, Labels_Train, Labels_Test, model, datagen)
    # printAcc(model)

    print("COMPLETE!_______________")
    outfile = open(PATH_INPUT + 'baseline_sobel.dat', 'wb')
    pickle.dump(model, outfile)
    outfile.close()


if __name__ == '__main__':
    Train_Cnn(0,'N')
