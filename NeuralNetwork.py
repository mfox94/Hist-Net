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
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import gc

PATH_INPUT = "../Preprocessed_Data/"
#PATH_INPUT = "gdrive/My Drive/"
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
    images_temp = np.ndarray(shape=(len(images), 50, 50))
    for i in range(images.shape[0]):
        curr_im = images[i].astype('uint8')
        curr_im = filters.sobel(color.rgb2gray(images[i]))
        images_temp[i] = curr_im
    return images_temp


def Applygrey(images):
    images_temp = np.ndarray(shape=(len(images), 50, 50))
    for i in range(images.shape[0]):
        curr_im = images[i].astype('uint8')
        curr_im = color.rgb2gray(images[i])
        images_temp[i] = curr_im
    return images_temp


def Applyroberts(images):
    images_temp = np.ndarray(shape=(len(images), 50, 50))
    for i in range(images.shape[0]):
        curr_im = images[i].astype('uint8')

        curr_im = filters.roberts(color.rgb2gray(images[i]))
        images_temp[i] = curr_im
    return images_temp


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
        vertical_flip=True
    )
    return model, datagen


def Hist_Net(input_shape):
    num_classes = 2
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


def fit_CNN(Images_Train, Images_Test, Labels_Train, Labels_Test, model, datagen, callbacks_list, input_shape):
    Images_Train = Images_Train.reshape(len(Images_Train), input_shape[0], input_shape[1], input_shape[2])
    Images_Test = Images_Test.reshape(len(Images_Test), input_shape[0], input_shape[1], input_shape[2])

    exec_model = model.fit_generator(datagen.flow(Images_Train, Labels_Train, batch_size=batch_size),
                                     steps_per_epoch=len(Images_Train) / batch_size,
                                     epochs=epochs, validation_data=[Images_Test, Labels_Test],
                                     callbacks=callbacks_list
                                     )
    return exec_model


def statistics(a, b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of IDC(-) Images: {}'.format(np.sum(b == 0)))
    print('Number of IDC(+) Images: {}'.format(np.sum(b == 1)))
    print('Percentage of positive images: {:.2f}%'.format(100 * np.mean(b)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))

def Train_CNN(cnn,filter,iteration):
    for i in range(iteration):
        print('ITERATION {}'.format(i + 1))

        print("Initialize training")
        print("\tLoading images")
        images = np.load(PATH_INPUT + 'images.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                         encoding='ASCII')
        print("\tLoading labels")
        labels = np.load(PATH_INPUT + 'labels.npy', mmap_mode=None, allow_pickle=True, fix_imports=True,
                         encoding='ASCII')
        print("\tSplitting 80%/20%")
        Images_Train, Images_Test, Labels_Train, Labels_Test = train_test_split(images, labels, test_size=0.2,
                                                                                random_state=42, stratify=labels)
        statistics(Images_Train, Labels_Train)
        Labels_Train = to_categorical(Labels_Train, num_classes=2)
        Labels_Test = to_categorical(Labels_Test, num_classes=2)

        del images
        del labels

        print('Applying the selected filter')
        input_shape = (50, 50, 3)
        if filter == 'N':
            Images_Train = Images_Train
            input_shape = (50, 50, 3)
        if filter == 'G':
            Images_Train = Applygrey(Images_Train)
            Images_Test = Applygrey(Images_Test)
            input_shape = (50, 50, 1)
        if filter == 'S':
            Images_Train = Applysobel(Images_Train)
            Images_Test = Applysobel(Images_Test)
            input_shape = (50, 50, 1)

        print('Compiling the selected CNN')
        model, datagen = None, None
        if cnn == 'B':
            model, datagen = Baseline_Net(input_shape)
        if cnn == 'H':
            model, datagen = Hist_Net(input_shape)

        checkpoint = ModelCheckpoint('Model'+cnn+'_'+filter+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        print('Fitting the selected CNN')
        model = fit_CNN(Images_Train, Images_Test, Labels_Train, Labels_Test, model, datagen, callbacks_list, input_shape)
        printAcc(model)
        del Images_Train
        del Images_Test
        del Labels_Train
        del Labels_Test
        del model
        del datagen
        del callbacks_list
        gc.collect()
        print("COMPLETE!_______________")

Train_CNN('B','N',3)