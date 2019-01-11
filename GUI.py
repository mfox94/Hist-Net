from tkinter import *
from tkinter import Tk, BOTH
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Frame, Label, Style

import cv2
import keras
import numpy as np
from PIL import Image, ImageTk
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
    AveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from skimage import color
from skimage import filters

MODEL_PATH = '../Models/'
filename = ''
pred = None
prob = None
used_cnn = ''
used_filter = ''


def set_B():
    global used_cnn
    used_cnn = 'B'


def set_H():
    global used_cnn
    used_cnn = 'H'


def set_N():
    global used_filter
    used_filter = 'N'


def set_G():
    global used_filter
    used_filter = 'G'


def set_S():
    global used_filter
    used_filter = 'S'

def predict_lab():
    global used_cnn
    global used_filter
    if used_cnn == '':
        used_cnn = 'B'
    if used_filter == '':
        used_filter = 'N'
    if filename == '':
        print("YOU MUST SELECT AN IMAGE, PLEASE RESTART THE APPLICATION...")
    load_model()
    probab = prob
    if pred == 'Healthy':
        Label(text=pred + ' with a probability of the ' + str(round(prob * 100, 2)) + '%', background="#6aa83c",
              foreground="white", font=("Helvetica", 14)).pack(
            fill=X)
    else:
        Label(text=pred + ' with a probability of the ' + str(round(prob * 100, 2)) + '%', background="#a50000",
              foreground="white", font=("Helvetica", 14)).pack(
            fill=X)


class GUI(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.master.title("Simple")
        self.pack(fill=BOTH, expand=1)

        self.master.title("EVVIVA!")
        self.pack(fill=BOTH, expand=1)

        Style().configure("TFrame", background="#193446")
        loadImageButton = Button(self, text="LOAD IMAGE", width=100, background='#e9c77b', command=load_image)
        loadImageButton.pack(side=TOP)
        Lab = Label(self, text="SELECT CNN", background="#c4d4e0", foreground="#193446", font=("Helvetica", 16),
                    justify=CENTER).pack(fill=X)

        CNN_Choice = StringVar()
        B = Radiobutton(self, text="BASELINE", value='B', var=CNN_Choice, pady=10, background="#c4d4e0",
                        command=set_B).pack(fill=X)
        H = Radiobutton(self, text="HIST-NET", value='H', var=CNN_Choice, pady=10, background="#c4d4e0",
                        command=set_H).pack(fill=X)

        Lab = Label(self, text="SELECT FILTER", background="#9aabb9", foreground="#193446",
                    font=("Helvetica", 16)).pack(
            fill=X)
        Filter_Choice = StringVar()
        N = Radiobutton(self, text="NO FILTER", value='N', var=Filter_Choice, pady=10, background="#9aabb9",
                        command=set_N).pack(
            fill=X)
        G = Radiobutton(self, text="GREYSCALE", value='G', var=Filter_Choice, pady=10, background="#9aabb9",
                        command=set_G).pack(
            fill=X)
        S = Radiobutton(self, text="SOBEL", value='S', var=Filter_Choice, pady=10, background="#9aabb9",
                        command=set_S).pack(fill=X)
        okButton = Button(self, text="EVALUATE", width=100, background='#e9c77b', command=predict_lab).pack()

        CNN_Choice.set('B')
        Filter_Choice.set('N')





def load_image():
    global filename
    fm = askopenfilename(filetypes=[("Images", "*.png")], initialdir="../Data")
    image = Image.open(fm)
    image = image.resize((150, 150), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label1 = Label(image=image, anchor=CENTER, background='#193446', width=500)
    label1.image = image
    label1.pack(fill=X, side=TOP)

    filename = fm


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


def proc_images(img):
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    full_size_image = cv2.imread(img)
    img = cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    return img


def Applysobel(im):
    curr_im = im.astype('uint8')
    curr_im = filters.sobel(color.rgb2gray(im))
    return curr_im


def Applygrey(im):
    curr_im = im.astype('uint8')
    curr_im = filters.roberts(color.rgb2gray(im))
    return curr_im


def load_model():
    global pred
    global prob
    print(used_filter, used_cnn)

    if used_cnn == 'B' and used_filter == 'N':
        input_shape = (50, 50, 3)
        model, _ = Baseline_Net(input_shape)
        model.load_weights(MODEL_PATH + 'BaseLine_Nofilter.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        image = proc_images(filename) / 255
        predictions = model.predict(image.reshape(1, 50, 50, 3), verbose=0)
    if used_cnn == 'B' and used_filter == 'G':
        input_shape = (50, 50, 1)
        model, _ = Baseline_Net(input_shape)
        model.load_weights(MODEL_PATH + 'BaseLine_Grey.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        image = proc_images(filename)
        image = Applygrey(image)
        predictions = model.predict(image.reshape(1, 50, 50, 1), verbose=0)
    if used_cnn == 'B' and used_filter == 'S':
        input_shape = (50, 50, 1)
        model, _ = Baseline_Net(input_shape)
        model.load_weights(MODEL_PATH + 'BaseLine_Sobel.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        image = proc_images(filename)
        image = Applysobel(image)
        predictions = model.predict(image.reshape(1, 50, 50, 1), verbose=0)

    if used_cnn == 'H' and used_filter == 'N':
        input_shape = (50, 50, 3)
        model, _ = Hist_Net(input_shape)
        model.load_weights(MODEL_PATH + 'Hist-Net_Nofilter.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        image = proc_images(filename) / 255
        predictions = model.predict(image.reshape(1, 50, 50, 3), verbose=0)
    if used_cnn == 'H' and used_filter == 'G':
        input_shape = (50, 50, 1)
        model, _ = Hist_Net(input_shape)
        model.load_weights(MODEL_PATH + 'Hist-Net_Grey.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        image = proc_images(filename)
        image = Applygrey(image)
        predictions = model.predict(image.reshape(1, 50, 50, 1), verbose=0)
    if used_cnn == 'H' and used_filter == 'S':
        input_shape = (50, 50, 1)
        model, _ = Hist_Net(input_shape)
        model.load_weights(MODEL_PATH + 'Hist-Net_Sobel.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        image = proc_images(filename)
        image = Applysobel(image)
        predictions = model.predict(image.reshape(1, 50, 50, 1), verbose=0)

    if predictions[0][0] > predictions[0][1]:
        pred = 'Healthy'
        prob = predictions[0][0]
    else:
        pred = 'Affected'
        prob = predictions[0][1]
    return predictions


print("GUI")
root = Tk()
root.geometry("500x500+300+300")
app = GUI()
root.mainloop()
