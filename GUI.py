import numpy as np
from tkinter import *
import tkinter.ttk as ttk

from PIL import Image, ImageTk
from tkinter import Tk, BOTH
from tkinter.ttk import Frame, Label, Style
from tkinter.filedialog import askopenfilename


class GUI(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.master.title("Simple")
        self.pack(fill=BOTH, expand=1)

        self.master.title("Absolute positioning")
        self.pack(fill=BOTH, expand=1)

        Style().configure("TFrame", background="#333")

        Lab = Label(self, text="SELECT CNN", background="red", foreground="white", font=("Helvetica", 16),
                    justify=CENTER).pack(fill=X)

        CNN_Choice = IntVar()
        R1 = Radiobutton(self, text="BASELINE", value=0, var=CNN_Choice, pady=10, background="red").pack(fill=X)
        R2 = Radiobutton(self, text="HIST-NET", value=1, var=CNN_Choice, pady=10, background="red").pack(fill=X)

        Lab = Label(self, text="SELECT FILTER", background="blue", foreground="white", font=("Helvetica", 16)).pack(
            fill=X)
        Filter_Choice = IntVar()
        R3 = Radiobutton(self, text="NO FILTER", value=0, var=Filter_Choice, pady=10, background="blue").pack(fill=X)
        R4 = Radiobutton(self, text="GREYSCALE", value=1, var=Filter_Choice, pady=10, background="blue").pack(fill=X)
        R5 = Radiobutton(self, text="SOBEL", value=2, var=Filter_Choice, pady=10, background="blue").pack(fill=X)
        filename = askopenfilename(filetypes=[("Images", "*.png")], initialdir="../Data")
        load_image(self, filename)
        load_button(self)


def load_image(self, filename):
    image = Image.open(filename)
    image=image.resize((90, 90), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label1 = Label(self, image=image)
    label1.image = image
    label1.pack(fill=X)


def load_button(self):
    okButton = Button(self, text="EVALUATE", width=500, height=10)
    okButton.pack(side=LEFT)


def main():
    print("GUI")
    root = Tk()
    root.geometry("500x500+300+300")
    app = GUI()
    root.mainloop()


if __name__ == '__main__':
    main()
