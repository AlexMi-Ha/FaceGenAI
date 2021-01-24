import numpy as np
from keras.layers import Input, Dense
from keras.models import model_from_json

import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.externals import joblib

from PIL import Image, ImageTk

import datetime

# Hier wird das User Interface zum generieren von Gesichtern erzeugt und gemanaged

# Ordner wo das zu  testende Model ist
MODEL_FOLDER = "models\\pca_5.8.2020\\"
# Welche epoche des Models soll geladen werden?
MODEL_ITER = 120

# Laden des Models. Sowohl Convolutional Encoder als auch Decoder


def loadModel(encoder_path, decoder_path, iter=0):
    json_file = open(f'{encoder_path}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _encoder_model = model_from_json(loaded_model_json)
    _encoder_model.load_weights(f"{encoder_path}.h5")

    json_file = open(f'{decoder_path}{iter}_modelDecoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _decoder_model = model_from_json(loaded_model_json)
    _decoder_model.load_weights(f"{decoder_path}{iter}_modelDecoder.h5")

    return (_encoder_model, _decoder_model)


def load_PCA():
    print("Loading PCA...")

    sc = joblib.load(f"{MODEL_FOLDER}\\pca\\scalar.save")
    pca = joblib.load(f"{MODEL_FOLDER}\\pca\\pca.save")

    return (sc, pca)


# Minima und Maxima der einzenlen Principle Compnents suchen
def load_PCA_minmax():
    print("Loading min and max values...")
    max_val = np.load(f"{MODEL_FOLDER}\\pca\\max.npy")
    min_val = np.load(f"{MODEL_FOLDER}\\pca\\min.npy")
    return (min_val, max_val)


# Model data einer Training session laden
def get_model_data():
    file = open(f"{MODEL_FOLDER}model_info.data", "r")
    args = file.readlines()
    return (args[0][args[0].index(': ') + 2:], args[5][args[5].index('(') + 1: -2])


# -----------------------------------------------------------------

# Random seed immer auf 0 setzen um gleiche verhaltungsmuster zu erhalten
np.random.seed(0)

# Bottleneck und Convolutional Encoder pfad aus den Model daten laden
bottleneck, CONVOLUTION_PATH = get_model_data()
bottleneck = int(bottleneck)

# Convolutional Encoder und Decoder laden
_encoder, _decoder = loadModel(encoder_path=CONVOLUTION_PATH,
                               decoder_path=f"{MODEL_FOLDER}models\\", iter=MODEL_ITER)


sc, pca = load_PCA()
pca_min, pca_max = load_PCA_minmax()

# ----------------------------------------------

# Fenster erstellen

root = Tk()

root.geometry("1200x800")
root.title("Face Generator")
canvas = Canvas(root, bg='white')

backSlider = Frame(root, bg='black')

scroll = Scrollbar(canvas, orient=VERTICAL, command=canvas.yview)
canvas.configure(yscrollcommand=scroll.set)


def onFrameConfigure(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))

# Mouse scroll event callback um bei den Slidern scrollen zu können


def mouse_scroll(event, canvas):
    if event.delta:
        canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
    else:
        if event.num == 5:
            move = 1
        else:
            move = -1

        canvas.yview_scroll(move, 'units')


# Events binden
backSlider.bind('<Configure>', lambda event,
                canvas=canvas: onFrameConfigure(canvas))
backSlider.bind_all('<MouseWheel>', lambda event,
                    canvas=canvas: mouse_scroll(event, canvas))
backSlider.bind_all('<Button-4>', lambda event,
                    canvas=canvas: mouse_scroll(event, canvas))
backSlider.bind_all('<Button-5>', lambda event,
                    canvas=canvas: mouse_scroll(event, canvas))

canvas.pack(side=LEFT, fill=Y)
canvas_frame = canvas.create_window((4, 4), window=backSlider, anchor="nw")
scroll.pack(side=LEFT, fill=Y)

# Gesichtsgenerations panel erstellen
backFace = Frame(root, bg='white', width=740, height=800)
# platzhalter Bild erstellen
img = np.zeros((64, 64, 3))
img = Image.fromarray(np.uint8(img)).convert('RGB')
img = img.resize((438, 438), Image.ANTIALIAS)
img = ImageTk.PhotoImage(image=img)

imgArea = Label(backFace, image=img)
imgArea.image = img
backFace.pack(side=RIGHT)
imgArea.pack(side=RIGHT, fill=BOTH, expand=1)

# Jetzige Principle Component werte updaten wenn Slider geändert werden


def update_slider_vals():
    for i in range(bottleneck):
        slideArr[i].set(currentPCA[i])


# Beliebiges Bild laden, das nachgemacht werden soll
def choose_file():
    filename = askopenfilename()
    if not (filename.endswith(".png") or filename.endswith(".jpg")):
        print("Invalid Filetype")
        return
    im = Image.open(filename)
    img_arr = np.array(im)
    if not img_arr.shape == (64, 64, 3):
        print("Invalid image shape")
        return
    #img_arr = np.reshape(img_arr, 12288)
    _arr = img_arr / 255
    test_arr = []
    test_arr.append(_arr)
    test_arr = _encoder.predict(np.asarray(test_arr))

    test_arr = sc.transform(test_arr)
    test_arr = pca.transform(test_arr)
    for i in range(bottleneck):
        currentPCA[i] = test_arr[0, i]
    update_slider_vals()


btn_file_chooser = Button(root,
                          text="Choose scaled and aligned Image to recreate", command=choose_file)
btn_file_chooser.pack(side=RIGHT, anchor='ne')


# Wie viele Slider sind in einer Reihe
sliders_in_row = 4

# Bild mit allen Principle Components auf 0.0 generieren
currentPCA = np.zeros((bottleneck,))


# Bild nach den jetzigen Principle Components generieren
def updatePCA(event):
    for i in range(bottleneck):
        currentPCA[i] = slideArr[i].get()
    img = _decoder.predict(np.array([currentPCA]))
    img = img * 255
    img = img.reshape((64, 64, 3))
    img = Image.fromarray(np.uint8(img)).convert('RGB')
    img = img.resize((438, 438), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image=img)
    imgArea.configure(image=img)
    imgArea.image = img


# Slider erstellen
slideArr = []
for i in range(bottleneck):
    slideArr.append(Scale(backSlider, from_=pca_min[i], to=pca_max[i],
                          orient=HORIZONTAL, label=f"PC{i}", resolution=0.01, command=updatePCA))
    slideArr[i].set(((pca_min[i] + pca_max[i]) / 2))
    slideArr[i].grid(row=int(i/sliders_in_row), column=i % sliders_in_row)

# Main loop des Fensters einleiten
root.mainloop()
