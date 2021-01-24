import numpy as np  # Mathe bibliothek
# Keras-layers: Vorgefertigtes Modul für verschiedene Layer in einem Neuronalen Netz
from keras.layers import Input, Dense, Dropout
# Keras-models: Teil von keras um das Model zu erstellen und ein model aus einer JSON datei zu laden
from keras.models import Model, model_from_json
from keras import regularizers  # Keras regularisation um Overfitting zu verhindern
# Keras Datenset normalisierung
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam  # Keras Weight optimizer Algorithmus

# SKLearn modul: Normalisiert die Bilder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # SKLearn für den PCA algorithmus

import os  # OS modul um das Datenset zu laden

from PIL import Image  # Pillow Image modul um das Datenset zu laden


# Matplotlib Pyplot um Graphen und Bilder zu erstellen und speichern
import matplotlib.pyplot as plt

import datetime  # Datetime modul um auf die Zeit zugreifen zu können

# -------------------------------------------------------------------

import dense_conv_dataset  # Skript zum komprimieren des Datensets
import create_PCA_files  # Skript um alle PCA relevante Dateien zu erstellen
import create_loss_plot  # Skript um die Loss werte in einem Graph darzustellen

# ----------------------------------------------------------------------------------------------------

# Mit diesem Skript wird die entgültige KI mit dem vortrainierten Convolutional Encoder trainiert

# Random Seed auf 0 setzen um ähnliche verhaltensmuster zu erwarten
np.random.seed(0)

# Dataset laden als Liste von 3 dimensionalen Listen, die alle Bilder beinhaltet in der normalen 2D form mit 3 Farbkanälen


def loadTrainData(folder='training_data\\pca\\dataset_aligned', save=True):
    dataset = []
    print("Loading Dataset...")
    for image_file_name in os.listdir(folder):
        if not image_file_name.endswith(".png"):
            continue
        im = Image.open(f'{folder}\\{image_file_name}')
        img_arr = np.array(im)
        #img_arr = np.reshape(img_arr, 12288)
        _arr = img_arr / 255

        dataset.append(_arr)
    print("Dataset loaded!")
    if save:
        print("Saving Dataset...")
        np.save("training_PCA\\data\\dataset.npy", np.asarray(dataset))
        print("Dataset saved!")
    return np.asarray(dataset)


# Neues Model erstellen
def newModel():
    print("Creating new Model...")

    # Der Decoder soll 'bottleneck'-input Neuronen haben
    input_decoder = Input(shape=(bottleneck,))

    # Dropout: 0.5 zu viel; 0.2 zu wenig; 0.35  zu viel; 0.25 zu wenig; 0.3 zu viel;  
    x = Dense(600, activation='relu')(input_decoder)    

    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2000, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(4000, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(8000, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(12288, activation='sigmoid')(x)

    decoder_model = Model(input_decoder, x)

    # Convolutional encoder aus dem gespeichertem Model laden
    json_file = open(
        f'{CONVOLUTION_ENCODER_PATH}{CONVOLUTION_ENCODER_VERSION}_modelConvEncoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _encoder_model = model_from_json(loaded_model_json)
    _encoder_model.load_weights(
        f"{CONVOLUTION_ENCODER_PATH}{CONVOLUTION_ENCODER_VERSION}_modelConvEncoder.h5")
    print("Trained Convolutional Encoder Model loaded!")

    # Model daten speichern
    file = open("training_PCA\\model_info.data", "w")
    file.write(f"Bottleneck: {bottleneck}\n")
    file.write(f"Learning Rate: {LR}\n")
    file.write(f"Batch Size: {BATCH_SIZE}\n")
    file.write(f"Starttime: {str(datetime.datetime.now())}\n")
    file.write(
        f"\nEncoder Model: ({CONVOLUTION_ENCODER_PATH}{CONVOLUTION_ENCODER_VERSION}_modelConvEncoder)\n")
    _encoder_model.summary(print_fn=lambda x: file.write(x + "\n"))
    file.write("\nDecoder Model:\n")
    decoder_model.summary(print_fn=lambda x: file.write(x + "\n"))

    file.write(f"Training Set: {training_set_size}\n")
    file.write(f"Validation Set: {len(arr) - training_set_size}")
    file.close()

    print("Model done!")
    return (_encoder_model, decoder_model)


# Decoder Model abspeichern
def saveModel(train=True, iter=0):
    print("Saving Model...")
    if train:
        folder = 'training_PCA\\models\\'
    else:
        folder = 'models\\'

    model_json = _decoder.to_json()
    with open(f"{folder}{iter}_modelDecoder.json", "w") as json_file:
        json_file.write(model_json)
    _decoder.save_weights(f"{folder}{iter}_modelDecoder.h5")

    print("Trained model saved!")

# Decoder Model laden


def loadModel(path, iter=0):
    # Convolutional encoder aus dem gespeichertem Model laden
    json_file = open(
        f'{CONVOLUTION_ENCODER_PATH}{CONVOLUTION_ENCODER_VERSION}_modelConvEncoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _encoder_model = model_from_json(loaded_model_json)
    _encoder_model.load_weights(
        f"{CONVOLUTION_ENCODER_PATH}{CONVOLUTION_ENCODER_VERSION}_modelConvEncoder.h5")
    print("Trained Convolutional Encoder Model loaded!")

    # Decoder laden
    json_file = open(f'{path}{iter}_modelDecoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _decoder_model = model_from_json(loaded_model_json)
    _decoder_model.load_weights(f"{path}{iter}_modelDecoder.h5")
    print("Trained Model loaded!")
    return (_encoder_model,_decoder_model)

# Decoder 'e'-epochen lang trainieren


def train(x_train, x_test, e, start_e=0):

    # Ausgang des Encoders berechnen
    print("Calculating Training Data Dense Representation...")
    x_train_dense = _encoder.predict(x_train)
    x_test_dense = _encoder.predict(x_test)
    print("Dense Representation done!")

    # Principle Components der Encodings berechnen
    print("Calculating Principal Components...")
    sc = StandardScaler()
    x_train_std = sc.fit_transform(
        x_train_dense.reshape(x_train_dense.shape[0], 3840))
    x_test_std = sc.transform(
        x_test_dense.reshape(x_test_dense.shape[0], 3840))

    pca = PCA(bottleneck)  # 'bottleneck'-Components behalten
    x_train_std = pca.fit_transform(x_train_std)
    x_test_std = pca.transform(x_test_std)

    print("Saving PCA files...")
    create_PCA_files.save_PCA(sc, pca, "training_PCA\\", x_train_std)

    # PCA Varianz der einzelnen Components als Graph abspeichern
    pca_variance = pca.explained_variance_ratio_
    plt.clf()
    plt.bar(range(bottleneck), pca_variance, align='center')
    plt.savefig('training_PCA\\pca\\PCA.png')

    print("PCA saved!")

    x_train = x_train.reshape(x_train.shape[0], 64*64*3)
    x_test = x_test.reshape(x_test.shape[0], 64*64*3)
    # Trainieren
    print("Training started...")
    for i in range(start_e, e):
        print(f"\nEPOCH: {i}")

        history = _decoder.fit(x_train_std, x_train, epochs=1, batch_size=BATCH_SIZE,
                               shuffle=True, validation_data=(x_test_std, x_test))

        # Protokol weiter führen
        file = open(f"training_PCA\\training_history.txt", "a")
        file.write(
            f"Epoch: {i}; Loss: {history.history['loss'][-1]}; Val_loss: {history.history['val_loss'][-1]}\n")
        file.close()

        # Alle 10 epochen wird ein Testsample abgespeichert
        if i % 10 == 0:
            test(x_test, x_test_dense, x_test_std, n=10, iter=i)

            saveModel(iter=i)
            # Loss graph der letzten 30 epochen abspeichern
            create_loss_plot.loss_plot(min_epoch=i-30 if i > 30 else 0, dpi='figure')
        # Alle 20 epochen wird das Model und der Trainings-loss graph gespeichert
        if i % 20 == 0:
            if i > 30:
                create_loss_plot.loss_plot(dpi='figure')

    # Am ende des Trainings wird das Netz nochmal getestet und die Test samples werden gespeichert
    test(x_test, x_test_dense, x_test_std, n=10, iter=e)
    # Das Model wird gespeichert
    saveModel(train=False, iter=e)

    # Finaler Loss-Graph wird erstellt
    create_loss_plot.loss_plot()

# Model testen und Test samples abspeichern


def test(x_test, x_test_dense, x_test_std, n=10, iter=1):

    decoded_v = _decoder.predict(x_test_std)

    plt.clf()
    plt.figure(figsize=(20, 9))
    # n test samples anzeigen
    for i in range(n):
        # Ursprüngliches Bild
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(x_test[i].reshape(64, 64, 3))
        plt.gray()

        # Conv Encoding
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(x_test_dense[i].reshape(32, 40, 3))

        # PCA
        ax = plt.subplot(4, n, i + 1 + n + n)
        plt.imshow(x_test_std[i].reshape(10, 20))
        plt.gray()

        # Generiertes Bild
        ax = plt.subplot(4, n, i + 1 + n + n + n)
        plt.imshow(decoded_v[i].reshape(64, 64, 3))
        plt.gray()
    # Test sample speichern
    plt.savefig(f'training_PCA\\scores\\{iter}_scores.png')
    print("Model test saved!")


# -------------------------------------------

# Bottleneck auf welche Dimension die Daten insgesamt geschrumpft werden sollen
bottleneck = 200
# Lernrate für das Training
LR = 0.00001
# Batch größe in wie vielen Bildern das Netz auf einmal trainiert wird
BATCH_SIZE = 300

# Welche epoche des in 'CONVOLUTIONAL_ENCODER_PATH' gespeicherten Convolutional Encoders soll verwendet werden
CONVOLUTION_ENCODER_VERSION = 170
CONVOLUTION_ENCODER_PATH = "models\\conv_1.8.2020\\models\\"

# --------------------------------------------------------------------


training_set_size = 18500  # ca. 75% von 21648 samples

# Datenset laden
arr = loadTrainData(folder='training_data\\pca\\dataset_aligned', save=False)
# Model erstellen
_encoder, _decoder = newModel()

# Dense dataset
print("Densing Dataset...")
dense_conv_dataset.dense_set(
   path="training_PCA\\", conv_path=f"{CONVOLUTION_ENCODER_PATH}{CONVOLUTION_ENCODER_VERSION}_modelConvEncoder")
print("Dense Dataset saved!")

# Decoder Model compilen
_decoder.compile(optimizer=Adam(lr=LR), loss='binary_crossentropy')

# Datenset in Training-set und Test-set teilen und 2000 epochen lang damit den Decoder trainieren
train(arr[:training_set_size], arr[training_set_size:], 2000, start_e=0)
