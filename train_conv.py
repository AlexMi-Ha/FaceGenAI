import numpy as np
from keras.layers import Input, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, model_from_json
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import os
from matplotlib import pyplot as plt
import datetime
from PIL import Image
import imageio


# In diesem Skript wird der ursprüngliche Encoder trainiert, der nachher benutzt wird um an den Encodings
# PCA anzuwenden und per Decoder wieder zu einem Bild zu machen

# Random Seed auf 0 setzen um ähnliche Verhaltensmuster erwarten zu können
np.random.seed(0)

# Datenset laden
def loadTrainData(folder='training_data\\conv\\dataset_aligned', save=True):
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
        np.save("training_Conv\\data\\dataset.npy", np.asarray(dataset))
        print("Dataset saved!")
    return np.asarray(dataset)


# -------------------------------------------

# Lernrate mit der das Netz trainiert wird
LR = 0.0001
# Batch größe in wie vielen Bildern das Netz auf einmal trainiert wird
BATCH_SIZE = 200

# Neues Model erstellen
def newModel():
    # 64x64x3 als Input
    input_conv = Input(shape=(64, 64, 3))

    # Encoder erstellen
    # 64x64x3
    x = Conv2D(filters=120, kernel_size=(
        3, 3), padding='same', activation='relu')(input_conv)
    # 64x64x120
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # 32x32x120
    x = Conv2D(filters=160, kernel_size=(
        3, 3), padding='same', activation='relu')(x)
    # 32x32x160
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # 16x16x160
    x = Conv2D(filters=200, kernel_size=(
        3, 3), padding='same', activation='relu')(x)
    # 16x16x200
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # 8x8x200
    x = Conv2D(filters=240, kernel_size=(
        3, 3), padding='same', activation='relu')(x)
    # 8x8x240
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # 4x4x240
    encoded = Reshape([4*4*240])(x) # = 3840

    # Decoder erstellen
    pre_decoded = Reshape([4, 4, 240])(encoded)

    x = UpSampling2D(size=(2, 2))(pre_decoded)
    # 8x8x240
    x = Conv2D(filters=200, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    # 8x8x200
    x = UpSampling2D(size=(2, 2))(x)
    # 16x16x200
    x = Conv2D(filters=160, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    # 16x16x160
    x = UpSampling2D(size=(2, 2))(x)
    # 32x32x160
    x = Conv2D(filters=120, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    # 32x32x120
    x = UpSampling2D(size=(2, 2))(x)
    # 64x64x120
    x = Conv2D(filters=15, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    # 64x64x15
    decoded = Conv2D(filters=3, kernel_size=(
        3, 3), padding='same', activation='sigmoid')(x)
    # 64x64x3

    _autoencoder_model = Model(input_conv, decoded)
    _encoder_model = Model(input_conv, encoded)

    # Model daten speichern
    file = open("training_Conv\\model_info.data", "w")
    file.write(f"Learning Rate: {LR}\n")
    file.write(f"Batch Size: {BATCH_SIZE}\n")
    file.write(f"Starttime: {str(datetime.datetime.now())}\n")
    file.write(f"Training Set: {training_set_size}\n")
    file.write(f"Validation Set: {len(arr_train) - training_set_size}")
    file.close()
    print("Model done!")

    return (_autoencoder_model, _encoder_model)

# Model speichern
def saveModel(train=True, iter=0):
    print("Saving Model...")
    if train:
        folder = 'training_Conv\\models\\'
    else:
        folder = 'models\\'

    # Encoder speichern
    model_json = encoder.to_json()
    with open(f"{folder}{iter}_modelConvEncoder.json", "w") as json_file:
        json_file.write(model_json)
    encoder.save_weights(f"{folder}{iter}_modelConvEncoder.h5")

    # Kompletten AutoEncoder abspeichern
    model_json = autoencoder.to_json()
    with open(f"{folder}{iter}_modelConvAutoencoder.json", "w") as json_file:
        json_file.write(model_json)
    autoencoder.save_weights(f"{folder}{iter}_modelConvAutoencoder.h5")

    print("Trained model saved!")

# Model laden
def loadModel(path, iter=0):
    # Encoder laden
    json_file = open(f'{path}{iter}_modelConvEncoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _encoder = model_from_json(loaded_model_json)
    _encoder.load_weights(f"{path}{iter}_modelConvEncoder.h5")

    # AutoEncoder laden
    json_file = open(f'{path}{iter}_modelConvAutoencoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _autoencoder = model_from_json(loaded_model_json)
    _autoencoder.load_weights(f"{path}{iter}_modelConvAutoencoder.h5")
    print("Trained Model loaded!")
    return (_autoencoder, _encoder)

# Model 'e' epochen lang trainieren
def train(x_train, x_test, e, start_e = 0):
    loss = []
    val_loss = []
    x_loss = []
    print("Training started...")
    for i in range(start_e, e):
        print(f"\nEPOCH: {i}/{e}")
        history = autoencoder.fit(x_train, x_train, epochs=1, batch_size=BATCH_SIZE,
                                  shuffle=True, validation_data=(x_test, x_test))

        # Protokol updaten und abspeichern
        file = open(f"training_Conv\\training_history.txt", "a")
        file.write(
            f"Epoch: {i}; Loss: {history.history['loss'][-1]}; Val_loss: {history.history['val_loss'][-1]}\n")
        file.close()

        # Alle 10 epochen wird ein Test sample abgespeichert
        if i % 5 == 0:
            test(x_test, x_test, n=10, iter=i)
            loss.append(history.history["loss"][-1])
            val_loss.append(history.history["val_loss"][-1])
            x_loss.append(i)
        # Alle 20 epochen wird zusätzlich noch das Model gespeichert
        if i % 10 == 0:
            saveModel(iter=i)

            plt.clf()
            plt.plot(x_loss, loss, label='loss')
            plt.plot(x_loss, val_loss, label='val_loss')
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc="upper left")
            plt.savefig(f'training_Conv\\loss\\{i}_loss.png')
            pass
    
    # Nach dem Training nochmal Test-samples abspeichern und das Model speichern
    test(x_test, x_test, n=10, iter=e)
    saveModel(train=False, iter=e)

    # Finalen Loss Graph erstellen
    plt.clf()
    plt.plot(x_loss, loss, label='loss')
    plt.plot(x_loss, val_loss, label='val_loss')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.savefig('training_Conv\\loss.png')


# Model testen und die Test samples abspeichern
def test(x_test, x_test_std, n=10, iter=1):

    encoded = encoder.predict(x_test)
    decoded = autoencoder.predict(x_test)

    plt.clf()
    plt.figure(figsize=(20, 6))
    for i in range(n):

        # Das ursprüngliche Bild
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(64, 64, 3))
        plt.gray()

        # Encoding
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded[i].reshape(32, 40, 3))
        plt.gray()

        # Generiertes Bild
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded[i].reshape(64, 64, 3))
        plt.gray()
    
    # Test sample abspeichern
    plt.savefig(f'training_Conv\\scores\\{iter}_scores.png')
    print("Model test saved!")


training_set_size = 16500 # ca. 75% von 21932

# Dataset laden
arr_train = loadTrainData(
    'training_data\\dataset_aligned')

# Neues Model erstellen
autoencoder, encoder = newModel()#loadModel("training_Conv\\models\\", 40)
# Model compilen
autoencoder.compile(optimizer=Adam(lr=LR), loss='binary_crossentropy')

# Aufteilen des Dataset in Training set und Test set und dann für 2000 epochen trainieren
train(arr_train[:training_set_size],
      arr_train[training_set_size:], 2000, start_e=0)
