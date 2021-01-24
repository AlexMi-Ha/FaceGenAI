import numpy as np
from keras.models import model_from_json

# Dieses Skript nimmt die 'dataset.npy' datei aus dem Pfad 'PATH'
# und lässt das Dataset durch den Convolutional Encoder und
# verkleinert somit die Dategröße des Datensets und lässt die KI im Test schneller laden


def dense_set(path, conv_path=""):

    if not conv_path:
        # KI eckdaten laden (Conv_path)
        file = open(f"{path}model_info.data", "r")
        args = file.readlines()
        file.close()
        conv_path = args[6][args[6].index('(') + 1: len(args[6]) - 2]

    # Convolutional Encoder laden
    json_file = open(
        f'{conv_path}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    _encoder_model = model_from_json(loaded_model_json)
    _encoder_model.load_weights(
        f"{conv_path}.h5")

    # dataset.npy laden und mit dem Encoder verkleinern
    big_data = np.load(f"{path}\\data\\dataset.npy")
    dense_rep = _encoder_model.predict(big_data)

    # Neues Datenset abspeichern
    np.save(f"{path}\\data\\dense_dataset.npy", dense_rep)
