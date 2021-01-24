import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.externals import joblib

# Dieses Skript wird benutzt um sämtliche Daten für die Principle Component Analysis (PCA) gesammelt und komprimiert abzuspeichern

# PCA anhand Datenset berechnen


def load_PCA(x_train, bottleneck):
    print("Loading PCA...")
    # Normalizen
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)

    # PCA Transform anwenden
    pca = PCA(bottleneck)
    x_train_std = pca.fit_transform(x_train_std)

    return (x_train_std, sc, pca)

# PCA und MinMax werte abspeichern


def save_PCA(sc, pca, path, x_train_std):
        # Scalar und PCA abspeichern
    joblib.dump(sc, f"{path}pca\\scalar.save")
    joblib.dump(pca, f"{path}pca\\pca.save")

    # MinMax werte abspeichern
    np.save(f"{path}pca\\max.npy",
            np.amax(x_train_std[:3000], axis=0))
    np.save(f"{path}pca\\min.npy",
            np.amin(x_train_std[:3000], axis=0))


# Model info holen (bottleneck, conv_path)
def get_model_data(path):
    file = open(f"{path}model_info.data", "r")
    args = file.readlines()
    return (args[0][args[0].index(': ') + 2:], args[6][args[6].index('(') + 1: -2])


# ----------------------------------------------------

def save_from_session(path, conv_encoder):
    np.random.seed(0)

    bottleneck, CONVOLUTION_PATH = get_model_data(path)
    bottleneck = int(bottleneck)

    # Datenset laden
    x_train = np.load(f"{path}data\\dense_dataset.npy")
    # PCA laden und speichern
    x_train_std, sc, pca = load_PCA(x_train, bottleneck)
    save_PCA(sc, pca, path, x_train_std)
