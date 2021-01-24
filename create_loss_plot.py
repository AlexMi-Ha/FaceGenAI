from matplotlib import pyplot as plt

# Mit diesem Skript kann man einen Graphen des Loss-verlaufs einer gesamten (oder von einer bestimmten epoche aus)
# Training-session speichern
# Dazu wird das Training Protokol 'training_history.txt' einer session benutzt


def loss_plot(path="training_PCA\\", min_epoch=0, max_epoch=-1, dpi=1200):
    # Protokol öffnen
    file = open(f"{path}training_history.txt", "r")
    arr = file.readlines()

    epochs = []
    loss = []
    val_loss = []

    # Daten aus dem Protokol auslesen
    for i in arr:
        i = i.split('; ')

        e = float(i[0][i[0].index(': ') + 2:])
        if e < min_epoch:
            continue
        if e > max_epoch and max_epoch > 0:
            break

        epochs.append(e)
        loss.append(float(i[1][i[1].index(': ') + 2:]))
        val_loss.append(float(i[2][i[2].index(': ') + 2:]))

    # Graph erstellen
    fig, ax = plt.subplots()

    plt.plot(epochs, loss, label='loss')
    plt.plot(epochs, val_loss, label='val_loss')

    ax.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=.4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")

    # Graph abspeichern
    plt.savefig(f'{path}\\loss\\complete_loss_{min_epoch}-{epochs[-1]}.png', dpi=dpi)
