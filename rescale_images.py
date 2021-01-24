import numpy as np
import os

import cv2
import imageio

# Diese Skript wird benutzt um vorhandene Bilder (zb. Aus dem CelebA dataset) auf die richtigen Maße zu skalieren,
# mit FaceDetection das Gesicht zentrieren und schlechte Bildqualitäten oder nicht erkennbare Gesichter auszusortieren


# Hier wird das Bild auf die richtigen Maße und zentriert um das Gesicht, geschnitten
def crop_image(img, img_ratio, face):
    # Aussortierung schlecht erkannter Gesichter (recht selten aber tritt auf)
    if face[2] != face[3]:
        print("Error: Face width is not equal to face height")
        assert()
    
    # Maße berechnen
    spacing = 40
    face_box_width = face[2]
    face_middle = (face[1] + face[3]) / 2

    w = face_box_width + spacing
    x = face[0] - (spacing / 2)

    h = w  # * img_ratio
    y = face_middle - (h / 2.5) + (spacing / 2)

    # Bild zuschneiden und zurückgeben
    cropped = img[int(y):int(y+h), int(x):int(x+w)]
    return cropped


# Hier wird eine Liste aller Gesichter zurückgegeben die in dem Bild gefunden werden könne
def detectFaces(img, scaleF=1.2, minNeigh=5, minDim=(30, 30)):
    # Face Detection cascade für den vorgefertigten Klassifizierer aus dem OpenCV modul (genommen aus einem Github FaceDetection projekt)
    faceCascade = cv2.CascadeClassifier(
        "git\\FaceDetect\\haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    # Gesichter suchen und zurückgeben
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

# --------------------------------------------------------------------------


# Ordner mit den nicht skalierten Bildern
FOLDER = "training_data\\pca\\complete_dataset"
# Ordner mit den fertigen Bildern
FOLDER_OUT = "training_data\\pca\\dataset_aligned"
# Ordner mit schlechten Bildern
FOLDER_PURGE = "training_data\\pca\\dataset_purged"

# Alle Bilder aus dem ursprungs ordner lesen
i = len(os.listdir(FOLDER_OUT))
for image_file_name in os.listdir(FOLDER):
    if not image_file_name.endswith(".jpg"):
        continue
    img_arr = cv2.imread(f'{FOLDER}\\{image_file_name}')
    # Gesichter erkennen
    faces = detectFaces(img_arr)

    # Bilder mit schlechter Gesichtsqualität, mehreren Gesichtern und keinen Gesichtern aussortieren
    if len(faces) != 1:
        cv2.imwrite(f"{FOLDER_PURGE}\\{image_file_name}", img_arr)
        continue

    # Bild zuschneiden
    cropped_image = crop_image(
        img_arr, (float(img_arr.shape[0]) / float(img_arr.shape[1])), faces[0])

    # Größe überprüfen (kann zu klein sein) und zu kleine Bilder aussortieren
    if cropped_image.shape[0] < 30 or cropped_image.shape[1] < 30:
        print(
            f"{image_file_name} has only a cropped size of {str(cropped_image.shape)}")
        cv2.imwrite(f"{FOLDER_PURGE}\\{image_file_name}", img_arr)
        continue

    # Bilder auf 64x64 skalieren und im Output ordner abspeichern
    res = cv2.resize(cropped_image, dsize=(64, 64),
                     interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{FOLDER_OUT}\\{i}.png', res)
    i = i + 1
