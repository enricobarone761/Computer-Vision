from pathlib import Path
import pickle
import numpy as np
import cv2 as cv

PATH = 'Progetto_Esame/Assignment_2/DATASET/AID'

def leggi_foto(cartella):
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            img = cv.imread(str(f), flags=cv.IMREAD_GRAYSCALE)
            if img is not None:# imread ritorna None se il file è corrotto
                print(f"Image {f} read successfully.")
                yield img

sift = cv.SIFT_create(nfeatures=1500)
sift_list = []  

for file in leggi_foto(PATH):
    keypoints, descriptors = sift.detectAndCompute(file, None)
    if descriptors is not None and len(descriptors) > 0:
        sift_list.append(descriptors)
        print(f"Descriptors shape: {descriptors.shape}")

print(f"Total descriptors extracted: {len(sift_list)}")

descriptors = np.vstack(sift_list)
cv.normalize(descriptors, descriptors, norm_type=cv.NORM_L2)

with open('descrittori.pkl', 'wb') as f:
    pickle.dump(descriptors, f)

print("Descriptors saved to descrittori.pkl")