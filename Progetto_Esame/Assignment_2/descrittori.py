from pathlib import Path
import pickle
import numpy as np
import cv2 as cv

PATH = r'Progetto_Esame/Assignment_2/DATASET/AID'

def leggi_foto(cartella):
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            img = cv.imread(str(f), flags=cv.IMREAD_GRAYSCALE)
            if img is not None:# imread ritorna None se il file è corrotto
                print(f"Image {f} read successfully.")
                yield img

N_FEATURES = 2000

sift = cv.SIFT_create(nfeatures=N_FEATURES)
sift_list = np.empty((N_FEATURES * 10_000, 128), dtype=np.float32)

idx = 0

for file in leggi_foto(PATH):
    keypoints, descriptors = sift.detectAndCompute(file, None)
    if descriptors is not None: #and len(descriptors) > 0
        cv.normalize(descriptors, descriptors, norm_type=cv.NORM_L2)
        r,c = descriptors.shape
        sift_list[idx:idx+r] = descriptors
        idx += r
        print(f"Descriptors shape: {r} x {c}, total so far: {idx}")

print(f"Total descriptors extracted: {len(sift_list)}")

# Taglia solo la parte effettivamente usata
sift_list = sift_list[:idx]

#descriptors = np.vstack(sift_list)
#cv.normalize(descriptors, descriptors, norm_type=cv.NORM_L2)

with open(r'Progetto_Esame/Assignment_2/descrittori_new.pkl', 'wb') as f:
    pickle.dump(sift_list, f)

print("Descriptors saved to descrittori_new.pkl")