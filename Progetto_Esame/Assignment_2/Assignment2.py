from pathlib import Path
import pickle
import cv2 as cv

PATH = 'Progetto_Esame/Assignment_2/DATASET/AID'

def leggi_foto(cartella):
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            img = cv.imread(str(f))
            if img is not None:  # imread ritorna None se il file è corrotto
                yield img

sift = cv.SIFT_create()
sift_list = []

for file in leggi_foto(PATH):
    keypoints, descriptors = sift.detectAndCompute(file, None)
    sift_list.append(descriptors)
    #print(f"Descriptors shape: {descriptors.shape}")

print(f"Total descriptors extracted: {len(sift_list)}")

with open('sift_descriptors.pkl', 'wb') as f:
    pickle.dump(sift_list, f)