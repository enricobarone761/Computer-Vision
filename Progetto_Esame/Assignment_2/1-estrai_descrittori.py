from pathlib import Path
import pickle
import cv2 as cv
from sklearn.preprocessing import normalize
import numpy as np

PATH = r'Progetto_Esame/Assignment_2/DATASET/AID'

def leggi_foto(cartella):
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            img = cv.imread(str(f), flags=cv.IMREAD_GRAYSCALE)
            if img is not None:# imread ritorna None se il file è corrotto
                print(f"caricata immagine {f}")
                yield img

N_FEATURES = 1000

sift = cv.SIFT_create(nfeatures=N_FEATURES)
sift_list = []

for img in leggi_foto(PATH):
    _, descrittori = sift.detectAndCompute(img, None)
    if descrittori is not None: #and len(descriptors) > 0
        descrittori = normalize(descrittori, norm='l2', axis=1)
        sift_list.extend(descrittori)
        print(f"estratti {descrittori.shape} descrittori dall'immagine")

print(f"Descrittori totali estratti: {len(sift_list)}")

with open(rf'Progetto_Esame/Assignment_2/descrittori_e_vacabolario/descrittori_{N_FEATURES}.pkl', 'wb') as f:
    pickle.dump(sift_list, f)

print("Lista descrittori salvata")