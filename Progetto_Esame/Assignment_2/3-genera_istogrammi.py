import pickle
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from pathlib import Path
import numpy as np

PATH_DATASET = r"Progetto_Esame/Assignment_2/DATASET/UCMerced_LandUse/Images"

#la classe viene presa dalla cartella padre, perché il dataset è organizzato per classi
#restituisce coppie (classe, immagine) senza caricare tutto in memoria
def leggi_foto(cartella):
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            img = cv.imread(str(f), flags=cv.IMREAD_GRAYSCALE)
            if img is not None:# imread ritorna None se il file è corrotto
                classe = f.parent.name
                print(f"caricata immagine {f}")
                yield (classe, img)


def genera_istogrammi(PATH_VOCABOLARIO):

    lista_istogrammi = []

    with open(PATH_VOCABOLARIO, 'rb') as f:
        km_vocabolario = pickle.load(f)
        print("vocabolario caricato")

    for classe, descrittori in lista_descrittori:
        prediction = km_vocabolario.predict(descrittori.astype(np.float64))

        #conta i visual word e costruisce il bag of words dell'immagine
        histogram = np.bincount(prediction, minlength=km_vocabolario.n_clusters)
        histogram = histogram / np.linalg.norm(histogram)  #normalizzazione L2

        # plt.bar(range(km_vocabolario.n_clusters), histogram)
        # plt.show()

        lista_istogrammi.append((classe, histogram))

    return lista_istogrammi
        

sift = cv.SIFT_create()
lista_descrittori = []

for classe, img in leggi_foto(PATH_DATASET):
    _, descrittori = sift.detectAndCompute(img, None)

    if descrittori is not None:
        descrittori = normalize(descrittori, norm='l2', axis=1)
        lista_descrittori.extend([(classe, descrittori)])
        print(f"estratti {descrittori.shape} descrittori dall'immagine")


for k in [50, 100, 500]:
    
    PATH = rf"Progetto_Esame/Assignment_2/descrittori_e_vacabolario/vocab_k{k}_1000.pkl"
    lista_istogrammi = genera_istogrammi(PATH)
    
    with open(rf"Progetto_Esame/Assignment_2/istogrammi_BoW/istogrammi_k{k}.pkl", 'wb') as f:
        pickle.dump(lista_istogrammi, f)

    print(f"istogrammi salvati per k={k}")
