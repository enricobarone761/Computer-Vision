import pickle
import cv2 as cv
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np

PATH_DATASET = r"Progetto_Esame\Assignment_2\DATASET\UCMerced_LandUse\Images"

#la seguente funzione di import estrae la classe dal nome del file, ignorando l'ordine
#l'outputn è un generatore di tuple (classe, immagine)
def leggi_foto(cartella):
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            img = cv.imread(str(f), flags=cv.IMREAD_GRAYSCALE)
            if img is not None:# imread ritorna None se il file è corrotto
                classe = f.parent.name
                print(f"Image {f} read successfully with class {classe}.")
                yield (classe, img)


def genera_istogrammi(PATH_VOCABOLARIO):

    sift = cv.SIFT_create()
    lista_istogrammi = []

    with open(PATH_VOCABOLARIO, 'rb') as f:
        km_vocabolario = pickle.load(f)
        print("fatto")

    for classe, img in leggi_foto(PATH_DATASET):
        print(f"Processing image of class {classe} with shape {img.shape}.")
        _ , descrittori = sift.detectAndCompute(img, None)

        if descrittori is not None:
            print(f"Extracted {len(descrittori)} descriptors from image of class {classe}.")
            cv.normalize(descrittori, descrittori, norm_type=cv.NORM_L2)
            prediction = km_vocabolario.predict(descrittori)
            print(f"Predicted class: {prediction}")

            histogram = np.bincount(prediction, minlength=km_vocabolario.n_clusters)
            histogram = histogram / np.linalg.norm(histogram) #normalizzazione L2
            # plt.bar(range(km_vocabolario.n_clusters), histogram)
            # plt.show()
            lista_istogrammi.append((classe, histogram))

    f.close()
    return lista_istogrammi
        

for k in [50, 100, 500]:
    PATH = f"Progetto_Esame/Assignment_2/descrittori&vacabolario/vocab_k{k}.pkl"
    lista_istogrammi = genera_istogrammi(PATH)
    
    with open(f"istogrammi_k{k}.pkl", 'wb') as f:
        pickle.dump(lista_istogrammi, f)






