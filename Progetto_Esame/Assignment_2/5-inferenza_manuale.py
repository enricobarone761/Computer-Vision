import pickle
import cv2 as cv
import numpy as np
from sklearn.preprocessing import normalize

IMMAGINE_TEST_PATH = r'Progetto_Esame/Assignment_2/test_image_from_internet.png'


#Miglior vocabolario con score più alto -> K=500 
PATH_VOCABOLARIO = r'Progetto_Esame/Assignment_2/descrittori_e_vacabolario/vocab_k500_1000.pkl'
#Miglior classificatore con accuracy più alta -> SVM
PATH_CLASSIFICATORE = r'Progetto_Esame/Assignment_2/modelli_addestrati/SVM_k500_addestrato.pkl'


#1. caricare foto singola
im = cv.imread(IMMAGINE_TEST_PATH, flags=cv.IMREAD_GRAYSCALE)

#2. estrarre feature sift con normalizzazione L2
sift = cv.SIFT_create()
keypoints, descrittori = sift.detectAndCompute(im, None)
if descrittori is not None:
    print(f"estratti {len(descrittori)} descrittori dall'immagine di test")
    descrittori = normalize(descrittori, norm='l2', axis=1)
else:
    raise ValueError("nessun descrittore trovato nell'immagine di test")

#3. caricare il vocabolario che ha restituito le migliori performance 
with open(PATH_VOCABOLARIO, 'rb') as f:
    km_vocabolario = pickle.load(f)
    print("vocabolario caricato")

#4. assegno ogni descrittore al cluster più vicino, cioè alla parola visiva
prediction = km_vocabolario.predict(descrittori.astype(np.float64)) #ho dovutro necessariamente castare i descrittori a float64 perché il vocabolario esce dal KMeans in questo modo, altrimenti si ottiene un errore.


#5. costruire l'istogramma (con normalizzazione L2)
histogram = np.bincount(prediction, minlength=km_vocabolario.n_clusters)
histogram = histogram / np.linalg.norm(histogram)  #normalizzazione L2


#6. caricare il classificatore addestrato migliore
with open(PATH_CLASSIFICATORE, 'rb') as f:
    classificatore = pickle.load(f)
    print("classificatore caricato")

    
#7. dedurre la classe di appartenenza della foto
classe_predetta = classificatore.predict(histogram.reshape(1, -1))

print(f"classe predetta per l'immagine di test: {classe_predetta[0]}")