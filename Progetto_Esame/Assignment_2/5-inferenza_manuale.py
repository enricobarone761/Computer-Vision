import pickle
from sklearn.linear_model import LogisticRegression
import cv2 as cv
import numpy as np

PATH_DATASET_TEST = r'Progetto_Esame/Assignment_2/test_image_from_internet.png'

#Miglior vocabolario con score più alto K=500 
PATH_VOCABOLARIO = r'Progetto_Esame/Assignment_2/descrittori_e_vacabolario/vocab_k500_300.pkl'

#Miglior classificatore con accuracy più alta SVM
#TODO specificare perche scelto questo
#TODO la prof ha autorizzato a riaddestrare il modello con tutti i dati
PATH_CLASSIFICATORE = r'Progetto_Esame/Assignment_2/modelli_addestrati/SVM_k500_addestrato.pkl'


#il programma deve:

#1. caricare foto singola
im = cv.imread(PATH_DATASET_TEST, flags=cv.IMREAD_GRAYSCALE)

#2. estrarre feature sift con normalizzazione L2
sift = cv.SIFT_create()
keypoints, descrittori = sift.detectAndCompute(im, None)
if descrittori is not None:
    print(f"Extracted {len(descrittori)} descriptors from the test image.")
    cv.normalize(descrittori, descrittori, norm_type=cv.NORM_L2)
else:
    print("No descriptors found in the test image.")
    raise ValueError("Cannot proceed without descriptors.")

#3. caricare il vocabolario addestrato migliore
with open(PATH_VOCABOLARIO, 'rb') as f:
    km_vocabolario = pickle.load(f)
    print("Loaded the best vocabulary.")

#4. per ogni descrittore estratto fare .predict() per assegnare il cluster di appartenenza (quindi la parola visiva)
prediction = km_vocabolario.predict(descrittori)
print(f"Predicted visual words: {prediction}")


#5. costruire l'istogramma (con normalizzazione L2)
histogram = np.bincount(prediction, minlength=km_vocabolario.n_clusters)
histogram = histogram / np.linalg.norm(histogram) #normalizzazione L2


#6. caricare il classificatore addestrato migliore
with open(PATH_CLASSIFICATORE, 'rb') as f:
    classificatore = pickle.load(f)
    print("Loaded the best classifier.")

    
#7. dedurre la classe di appartenenza della foto
#TODO spiegare il reshape e negli altri file no
classe_predetta = classificatore.predict(histogram.reshape(1, -1))
print(f"Predicted class for the test image: {classe_predetta[0]}")