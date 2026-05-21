import pickle
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path

PATH_DATASET_TEST = r''

#Miglior vocabolario con score più alto K=TODO 
PATH_VOCABOLARIO = r''

#Miglior classificatore con accuracy più alta TODO
PATH_CLASSIFICATORE = r''


#il programma deve:

#1. caricare foto singola 

#2. estrarre feature sift con normalizzazione L2 

#3. caricare il vocabolario addestrato migliore

#4. per ogni descrittore estratto fare kmeans.predict() per assegnare il cluster di appartenenza (quindi la parola visiva)

#5. costruire l'istogramma (con normalizzazione L2)

#6. caricare il classificatore addestrato migliore
    
#7. dedurre la classe di appartenenza della foto