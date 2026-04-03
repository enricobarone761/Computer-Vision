import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import Assignment1_Ufficiale as A1

# Percorsi
PATH_TEST = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "test")
GT_PATH = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "GT.csv")

# ==========================================
# INSERISCI QUI I PARAMETRI SCELTI!
# ==========================================
METODO = 'Powell'  
BINS = 64              
# ==========================================

def main():
    
    immagini = list(A1.load_dataset(PATH_TEST, filtri=False))
    gt_test = pd.read_csv(GT_PATH, sep=';')
    gt_test = gt_test[gt_test['Dataset'] == 'test'].reset_index(drop=True)

    risultati = []
    
    for i in range(len(immagini)):
        imR = immagini[i][0]
        imT = immagini[i][1]
        # preparo le immagini in grigio
        imR_mod = cv.GaussianBlur(cv.cvtColor(imR, cv.COLOR_BGR2GRAY), (5, 5), 0)
        imT_mod = cv.GaussianBlur(cv.cvtColor(imT, cv.COLOR_BGR2GRAY), (5, 5), 0)

        # stimo i parametri con bin e metodo deciso con il validation set
        (tx, ty, theta), grafico_MI = A1.massimizza_mutua_informazione(imR_mod, imT_mod, BINS, METODO)
        
        h , w = imT_mod.shape

        T = np.array([[np.cos(theta), -np.sin(theta), tx], 
                      [np.sin(theta),  np.cos(theta), ty]],
                      dtype=np.float32)

        imT_allineata = cv.warpAffine(imT, T, (w, h), flags=cv.INTER_LINEAR)
        
        # Calcolo degli errori per la visualizzazione e per i risultati finali
        err_tx = tx - gt_test.loc[i, 'Tx']
        err_ty = ty - gt_test.loc[i, 'Ty']
        err_angolo = theta - gt_test.loc[i, 'AngleRad']
        
        valori = {
            'Tx_calc': tx,
            'Ty_calc': ty,
            'Angolo_calc': theta,
            'Err_Tx': err_tx,
            'Err_Ty': err_ty,
            'Err_Angolo': err_angolo
        }
        risultati.append(valori)

        # Visualizzazione con grafico MI incorporato
        A1.plot_risultato(imR, imT_allineata, grafico_MI, i + 1, valori, METODO, BINS)

    # creo DataFrame finale direttamente con tutti i dati calcolati
    df = pd.DataFrame(risultati)

    # Mostro i risultati per ogni coppia di immagini
    print(f"\n--- Risultati per METODO: {METODO}, BINS: {BINS} ---")
    print(df.round(4))

    # Calcolo RMSE e statistiche
    statistiche = A1.calcola_statistiche(df, gt_test, METODO, BINS)

    # Stampo riepilogo
    A1.stampa_riepilogo_finale([statistiche])

if __name__ == "__main__":
    main()
