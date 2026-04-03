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
METODO = 'Powell'  # Puoi cambiarlo con 'BFGS' o 'Powell'
BINS = 64              # 64, 128 o 256
# ==========================================

def main():
    
    dati = list(A1.load_dataset(PATH_TEST, filtri=False))
    gt = pd.read_csv(GT_PATH, sep=';')
    gt = gt[gt['Dataset'] == 'test'].reset_index(drop=True)
    risultati = []
    
    for i in range(len(dati)):
        imR = dati[i][0]
        imT = dati[i][1]
        # preparo le immagini in grigio
        imR_mod = cv.GaussianBlur(cv.cvtColor(imR, cv.COLOR_BGR2GRAY), (5, 5), 0)
        imT_mod = cv.GaussianBlur(cv.cvtColor(imT, cv.COLOR_BGR2GRAY), (5, 5), 0)

        # trovo parametri
        tx, ty, theta = A1.massimizza_mutua_informazione(imR_mod, imT_mod, BINS, METODO)
        
        # creo l'immagine per farla vedere
        h , w = imT_mod.shape

        T = np.array([[np.cos(theta), -np.sin(theta), tx], 
                      [np.sin(theta),  np.cos(theta), ty]],
                      dtype=np.float32)

        imT_allineata = cv.warpAffine(imT, T, (w, h), flags=cv.INTER_LINEAR)
        
        diff = cv.absdiff(imR, imT_allineata)
        
        # Calcolo degli errori per la visualizzazione e per i risultati finali
        err_tx = tx - gt.loc[i, 'Tx']
        err_ty = ty - gt.loc[i, 'Ty']
        err_angolo = theta - gt.loc[i, 'AngleRad']
        
        risultati.append([tx, ty, theta, err_tx, err_ty, err_angolo])

        # Visualizzazione semplice
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
        
        ax1.imshow(cv.cvtColor(imR, cv.COLOR_BGR2RGB))
        ax1.set_title("Statica")
        
        ax2.imshow(cv.cvtColor(imT_allineata, cv.COLOR_BGR2RGB))
        ax2.set_title("Allineata")
        
        ax3.imshow(cv.cvtColor(diff, cv.COLOR_BGR2RGB))
        ax3.set_title("Differenza")
        
        plt.suptitle(f"Test {i + 1} | Tx:{tx:.2f} Ty:{ty:.2f} Ang:{theta:.4f} | ErrTx:{err_tx:.2f} ErrTy:{err_ty:.2f} ErrAng:{err_angolo:.4f}")
        plt.show()

    # creo DataFrame finale direttamente con tutti i dati calcolati
    colonne = ['Tx_calc', 'Ty_calc', 'Angolo_calc', 'Err_Tx', 'Err_Ty', 'Err_Angolo']
    df = pd.DataFrame(risultati, columns=colonne)

    # Mostro i risultati per ogni coppia di immagini
    print(f"\n--- Risultati per METODO: {METODO}, BINS: {BINS} ---")
    print(df.round(4))

    # Calcolo RMSE e statistiche
    rmse_angolo = root_mean_squared_error(df['Angolo_calc'], gt['AngleRad'])
    rmse_tx = root_mean_squared_error(df['Tx_calc'], gt['Tx'])
    rmse_ty = root_mean_squared_error(df['Ty_calc'], gt['Ty'])

    media_errore_traslazione = df[['Err_Tx', 'Err_Ty']].abs().mean().mean()
    media_errore_angolo = df['Err_Angolo'].abs().mean()
    deviazione_std_errore_traslazione = df[['Err_Tx', 'Err_Ty']].std().mean()
    deviazione_std_errore_angolo = df['Err_Angolo'].std()

    # Stampo riepilogo
    print("\n" + "="*80)
    print("RIEPILOGO FINALE")
    print("="*80)
    dati_finale = [[METODO, BINS, rmse_tx, rmse_ty, rmse_angolo, media_errore_traslazione, media_errore_angolo, deviazione_std_errore_traslazione, deviazione_std_errore_angolo]]
    colonne = ['Metodo', 'Bins', 'RMSE_Tx', 'RMSE_Ty', 'RMSE_Angolo', 'Media_Err_XY', 'Media_Err_Angolo', 'STD_Err_XY', 'STD_Err_Angolo']
    print(pd.DataFrame(dati_finale, columns=colonne).round(4))

if __name__ == "__main__":
    main()
