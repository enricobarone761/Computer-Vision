# ***********************************************************************

# NON MODIFICARE QUESTO FILE SE NON RICHIESTO ESPLICITAMENTE DALL'UTENTE

# ***********************************************************************

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.optimize

PATH_VAL = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "val")
PATH_TEST = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "test")
GT_PATH = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "GT.csv")
METODO = ['Powell', 'Nelder-Mead', 'BFGS']
BINS = [64,128,256]

def load_dataset(PATH, filtri=True):
    for subdir in sorted(os.listdir(PATH)):
        subdir_path = os.path.join(PATH, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            
            # Identify R and T files
            file_R = next((f for f in files if '_R.png' in f), None)
            file_T = next((f for f in files if '_T.png' in f), None)
            
            if file_R and file_T:
                imR = cv.imread(os.path.join(subdir_path, file_R))
                imT = cv.imread(os.path.join(subdir_path, file_T))

                # Se il flag è True, applichiamo i filtri
                if filtri:
                    imR = cv.cvtColor(imR, cv.COLOR_BGR2GRAY)
                    imR = cv.GaussianBlur(imR, (7, 7), 0)
                    imT = cv.cvtColor(imT, cv.COLOR_BGR2GRAY)
                    imT = cv.GaussianBlur(imT, (7, 7), 0)

                yield (imR, imT)

def mutua_informazione(img_ref, img_mov, bins):
    # Istogramma congiunto 2D
    hist_2d, _, _ = np.histogram2d(
        img_ref.ravel(), img_mov.ravel(),
        bins=bins, range=[[0, 255], [0, 255]]
    )
    # Distribuzione di probabilità congiunta (evita log(0))
    p_joint = hist_2d / hist_2d.sum()
    p_joint = p_joint[p_joint > 0]

    # Distribuzioni marginali
    p_ref = hist_2d.sum(axis=1)
    p_ref = p_ref / p_ref.sum()
    p_ref = p_ref[p_ref > 0]

    p_mov = hist_2d.sum(axis=0)
    p_mov = p_mov / p_mov.sum()
    p_mov = p_mov[p_mov > 0]

    # Entropie
    H_ref = -np.sum(p_ref * np.log2(p_ref))
    H_mov = -np.sum(p_mov * np.log2(p_mov))
    H_joint = -np.sum(p_joint * np.log2(p_joint))

    return H_ref + H_mov - H_joint

def funzione_obiettivo(params, imR_img, imT_img, bins):
    tx, ty, theta = params

    h, w = imT_img.shape
    #cx, cy = w // 2, h // 2

    # Matrice di rotazione e traslazione calcolata manualmente
    T = np.array([
        [np.cos(theta) , -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty]
    ], dtype=np.float32)

    imT_warped = cv.warpAffine(imT_img, T, (w, h), flags=cv.INTER_LINEAR)

    # We minimize negative mutual information
    return -mutua_informazione(imR_img, imT_warped, bins)

def massimizza_mutua_informazione(imR_mod, imT_mod, bins, metodo):
    initial_guess = np.array([0.0, 0.0, 0.0])

    opts = {}
    if metodo == 'BFGS':
        opts = {'eps': [1.0, 1.0, 0.001]}

    res = scipy.optimize.minimize(
        funzione_obiettivo,
        initial_guess,
        args=(imR_mod, imT_mod, bins),
        method=metodo,
        options=opts
    )
    return res.x

def main():
    immagini = list(load_dataset(PATH_VAL))
    gt = pd.read_csv(GT_PATH, sep=';')
    gt_val = gt[gt['Dataset'] == 'val'].reset_index(drop=True)
    
    riepilogo = []

    for b in BINS:
        for m in METODO:
            # 1. Trovo parametri per ogni immagine
            risultati = []
            for imR, imT in immagini:
                tx, ty, angle = massimizza_mutua_informazione(imR, imT, b, m)
                risultati.append([tx, ty, angle])

            # 2. Creo il DataFrame con i parametri trovati
            df = pd.DataFrame(risultati, columns=['Tx_calc', 'Ty_calc', 'Angolo_calc'])

            # 3. Calcolo gli errori per ogni parametro
            df['Err_Tx'] = df['Tx_calc'] - gt_val['Tx']
            df['Err_Ty'] = df['Ty_calc'] - gt_val['Ty']
            df['Err_Angolo'] = df['Angolo_calc'] - gt_val['AngleRad']

            # 4. Calcolo gli MSE per riga (per informazione)
            df['MSE_Scostamento'] = (df['Err_Tx']**2 + df['Err_Ty']**2)
            df['MSE_Angolo'] = df['Err_Angolo']**2
            
            # Mostro i risultati per ogni coppia di immagini (parametri calcolati e errori)
            print(f"\n--- Risultati per METODO: {m}, BINS: {b} ---")
            print(df[['Tx_calc', 'Ty_calc', 'Angolo_calc', 'Err_Tx', 'Err_Ty', 'Err_Angolo']].round(4))

            # 5. Calcolo MSE globali per questo set di parametri
            mse_scostamento = df['MSE_Scostamento'].mean()
            mse_angolo = df['MSE_Angolo'].mean()
            mse_totale = mse_scostamento + mse_angolo
            
            # Salvo per il riepilogo
            riepilogo.append([m, b, mse_scostamento, mse_angolo, mse_totale])

    # Stampo la classifica finale
    colonne = ['Metodo', 'Bins', 'MSE_Scostamento', 'MSE_Angolo', 'MSE_Totale']
    df_finale = pd.DataFrame(riepilogo, columns=colonne)
    print("\n" + "="*50)
    print("RIEPILOGO FINALE (MSE)")
    print("="*50)
    print(df_finale.sort_values(by='MSE_Totale').round(6))

if __name__ == "__main__":
    main()

# ***********************************************************************

# NON MODIFICARE QUESTO FILE SE NON RICHIESTO ESPLICITAMENTE DALL'UTENTE

# ***********************************************************************