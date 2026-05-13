import cv2 as cv
import numpy as np
import pandas as pd
import Assignment1_Ufficiale as A1

TEST_PATH = 'Progetto_Esame/Assignment_1/DATASET/test'
GT_PATH = 'Progetto_Esame/Assignment_1/DATASET/GT.csv'

#PARAMETRI SCELTI CON IL VALIDATION SET
METODO = 'Nelder-Mead'  
BINS = 64              

def main():
    
    # Carichiamo le immagini senza filtri: li applichiamo manualmente dopo in quanto servono entrambe le versioni
    immagini = list(A1.load_dataset(TEST_PATH, filtri=False))
    gt_test = pd.read_csv(GT_PATH, sep=';')
    gt_test = gt_test[gt_test['Dataset'] == 'test'].reset_index(drop=True)  #reset_index necessario per usare .loc[i] nel loop

    risultati = []
    
    for i in range(len(immagini)):
        imR = immagini[i][0]
        imT = immagini[i][1]
        #versioni in grigio e filtrate usate solo per l'ottimizzazione
        imR_mod = cv.GaussianBlur(cv.cvtColor(imR, cv.COLOR_BGR2GRAY), (5, 5), 0)
        imT_mod = cv.GaussianBlur(cv.cvtColor(imT, cv.COLOR_BGR2GRAY), (5, 5), 0)

        tx, ty, theta = A1.massimizza_mutua_informazione(imR_mod, imT_mod, BINS, METODO)
        
        h , w = imT_mod.shape

        T = np.array([[np.cos(theta), -np.sin(theta), tx], 
                      [np.sin(theta),  np.cos(theta), ty]],
                      dtype=np.float32)

        imT_allineata = cv.warpAffine(imT, T, (w, h), flags=cv.INTER_LINEAR)
        
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

        A1.plot_risultato(imR, imT_allineata, f'c{i+1}', is_test_plot=True)

    df = pd.DataFrame(risultati)

    print(f"\n--- Risultati per METODO: {METODO}, BINS: {BINS} ---")
    print(df.round(4))

    statistiche = A1.calcola_statistiche(df, gt_test, METODO, BINS)

    A1.stampa_riepilogo_finale([statistiche])

if __name__ == "__main__":
    main()
