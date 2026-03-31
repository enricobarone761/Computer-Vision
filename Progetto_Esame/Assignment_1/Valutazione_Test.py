import cv2 as cv
import numpy as np
import pandas as pd
import os
import scipy.optimize
import Assignment1_Ufficiale as A1

# Percorsi
PATH_TEST = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "test")
GT_PATH = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "GT.csv")

# ==========================================
# INSERISCI QUI I PARAMETRI SCELTI!
# ==========================================
METODO_SCELTO = 'Powell'  # Puoi cambiarlo con 'BFGS' o 'Powell'
BINS_SCELTI = 128              # 64, 128 o 256
# ==========================================

def main():
    
    dati = list(A1.load_dataset(PATH_TEST, filtri=False))
    gt = pd.read_csv(GT_PATH, sep=';')
    gt = gt[gt['Dataset'] == 'test'].reset_index(drop=True)
    
    risultati = []
    conto = 0
    
    for r, t in dati:
        # preparo le immagini in grigio
        r_gray = cv.cvtColor(r, cv.COLOR_BGR2GRAY)
        r_gray = cv.GaussianBlur(r_gray, (7, 7), 0)
        t_gray = cv.cvtColor(t, cv.COLOR_BGR2GRAY)
        t_gray = cv.GaussianBlur(t_gray, (7, 7), 0)

        # trovo parametri
        tx, ty, angolo = A1.massimizza_mutua_informazione(r_gray, t_gray, BINS_SCELTI, METODO_SCELTO)
        risultati.append([tx, ty, angolo])
        
        # creo l'immagine per farla vedere
        h , w = t_gray.shape

        M = cv.getRotationMatrix2D((w // 2, h // 2), np.degrees(angolo), 1.0)
        M[0, 2] = M[0, 2] + tx
        M[1, 2] = M[1, 2] + ty
        t_allineata = cv.warpAffine(t, M, (w, h))
        
        diff = cv.absdiff(r, t_allineata)
        finestra = np.hstack((r, t_allineata, diff))
        
        cv.imshow("Risultato " + str(conto), finestra)
        cv.waitKey(0)    
        cv.destroyAllWindows()
        
        conto = conto + 1

    # calcolo errori finali
    df = pd.DataFrame(risultati, columns=['Tx', 'Ty', 'Angolo'])
    df['MSE_trasl'] = ((df['Tx'] - gt['Tx'])**2 + (df['Ty'] - gt['Ty'])**2) / 2
    df['MSE_rot'] = (df['Angolo'] - gt['AngleRad'])**2
    df['MSE_tot'] = df['MSE_trasl'] + df['MSE_rot']

    print(df)
    print("Media traslazione:", df['MSE_trasl'].mean())
    print("Media rotazione:", df['MSE_rot'].mean())
    print("Media totale:", df['MSE_tot'].mean())

if __name__ == "__main__":
    main()
