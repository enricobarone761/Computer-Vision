import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import Assignment1_Ufficiale as A1

# Percorsi
PATH_VAL = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "val")
GT_PATH = os.path.join("Progetto_Esame", "Assignment_1", "DATASET", "GT.csv")

def main():
    print("Caricamento dataset di validazione...")
    # Carichiamo le immagini originali, senza i filtri gaussiani e bilineari
    immagini = list(A1.load_dataset(PATH_VAL, filtri=False))
    
    print("Caricamento Ground Truth...")
    gt = pd.read_csv(GT_PATH, sep=';')
    # Indicizzo il validation set per Pair, così la corrispondenza non dipende dall'ordine delle righe
    gt = gt[gt['Dataset'] == 'val'].set_index('Pair')
    
    print(f"Trovate {len(immagini)} coppie di immagini.")
    
    for i in range(len(immagini)):
        pair = f"c{i + 1}"
        imR = immagini[i][0]
        imT = immagini[i][1]
        
        # Prelevo i parametri esatti dal Ground Truth
        riga_gt = gt.loc[pair]
        tx = riga_gt['Tx']
        ty = riga_gt['Ty']
        angolo_deg = float(riga_gt['AngleDegree'])
        angolo_rad = riga_gt['AngleRad']

        # Il GT usa una rotazione attorno all'origine seguita dalla traslazione.
        # Per allineare T a R serve l'inversa di quella matrice.
        theta = angolo_rad
        c = np.cos(theta)
        s = np.sin(theta)
        M = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)
        #M = cv.invertAffineTransform(M_gt)

        # Ricavo le dimensioni per l'applicazione della trasformazione
        h, w = imT.shape[:2]

        # Applico la trasformazione all'immagine T (Moving)
        imT_allineata = cv.warpAffine(imT, M, (w, h))
        
        # Calcolo la differenza in valore assoluto per vedere se sono allineate perfettamente
        diff = cv.absdiff(imR, imT_allineata)
        
        # Visualizzazione con matplotlib per un confronto affiancato
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
        
        ax1.imshow(cv.cvtColor(imR, cv.COLOR_BGR2RGB))
        ax1.set_title("Immagine di Riferimento (R)")
        ax1.axis('off')
        
        ax2.imshow(cv.cvtColor(imT_allineata, cv.COLOR_BGR2RGB))
        ax2.set_title("Immagine da Allineare (T) con GT inverso")
        ax2.axis('off')
        
        ax3.imshow(cv.cvtColor(diff, cv.COLOR_BGR2RGB))
        ax3.set_title("Differenza (Ideale se vicina al nero/grigio)")
        ax3.axis('off')
        
        plt.suptitle(
            f"Val Set {pair} | GT -> Tx: {tx:.2f}, Ty: {ty:.2f}, Angolo: {angolo_deg:.4f} deg ({angolo_rad:.4f} rad)"
        )
        plt.show()

if __name__ == "__main__":
    main()
