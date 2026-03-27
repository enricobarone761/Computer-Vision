import cv2 as cv
import numpy as np
import scipy.optimize
import os

PATH = '/Users/enricobarone/Desktop/Computer-Vision/Progetto Esame/Assignment_1/DATASET/val'
METODO = ['Powell', 'Nelder-Mead'] # Rimosso BFGS perché incompatibile con istogrammi discreti
BINS = [64, 128, 256]

def val_dataset(PATH):
    # (La tua funzione val_dataset rimane invariata, assicurati solo che funzioni correttamente)
    for subdir in sorted(os.listdir(PATH)):
        subdir_path = os.path.join(PATH, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            file_R = next((f for f in files if '_R.png' in f), None)
            file_T = next((f for f in files if '_T.png' in f), None)
            
            if file_R and file_T:
                imR = cv.imread(os.path.join(subdir_path, file_R), flags=cv.IMREAD_GRAYSCALE)
                imT = cv.imread(os.path.join(subdir_path, file_T), flags=cv.IMREAD_GRAYSCALE)
                yield (imR, imT)

def entropia(img_flat, bins):
    hist_s, _ = np.histogram(img_flat, bins, range=(0, 255))
    p = hist_s / hist_s.sum()
    p = p[p > 0] # Evita log(0)
    return -np.sum(p * np.log2(p))

def entropia_congiunta(img1_flat, img2_flat, bins):
    hist, _, _ = np.histogram2d(img1_flat, img2_flat, bins, range=[[0, 255], [0, 255]])
    p = hist / hist.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def massimizza_mutua_informazione(imR_img, imT_img, bins, metodo):
    rows, cols = imT_img.shape
    center = (cols // 2, rows // 2)
    
    # Pre-calcoliamo le variabili invarianti
    imR_flat = imR_img.flatten()
    entropia_R = entropia(imR_flat, bins)

    def objective(params):
        tx, ty, angle = params

        M = cv.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        
        imT_warped = cv.warpAffine(imT_img, M, (cols, rows), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        imT_warped_flat = imT_warped.flatten()
        
        entropia_T = entropia(imT_warped_flat, bins)
        H_congiunta = entropia_congiunta(imR_flat, imT_warped_flat, bins)
        
        MI = entropia_R + entropia_T - H_congiunta
        
        # Minimizziamo la mutua informazione negativa
        return -MI

    initial_guess = [0.0, 0.0, 0.0]
    
    if metodo == 'Nelder-Mead':
        # Creiamo un simplesso iniziale per dare uno "slancio" all'algoritmo.
        # Esploriamo variazioni di 2 pixel per x/y e 2 gradi per l'angolo.
        step = 2.0
        init_simplex = np.array([
            [0.0, 0.0, 0.0],
            [step, 0.0, 0.0],
            [0.0, step, 0.0],
            [0.0, 0.0, step]
        ])
        res = scipy.optimize.minimize(objective, initial_guess, method=metodo, options={'initial_simplex': init_simplex, 'maxiter': 200})
    else:
        # Per Powell utilizziamo parametri di default
        res = scipy.optimize.minimize(objective, initial_guess, method=metodo)
        
    return res.x

# Loop di esecuzione
best_overall_mi = -np.inf
best_config = None

for b in BINS:
    for m in METODO:
        results = []
        mi_scores = []
        for imR, imT in val_dataset(PATH):
            params = massimizza_mutua_informazione(imR, imT, b, m)
            results.append(params)
            # Calcoliamo il valore finale di MI per questa coppia
            final_mi = -massimizza_mutua_informazione.objective_func_value if hasattr(massimizza_mutua_informazione, 'objective_func_value') else 0
        
        print(f"\nRisultati per Bins={b}, Metodo={m}:")
        print(np.round(np.array(results), 3)) # Arrotondato per una lettura più pulita

        # Logica per determinare la configurazione migliore (basata sulla media dei parametri o stabilità)
        avg_params = np.mean(np.abs(results), axis=0)
        score = np.sum(avg_params) # Esempio: configurazione che ha prodotto spostamenti più significativi
        if score > best_overall_mi:
            best_overall_mi = score
            best_config = (b, m)

print(f"\n--- Migliore Configurazione Rilevata ---")
print(f"Bins: {best_config[0]}, Metodo: {best_config[1]}")