import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import tabulate as tb

PATH = r"Progetto_Esame\Assignment_1\DATASET\val"
METODO = ['Powell', 'Nelder-Mead', 'BFGS']
BINS = [64,128,256]

def val_dataset(PATH):
    for subdir in sorted(os.listdir(PATH)):
        subdir_path = os.path.join(PATH, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            
            # Identify R and T files
            file_R = next((f for f in files if '_R.png' in f), None)
            file_T = next((f for f in files if '_T.png' in f), None)
            
            if file_R and file_T:
                imR = cv.imread(os.path.join(subdir_path, file_R), flags=cv.IMREAD_GRAYSCALE)
                imT = cv.imread(os.path.join(subdir_path, file_T), flags=cv.IMREAD_GRAYSCALE)

                # Yield the 2D images directly
                yield (imR, imT)

def massimizza_mutua_informazione(imR_img, imT_img, bins, metodo):

    def entropia(img_flat):
        hist_s, _ = np.histogram(img_flat, bins, range=(0, 255))
        p = hist_s / hist_s.sum()  # probabilità reali, somma = 1
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    def entropia_congiunta(img1_flat, img2_flat):
        hist, _, _ = np.histogram2d(img1_flat, img2_flat, bins, range=[[0,255],[0,255]])
        p = hist / hist.sum()      # stessa cosa in 2D
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    def mutua_informazione(img1_flat, img2_flat):
        return entropia(img1_flat) + entropia(img2_flat) - entropia_congiunta(img1_flat, img2_flat)


    rows, cols = imT_img.shape
    center = (cols // 2, rows // 2)

    def objective(params):
        tx, ty, angle = params

        M = cv.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        

        imT_warped = cv.warpAffine(imT_img, M, (cols, rows), flags=cv.INTER_LINEAR)
        
        # We minimize negative mutual information
        return -mutua_informazione(imR_img.flatten(), imT_warped.flatten())


    initial_guess = np.array([0.0, 0.0, 0.0])
    
    opts = {}
    if metodo == 'BFGS':
        # BFGS usa differenze finite per stimare il gradiente. Il default (~1e-8) è 
        # troppo piccolo per alterare i bin dell'istogramma. Forziamo uno step di 1.0.
        opts = {'eps': 1.0} 
    elif metodo == 'Nelder-Mead':
        # Nelder-Mead con guess iniziali a 0 crea un simplesso esplorativo minuscolo (0.00025).
        # Costruiamo un simplesso iniziale manuale con scostamenti di 2.0 pixel/gradi.
        step = 2.0
        opts = {'initial_simplex': np.array([
            [0.0, 0.0, 0.0],
            [step, 0.0, 0.0],
            [0.0, step, 0.0],
            [0.0, 0.0, step]
        ])}

    res = scipy.optimize.minimize(objective, initial_guess, method=metodo, options=opts)
    return res.x

for b in BINS:
    for m in METODO:
        results = []
        for imR, imT in val_dataset(PATH):
            params = massimizza_mutua_informazione(imR, imT, b, m)
            params[2] = (params[2] * np.pi) / 180
            results.append(params)
        
        print(f"\nRisultati per Bins={b}, Metodo={m}:")
        print( tb.tabulate( np.array(results) , tablefmt='rounded_grid' ) )

# # Plot the histogram
# hist_2d, _, _ = np.histogram2d(imR_flat, imT_flat, BINS=128)
# plt.imshow(hist_2d, origin='lower', cmap='hot')
# plt.show()
