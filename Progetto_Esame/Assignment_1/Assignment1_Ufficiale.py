import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.optimize
from sklearn.metrics import root_mean_squared_error, mean_squared_error

VAL_PATH = 'Progetto_Esame/Assignment_1/DATASET/val'
GT_PATH = 'Progetto_Esame/Assignment_1/DATASET/GT.csv'

METODO = ['Powell', 'Nelder-Mead', 'BFGS']
BINS = [64,128,256]

storico_mi = []

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
                    k = 5
                    imR = cv.cvtColor(imR, cv.COLOR_BGR2GRAY)
                    imR = cv.GaussianBlur(imR, (k, k), 0)
                    imT = cv.cvtColor(imT, cv.COLOR_BGR2GRAY)
                    imT = cv.GaussianBlur(imT, (k, k), 0)

                yield (imR, imT)

def mutua_informazione(im_rif, im_mov, bins):
    # Istogramma congiunto 2D
    hist2d, _, _ = np.histogram2d(
        im_rif.ravel(), im_mov.ravel(),
        bins=bins, range=[[0, 255], [0, 255]]
    )
    # Distribuzione di probabilità congiunta (evita log(0))
    p_con = hist2d / hist2d.sum()
    p_con = p_con[p_con > 0]

    # Distribuzioni marginali
    p_rif = hist2d.sum(axis=1)
    p_rif = p_rif / p_rif.sum()
    p_rif = p_rif[p_rif > 0]

    p_mov = hist2d.sum(axis=0)
    p_mov = p_mov / p_mov.sum()
    p_mov = p_mov[p_mov > 0]

    # Entropie
    Entr_rif = -np.sum(p_rif * np.log2(p_rif))
    Entr_mov = -np.sum(p_mov * np.log2(p_mov))
    Entr_con = -np.sum(p_con * np.log2(p_con))

    return Entr_rif + Entr_mov - Entr_con

def funzione_obiettivo(params, imR_img, imT_img, bins):

    tx, ty, theta = params

    # cv2.getRotationMatrix2D vuole l'angolo in gradi
    #theta_deg = np.degrees(theta)

    h, w = imT_img.shape
    
    T = np.array([[np.cos(theta), -np.sin(theta), tx], 
                  [np.sin(theta),  np.cos(theta), ty]],
                  dtype=np.float32)

    imT_warped = cv.warpAffine(imT_img, T, (w, h), flags=cv.INTER_LINEAR)

    # We minimize negative mutual information
    mi = mutua_informazione(imR_img, imT_warped, bins)
    storico_mi.append(mi)
    return -mi

def massimizza_mutua_informazione(imR_mod, imT_mod, bins, metodo):
    storico_mi.clear() #necessario altrimenti il grafico conterrebbe i valori di MI di tutte le iterazioni di tutti i metodi
    initial_guess = np.array([0, 0, 0])

    opts = {}
    if metodo == 'Nelder-Mead':
        # Con x0 = [0,0,0] il simplesso di default e' troppo piccolo.
        simplesso_iniziale=np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0.01]],
                                     dtype=np.float32)

        opts = {
            'initial_simplex': simplesso_iniziale,
        }

    if metodo == 'BFGS':
        opts = {'eps': [1, 1, 0.01]}

    res = scipy.optimize.minimize(
        funzione_obiettivo,
        initial_guess,
        args=(imR_mod, imT_mod, bins),
        method=metodo,
        options=opts
    )

    return res.x

def plot_risultato(imR, imT_allineata, index, is_test_plot=False):

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    ax1, ax2, ax3, ax4 = axes
    
    ax1.imshow(cv.cvtColor(imR, cv.COLOR_BGR2RGB))
    ax1.set_title("Statica")
    ax1.axis('off')
    
    ax2.imshow(cv.cvtColor(imT_allineata, cv.COLOR_BGR2RGB))
    ax2.set_title("Allineata")
    ax2.axis('off')
    
    diff = cv.absdiff(imR, imT_allineata)
    ax3.imshow(cv.cvtColor(diff, cv.COLOR_BGR2RGB))
    ax3.set_title("Differenza")
    ax3.axis('off')

    if storico_mi:
        ax4.plot(storico_mi)
        ax4.set_title("Andamento MI")
        ax4.set_xlabel("Iterazione")
        ax4.set_ylabel("MI")
    else:
        ax4.axis('off')
        
    fig.suptitle(f"Test {index}")
    
    if is_test_plot == True:
        os.makedirs('Progetto_Esame/Assignment_1/output_test_plots', exist_ok=True)
        fig.savefig(f'Progetto_Esame/Assignment_1/output_test_plots/risultato_test_{index}', dpi=300)
        plt.close(fig)
    else:
        os.makedirs('Progetto_Esame/Assignment_1/output_validation_plots', exist_ok=True)
        fig.savefig(f'Progetto_Esame/Assignment_1/output_validation_plots/risultato_val_{index}', dpi=300)
        plt.close(fig)


def calcola_statistiche(df, gt, metodo, bins):
    rmse_angolo = root_mean_squared_error(df['Angolo_calc'], gt['AngleRad'])
    rmse_tx = root_mean_squared_error(df['Tx_calc'], gt['Tx'])
    rmse_ty = root_mean_squared_error(df['Ty_calc'], gt['Ty'])

    media_errore_traslazione = df[['Err_Tx', 'Err_Ty']].abs().mean().mean()
    media_errore_angolo = df['Err_Angolo'].abs().mean()
    deviazione_std_errore_traslazione = df[['Err_Tx', 'Err_Ty']].std().mean()
    deviazione_std_errore_angolo = df['Err_Angolo'].std()

    return [
        metodo,
        bins,
        rmse_tx,
        rmse_ty,
        rmse_angolo,
        media_errore_traslazione,
        media_errore_angolo,
        deviazione_std_errore_traslazione,
        deviazione_std_errore_angolo
    ]

def stampa_riepilogo_finale(riepilogo):
    colonne = ['Metodo', 'Bins', 'RMSE_Tx', 'RMSE_Ty', 'RMSE_Angolo', 'Media_Err_XY', 'Media_Err_Angolo', 'STD_Err_XY', 'STD_Err_Angolo']
    df_finale = pd.DataFrame(riepilogo, columns=colonne)
    print("\n" + "="*80)
    print("RIEPILOGO FINALE (ORDINATO PER MEDIA)")
    print("="*80)
    print(df_finale.sort_values(by='Media_Err_XY', ascending=True).round(4))

def main():
    immagini = list(load_dataset(VAL_PATH))
    gt = pd.read_csv(GT_PATH, sep=';')
    gt_val = gt[gt['Dataset'] == 'val']
    
    riepilogo = []

    for b in BINS:
        for m in METODO:

            risultati = []
            i=1 #variabile che mi serve per numerare e salvare correttamente le immagini su disco
            for imR, imT in immagini:
                tx, ty, angle = massimizza_mutua_informazione(imR, imT, b, m)
                risultati.append([tx, ty, angle])

                plot_risultato(imR, imT, f'{m}_{b}_c{i}')
                i+=1

            # 2. Creo il DataFrame con i parametri trovati
            df = pd.DataFrame(risultati, columns=['Tx_calc', 'Ty_calc', 'Angolo_calc'])

            # 3. Calcolo gli errori per ogni parametro
            df['Err_Tx'] = df['Tx_calc'] - gt_val['Tx']
            df['Err_Ty'] = df['Ty_calc'] - gt_val['Ty']
            df['Err_Angolo'] = df['Angolo_calc'] - gt_val['AngleRad']
            
            # Mostro i risultati per ogni coppia di immagini (parametri calcolati e errori)
            print(f"\n--- Risultati per METODO: {m}, BINS: {b} ---")
            print(df)

            # 5. Calcolo MSE globali e statistiche
            statistiche = calcola_statistiche(df, gt_val, m, b)
            riepilogo.append(statistiche)

    # Stampo la classifica finale
    stampa_riepilogo_finale(riepilogo)

if __name__ == "__main__":
    main()