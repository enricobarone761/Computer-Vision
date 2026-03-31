"""
Assignment 1 — Registrazione di Immagini Multimodali
=====================================================
Stima di una roto-traslazione tra coppie di immagini multimodali
massimizzando la Mutua Informazione (MI).
"""

import os
import pandas as pd
import numpy as np
import cv2
import scipy.optimize

PATH_VAL = r"Progetto_Esame\Assignment_1\DATASET\val"
PATH_TEST = r"Progetto_Esame\Assignment_1\DATASET\test"
METODO = ['Powell', 'Nelder-Mead', 'BFGS']
BINS = [64,128,256]

# ──────────────────────────────────────────────
# 1. Calcolo della Mutua Informazione
# ──────────────────────────────────────────────
def mutual_information(img_ref, img_mov, bins):
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


# ──────────────────────────────────────────────
# 2. Trasformazione geometrica (roto-traslazione)
# ──────────────────────────────────────────────
def apply_transform(img, params, out_shape):
    """Applica la roto-traslazione all'immagine con rotazione attorno al centro."""
    theta_rad, tx, ty = params
    h, w = out_shape[:2]
    center = (w / 2, h / 2)
    
    # cv2.getRotationMatrix2D vuole l'angolo in gradi
    theta_deg = np.degrees(theta_rad)
    
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────
# 3. Funzione obiettivo per l'ottimizzazione
# ──────────────────────────────────────────────
def neg_mi(params, img_ref, img_mov, bins):
    """Negativo della MI (da minimizzare)."""
    warped = apply_transform(img_mov, params, img_ref.shape)
    return -mutual_information(img_ref, warped, bins=bins)


# ──────────────────────────────────────────────
# 4. Caricamento dataset e ground truth
# ──────────────────────────────────────────────

def load_dataset(PATH):
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

# ──────────────────────────────────────────────
# 5. Registrazione di una singola coppia
# ──────────────────────────────────────────────
def register_pair(img_ref, img_mov, bins=64, method="Powell"):
    """Stima la roto-traslazione ottimale massimizzando la MI."""
    blur_ksize=7

    img_ref = cv2.GaussianBlur(img_ref, (blur_ksize, blur_ksize), 0)
    img_mov = cv2.GaussianBlur(img_mov, (blur_ksize, blur_ksize), 0)

    x0 = np.array([0.0, 0.0, 0.0])  # [theta, tx, ty] — identità

    # BFGS stima il gradiente con differenze finite.
    # L'epsilon di default (1e-8) è troppo piccolo per spostare pixel tra i bin
    # dell'istogramma, quindi il gradiente risulta zero e l'ottimizzatore si ferma
    # subito al punto di partenza. Usiamo eps=1.0 (1 pixel / ~0.017 rad) per
    # garantire che le perturbazioni siano abbastanza grandi da cambiare la MI.
    if method == "BFGS":
        opts = {"eps": [0.001, 1.0, 1.0]}

    result = scipy.optimize.minimize(
        neg_mi, x0,
        args=(img_ref, img_mov, bins),
        method=method,
        options=opts,
    )
    return result.x  # [theta, tx, ty]


# ──────────────────────────────────────────────
# 6. Valutazione: errore medio su un sottoinsieme
# ──────────────────────────────────────────────

#TODO
# ──────────────────────────────────────────────
# 8. Visualizzazione immagini allineate e differenza
# ──────────────────────────────────────────────
#TODO


# ──────────────────────────────────────────────
# 7. Main: selezione iperparametri + test finale
# ──────────────────────────────────────────────
def main():
    gt = load_gt(GT_PATH)

    # ── Fase 1: Ricerca iperparametri sul validation set ──
    bins_options = [64, 128, 256]
    methods = ["Powell", "Nelder-Mead", "BFGS"]

    print("=" * 70)
    print("FASE 1: Selezione iperparametri sul VALIDATION set")
    print("=" * 70)

    best_config = None
    best_total_error = float("inf")
    results_summary = []
    all_val_preds = {}   # chiave: (method, bins) → lista predizioni per coppia

    for method in methods:
        for bins in bins_options:
            print(f"\n--- Metodo: {method} | Bins: {bins} ---")
            mean_tx, mean_ty, mean_angle, preds = evaluate(
                gt["val"], bins, method, "VAL"
            )
            total_error = mean_tx + mean_ty + mean_angle
            results_summary.append({
                "method": method, "bins": bins,
                "mean_tx": mean_tx, "mean_ty": mean_ty,
                "mean_angle": mean_angle, "total": total_error
            })
            all_val_preds[(method, bins)] = preds
            print(f"  >> Media: Δtx={mean_tx:.4f}  Δty={mean_ty:.4f}  "
                  f"Δθ={mean_angle:.6f} rad  |  Totale={total_error:.4f}")

            if total_error < best_total_error:
                best_total_error = total_error
                best_config = (method, bins)

    # ── Fase 2: Valutazione finale sul test set ──
    print("\n" + "=" * 70)
    print(f"FASE 2: Valutazione sul TEST set  (Metodo={best_method}, Bins={best_bins})")
    print("=" * 70)

    mean_tx, mean_ty, mean_angle, test_preds = evaluate(
        gt["test"], best_bins, best_method, "TEST"
    )

    # ── Punto 3 della consegna: immagini allineate e differenza per ogni coppia di test ──
    show_aligned_and_diff(gt["test"], test_preds, best_config)


if __name__ == "__main__":
    main()
