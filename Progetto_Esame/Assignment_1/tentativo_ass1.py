"""
Assignment 1 — Registrazione di Immagini Multimodali
=====================================================
Stima di una roto-traslazione tra coppie di immagini multimodali
massimizzando la Mutua Informazione (MI).
"""

import os, warnings
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DATASET")
GT_PATH = os.path.join(DATASET_DIR, "GT.csv")


# ──────────────────────────────────────────────
# 1. Calcolo della Mutua Informazione
# ──────────────────────────────────────────────
def mutual_information(img_ref, img_mov, bins=64):
    """Calcola la Mutua Informazione tra due immagini in scala di grigi."""
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
def load_gt(gt_path):
    """Carica il ground truth dal CSV usando pandas."""
    df = pd.read_csv(gt_path, sep=";")
    # Rinominiamo le colonne per uniformità con il resto del codice
    df = df.rename(columns={
        "Filename": "filename",
        "Pair": "pair",
        "Dataset": "dataset",
        "AngleRad": "angle_rad",
        "Tx": "tx",
        "Ty": "ty"
    })
    return {
        "val": df[df["dataset"] == "val"].to_dict("records"),
        "test": df[df["dataset"] == "test"].to_dict("records")
    }


def load_image_pair(subset, pair, filename):
    """Carica la coppia di immagini (Reference e Moving) in scala di grigi."""
    pair_dir = os.path.join(DATASET_DIR, subset, pair)
    ref_path = os.path.join(pair_dir, f"{filename}_R.png")
    mov_path = os.path.join(pair_dir, f"{filename}_T.png")
    img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    img_mov = cv2.imread(mov_path, cv2.IMREAD_GRAYSCALE)
    if img_ref is None:
        raise FileNotFoundError(f"Reference non trovata: {ref_path}")
    if img_mov is None:
        raise FileNotFoundError(f"Moving non trovata: {mov_path}")
    return img_ref, img_mov


# ──────────────────────────────────────────────
# 5. Registrazione di una singola coppia
# ──────────────────────────────────────────────
def register_pair(img_ref, img_mov, bins=64, method="Powell", blur_ksize=7):
    """Stima la roto-traslazione ottimale massimizzando la MI."""
    if blur_ksize > 0:
        img_ref = cv2.GaussianBlur(img_ref, (blur_ksize, blur_ksize), 0)
        img_mov = cv2.GaussianBlur(img_mov, (blur_ksize, blur_ksize), 0)

    x0 = np.array([0.0, 0.0, 0.0])  # [theta, tx, ty] — identità

    # BFGS stima il gradiente con differenze finite.
    # L'epsilon di default (1e-8) è troppo piccolo per spostare pixel tra i bin
    # dell'istogramma, quindi il gradiente risulta zero e l'ottimizzatore si ferma
    # subito al punto di partenza. Usiamo eps=1.0 (1 pixel / ~0.017 rad) per
    # garantire che le perturbazioni siano abbastanza grandi da cambiare la MI.
    if method == "BFGS":
        opts = {"disp": False, "eps": [0.001, 1.0, 1.0], "gtol": 1e-4}
    else:
        opts = {"disp": False}

    result = minimize(
        neg_mi, x0,
        args=(img_ref, img_mov, bins),
        method=method,
        options=opts,
    )
    return result.x  # [theta, tx, ty]


# ──────────────────────────────────────────────
# 6. Valutazione: errore medio su un sottoinsieme
# ──────────────────────────────────────────────
def evaluate(records, bins, method, subset_name):
    """Valuta il metodo su un sottoinsieme; ritorna errori medi e predizioni."""
    errors_tx, errors_ty, errors_angle = [], [], []
    predictions = []  # lista di dict con parametri predetti per ogni coppia
    for rec in records:
        img_ref, img_mov = load_image_pair(rec["dataset"], rec["pair"], rec["filename"])
        est = register_pair(img_ref, img_mov, bins=bins, method=method)
        theta_est, tx_est, ty_est = est

        err_tx = abs(tx_est - rec["tx"])
        err_ty = abs(ty_est - rec["ty"])
        err_angle = abs(theta_est - rec["angle_rad"])

        errors_tx.append(err_tx)
        errors_ty.append(err_ty)
        errors_angle.append(err_angle)
        predictions.append({"pair": rec["pair"], "theta": theta_est,
                             "tx": tx_est, "ty": ty_est})

        print(f"  [{subset_name}] {rec['pair']}/{rec['filename']}  "
              f"Δtx={err_tx:.4f}  Δty={err_ty:.4f}  Δθ={err_angle:.6f} rad")

    mean_tx = np.mean(errors_tx)
    mean_ty = np.mean(errors_ty)
    mean_angle = np.mean(errors_angle)
    return mean_tx, mean_ty, mean_angle, predictions


# ──────────────────────────────────────────────
# 8. Visualizzazione immagini allineate e differenza (consegna punto 3)
# ──────────────────────────────────────────────
def show_aligned_and_diff(records, predictions, best_config):
    """Per ogni coppia di test mostra l'immagine di riferimento, allineata e differenza."""
    print("\n" + "=" * 70)
    print("VISUALIZZAZIONE RISULTATI (Test set)")
    print("Premi un tasto su una finestra per passare alla coppia successiva.")
    print("=" * 70)

    for rec, pred in zip(records, predictions):
        img_ref, img_mov = load_image_pair(rec["dataset"], rec["pair"], rec["filename"])
        params = [pred["theta"], pred["tx"], pred["ty"]]
        aligned = apply_transform(img_mov, params, img_ref.shape)

        # Differenza assoluta e colormap
        diff_f = np.abs(img_ref.astype(np.float32) - aligned.astype(np.float32))
        diff_u8 = np.clip(diff_f, 0, 255).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)

        # Preparazione stack orizzontale (Ref | Aligned | Diff)
        # Convertiamo i grigi in BGR per lo stack con la mappa colori
        ref_bgr = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
        aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
        
        vis = np.hstack((ref_bgr, aligned_bgr, diff_color))
        
        pair = rec["pair"]
        window_name = f"Coppia {pair} (Ref | Aligned | Diff)"
        cv2.imshow(window_name, vis)
        print(f"  Visualizzando {pair}... (chiudi la finestra o premi un tasto)")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    print("\n  Visualizzazione completata.")


def save_prediction_csv(all_preds, pair_labels, csv_path):
    """Salva su CSV la matrice dei parametri predetti (θ, tx, ty) per ogni config."""
    rows = []
    for i, pair in enumerate(pair_labels):
        row = {"pair": pair}
        for (method, bins), preds in all_preds.items():
            p = preds[i]
            col_prefix = f"{method}_{bins}b"
            row[f"{col_prefix}_theta"] = round(p["theta"], 6)
            row[f"{col_prefix}_tx"]    = round(p["tx"], 4)
            row[f"{col_prefix}_ty"]    = round(p["ty"], 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("pair")
    df.to_csv(csv_path)
    print(f"  → Salvato: {csv_path}")
    return df


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

    # ── Riepilogo ricerca ──
    print("\n" + "=" * 70)
    print("RIEPILOGO RICERCA IPERPARAMETRI (Validation)")
    print("=" * 70)
    print(f"{'Metodo':<15} {'Bins':<6} {'Err Tx':<10} {'Err Ty':<10} "
          f"{'Err θ (rad)':<14} {'Totale':<10}")
    print("-" * 65)
    for r in results_summary:
        marker = " ★" if (r["method"], r["bins"]) == best_config else ""
        print(f"{r['method']:<15} {r['bins']:<6} {r['mean_tx']:<10.4f} "
              f"{r['mean_ty']:<10.4f} {r['mean_angle']:<14.6f} "
              f"{r['total']:<10.4f}{marker}")

    best_method, best_bins = best_config
    print(f"\n✓ Configurazione migliore: {best_method}, {best_bins} bin")

    # ── Creazione cartella per i risultati di questa esecuzione ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, "results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Salvataggio CSV predizioni validation (tutte le 9 combinazioni) ──
    pair_labels = [r["pair"] for r in gt["val"]]
    val_csv = os.path.join(run_dir, "predictions_val.csv")
    print("\n" + "=" * 70)
    print("MATRICE PREDIZIONI VALIDATION — parametri stimati per le 9 config")
    print("=" * 70)
    save_prediction_csv(all_val_preds, pair_labels, val_csv)

    # ── Fase 2: Valutazione finale sul test set ──
    print("\n" + "=" * 70)
    print(f"FASE 2: Valutazione sul TEST set  (Metodo={best_method}, Bins={best_bins})")
    print("=" * 70)

    mean_tx, mean_ty, mean_angle, test_preds = evaluate(
        gt["test"], best_bins, best_method, "TEST"
    )
    
    # ── Riepilogo errori per coppia sul Test Set ──
    print("\n" + "=" * 70)
    print("DETTAGLIO ERRORI PER COPPIA (Test Set)")
    print("=" * 70)
    print(f"{'Paio':<8} {'Δtx':<10} {'Δty':<10} {'Dist. px':<12} {'Δθ (deg)':<10}")
    print("-" * 60)
    for rec, pred in zip(gt["test"], test_preds):
        err_tx = abs(pred["tx"] - rec["tx"])
        err_ty = abs(pred["ty"] - rec["ty"])
        err_dist = np.sqrt(err_tx**2 + err_ty**2)
        err_angle_deg = np.degrees(abs(pred["theta"] - rec["angle_rad"]))
        print(f"{rec['pair']:<8} {err_tx:<10.4f} {err_ty:<10.4f} {err_dist:<12.4f} {err_angle_deg:<10.4f}")
    print("=" * 70)

    # ── Salvataggio CSV predizioni test (config migliore) ──
    test_pair_labels = [r["pair"] for r in gt["test"]]
    test_csv = os.path.join(run_dir, "predictions_test.csv")
    print(f"\n{'='*70}")
    print(f"MATRICE PREDIZIONI TEST — config ottimale ({best_method}, {best_bins} bin)")
    print(f"{'='*70}")
    save_prediction_csv({(best_method, best_bins): test_preds}, test_pair_labels, test_csv)
    print(f"\n{'='*70}")
    print(f"RISULTATI FINALI (TEST)")
    print(f"{'='*70}")
    print(f"  Errore medio traslazione X : {mean_tx:.4f} px")
    print(f"  Errore medio traslazione Y : {mean_ty:.4f} px")
    print(f"  Errore medio rotazione     : {mean_angle:.6f} rad")
    print(f"  Errore totale combinato    : {mean_tx + mean_ty + mean_angle:.4f}")
    print("=" * 70)

    # ── Punto 3 della consegna: immagini allineate e differenza per ogni coppia di test ──
    show_aligned_and_diff(gt["test"], test_preds, best_config)


if __name__ == "__main__":
    main()
