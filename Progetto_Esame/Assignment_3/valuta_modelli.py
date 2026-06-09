import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import utils

PATH_UCMERCED = "/home/enrib/progetto/dataset/DATASET/UCMerced_LandUse/Images"
SEED = 42

def evaluate_on_test_set(model, X_test, y_test_ohe, class_names, title_prefix="Test Set"):
    """
    Esegue l'inferenza sul test set e calcola le metriche
    (Accuracy, Precision, Recall, F1) + Confusion Matrix, usando classification_report.
    """
    print(f"\n[{title_prefix}] Avvio inferenza sul Test Set...")
    
    # 1. Predizione
    y_pred_prob = model.predict(X_test)
    y_pred_idx  = np.argmax(y_pred_prob, axis=1)
    y_true_idx  = np.argmax(y_test_ohe, axis=1)
    
    # 2. Estrazione metriche tramite classification_report (formato dict)
    report = classification_report(
        y_true=y_true_idx,
        y_pred=y_pred_idx,
        target_names=class_names,
        output_dict=True
    )
    
    acc  = report["accuracy"]
    prec = report["macro avg"]["precision"]
    rec  = report["macro avg"]["recall"]
    f1   = report["macro avg"]["f1-score"]
    
    # Stampa formattata
    print(f"\n=== Risultati {title_prefix} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (Macro-Avg)")
    print(f"Recall:    {rec:.4f} (Macro-Avg)")
    print(f"F1-Score:  {f1:.4f} (Macro-Avg)")
    print("="*30)
    
    # Stampo anche il report testuale completo a terminale
    print("\nClassification Report completo:")
    print(classification_report(y_true_idx, y_pred_idx, target_names=class_names))
    
    # 3. Matrice di confusione
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true_idx,
        y_pred=y_pred_idx,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        xticks_rotation='vertical',
        colorbar=False,
        ax=ax
    )
    titolo = (
        f"{title_prefix}\n"
        f"Accuracy: {acc:.2f} | F1-Macro: {f1:.2f}\n"
        f"Precision: {prec:.2f} | Recall: {rec:.2f}"
    )
    ax.set_title(titolo)
    
    # Rendi il layout stretto in modo che i tick verticali non vengano tagliati
    plt.tight_layout()
    
    # Salva il plot
    os.makedirs("Progetto_Esame/Assignment_3/Modelli_e_CF", exist_ok=True)
    base_name = os.path.basename(title_prefix)
    file_name = os.path.join("Progetto_Esame/Assignment_3/Modelli_e_CF", f"{base_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(file_name)
    print(f"Matrice di confusione salvata in '{file_name}'")
    plt.close()
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

print("Caricamento dataset UCMerced per la valutazione...")
X, y = utils.load_dataset(PATH_UCMERCED)
_, _, (X_test, y_test), class_names, _ = utils.prepare_ucmerced_data(X, y, seed=SEED)

print(f"Test set caricato: {X_test.shape[0]} campioni.")

# Trova tutti i modelli keras nella cartella Modelli_e_CF, escludendo il pre-addestramento su AID puro
model_files = glob.glob("Progetto_Esame/Assignment_3/Modelli_e_CF/*.keras")
models_to_evaluate = [f for f in model_files if "aid_pretrained" not in f]

if not models_to_evaluate:
    print("Nessun modello di fine-tuning o strategia 2 trovato nella cartella Modelli_e_CF (estensione .keras).")
    exit(0)

results = []

for model_file in models_to_evaluate:
    print(f"\n======================================")
    print(f"Valutazione del modello: {model_file}")
    print(f"======================================")
    
    # Carica modello
    model = keras.models.load_model(model_file)
    
    # Valuta (questo salva già la confusion matrix)
    metrics = evaluate_on_test_set(
        model=model,
        X_test=X_test,
        y_test_ohe=y_test,
        class_names=class_names,
        title_prefix=model_file.replace(".keras", "")
    )
    
    metrics["Modello"] = os.path.basename(model_file)
    results.append(metrics)

# Formatta i risultati in un DataFrame Pandas
cols = ["Modello", "Accuracy", "Precision", "Recall", "F1-Score"]
df_results = pd.DataFrame(results, columns=cols).sort_values(by="Accuracy", ascending=False)

print("\n\n" + "="*80)
print("RIEPILOGO PRESTAZIONI MODELLI SU TEST SET (UCMerced)")
print("="*80)
print(df_results)
print("="*80)
