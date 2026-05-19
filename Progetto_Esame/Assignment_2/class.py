import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Configurazione della Cross-Validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Definizione dei modelli da testare
modelli = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

risultati = []

# Ciclo principale sui file dei diversi k
for k in [50, 100, 500]:
    
    with open(rf"Progetto_Esame/Assignment_2/istogrammi/istogrammi_k{k}.pkl", 'rb') as f:
        lista_istogrammi = pickle.load(f)
    
    # Preparazione rapida delle feature e dei target
    X = np.array([istogramma for _, istogramma in lista_istogrammi])
    y = np.array([classe for classe, _ in lista_istogrammi])
    
    for nome_modello, modello in modelli.items():
        y_pred = cross_val_predict(modello, X, y, cv=skf, n_jobs=-1)
        
        # Estrazione metriche
        report = classification_report(y, y_pred, output_dict=True)
        acc  = report["accuracy"]
        prec = report["macro avg"]["precision"]
        rec  = report["macro avg"]["recall"]
        f1   = report["macro avg"]["f1-score"]
        
        ConfusionMatrixDisplay.from_predictions(y_true=y, 
                                                y_pred=y_pred, 
                                                cmap=plt.cm.Blues,
                                                xticks_rotation='vertical',
                                                colorbar=False)
        
        # Configurazione del titolo del sotto-grafico
        titolo = (
            f"{nome_modello}\n"
            f"Accuracy: {acc:.2f} | F1-Macro: {f1:.2f}\n"
            f"Precision: {prec:.2f} | Recall: {rec:.2f}"
        )

        risultati.append({
            'k': k,
            'Model': nome_modello,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        plt.title(titolo)
        plt.savefig(rf"Progetto_Esame/Assignment_2/risultati/confusion_matrix_k{k}_{nome_modello}.png", dpi=300, bbox_inches='tight', pad_inches=0.3)

# Salvataggio dei risultati in un file CSV
risultati = pd.DataFrame(risultati, index=None).sort_values(by=['Accuracy'], ascending=False)
risultati.to_csv(rf"Progetto_Esame/Assignment_2/risultati/metriche_modelli.csv")
print(risultati)