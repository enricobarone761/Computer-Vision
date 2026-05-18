import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn

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

# Ciclo principale sui file dei diversi k
for k in [50, 100, 500]:
    
    with open(f"Progetto_Esame\\Assignment_2\\istogrammi\\istogrammi_k{k}.pkl", 'rb') as f:
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
        
        ConfusionMatrixDisplay.from_predictions(y, 
                                                y_pred, 
                                                cmap=plt.cm.Blues,
                                                xticks_rotation='vertical',
                                                colorbar=False)
        
        # Configurazione del titolo del sotto-grafico
        titolo_sub = (
            f"{nome_modello}\n"
            f"Acc: {acc:.2f} | F1-Macro: {f1:.2f}\n"
            f"Prec: {prec:.2f} | Rec: {rec:.2f}"
        )

        plt.title(titolo_sub)
        plt.tight_layout()
        plt.show()