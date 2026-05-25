import pickle
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold


#cross-validation stratificata, così ogni fold mantiene la stessa distribuzione delle classi
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

modelli = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=400),
    'SVM': SVC(kernel='rbf', random_state=42)
}

risultati = []

for k in [50, 100, 500]:
    
    with open(rf"Progetto_Esame/Assignment_2/istogrammi_BoW/istogrammi_k{k}.pkl", 'rb') as f:
        lista_istogrammi = pickle.load(f)
    
    X = [istogramma for _, istogramma in lista_istogrammi]
    y = [classe for classe, _ in lista_istogrammi]
    
    for nome_modello, modello in modelli.items():

        #cross_val_predict esegue la cross-validation e restituisce le predizioni per ogni campione,
        #ottenute dal modello addestrato sui fold di training corrispondenti.
        #il modello viene addestrato separatamente su ogni fold in jobs paralleli (n_jobs=-1).
        #ogni ciclo di addestramento e predizione è completamente indipendente dagli altri.
        #il modello viene clonato automaticamente dal dizionario prima di ogni addestramento. 
        y_pred = cross_val_predict(modello, X, y, cv=skf, n_jobs=-1)

        # y_pred = np.empty_like(y)
        # for train_idx, test_idx in skf.split(X, y):
        #     m = clone(modello)
        #     m.fit(X[train_idx], y[train_idx])
        #     y_pred[test_idx] = m.predict(X[test_idx])
        
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
        
        #titolo compatto con le metriche principali
        titolo = (
            f"{nome_modello} | k={k}\n"
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
        plt.savefig(rf"Progetto_Esame/Assignment_2/confusion_matrix/confusion_matrix_k{k}_{nome_modello}.png", dpi=300, bbox_inches='tight', pad_inches=0.3)


# Usiamo l'Accuracy come metrica principale perché il dataset è ben bilanciato tra le classi.
# Precision, Recall e F1-Score sarebbero importanti con dataset sbilanciati, ma qui sono ridondanti con Accuracy
risultati = pd.DataFrame(risultati).sort_values(by=['Accuracy'], ascending=False)
print(risultati)

#################
# di seguito addestro il modello finale con il miglior k e il miglior classificatore sull'intoro dataset 
# (senza distinzioni tra train e test set) così da poterlo utilizzare per l'inferenza su immagini fuori dal dataset
# (come suggerito a lezione)
#################

best_model_config = risultati.iloc[0]
best_k = best_model_config['k']
best_model_name = best_model_config['Model']

print(f"addestramento finale: {best_model_name} con k={best_k}")

with open(rf"Progetto_Esame/Assignment_2/istogrammi_BoW/istogrammi_k{best_k}.pkl", 'rb') as f:
    lista_istogrammi = pickle.load(f)

X = [istogramma for _, istogramma in lista_istogrammi]
y = [classe for classe, _ in lista_istogrammi]

model = clone(modelli[best_model_name])
model.fit(X, y)

with open(rf"Progetto_Esame/Assignment_2/modelli_addestrati/{best_model_name}_k{best_k}_addestrato.pkl", 'wb') as f:
    pickle.dump(model, f)
print("Modello finale salvato correttamente")
