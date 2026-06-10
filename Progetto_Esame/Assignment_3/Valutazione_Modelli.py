import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import utils

PATH_UCMERCED = "/home/enrib/progetto/dataset/DATASET/UCMerced_LandUse/Images"

# carichiamo il dataset per fare i test
X, y = utils.load_dataset(PATH_UCMERCED)
_, _, (X_test, y_test), class_names = utils.divide_and_encode_data(X, y)

strategia1_model = "Progetto_Esame/Assignment_3/Modelli_e_CF/partial_finetuned_fase2.keras"
strategia2_model = "Progetto_Esame/Assignment_3/Modelli_e_CF/strategia2.keras"

models = {
    "Strategia 1": strategia1_model,
    "Strategia 2": strategia2_model
}

risultati = []

for name, path in models.items():
        
    print(f"Valutazione modello: {name}")
    model = keras.models.load_model(path)

    # predizioni
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # svuoto la memoria dal modello non più necessario per il ciclo
    del model

    report = classification_report(y_true, y_pred, output_dict=True)
    acc  = report["accuracy"]
    prec = report["macro avg"]["precision"]
    rec  = report["macro avg"]["recall"]
    f1   = report["macro avg"]["f1-score"]
    
    # plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, 
        y_pred=y_pred, 
        display_labels=class_names,
        cmap=plt.cm.Blues,
        xticks_rotation='vertical',
        colorbar=False
    )

    risultati.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

    titolo = f"{name} (Acc: {acc})"
    plt.title(titolo)
    plt.savefig(rf"Progetto_Esame/Assignment_3/Modelli_e_CF/confusion_matrix_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

risultati = pd.DataFrame(risultati).sort_values(by=['Accuracy'], ascending=False)
print(risultati)
