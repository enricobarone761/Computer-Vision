import cv2 as cv
import numpy as np
import keras
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Carica il modello della Strategia 1 e l'immagine preprocessata (256x256)
model = keras.models.load_model("Progetto_Esame/Assignment_3/Modelli_e_CF/partial_finetuned_fase2.keras")
IMAGE_PATH = r'Progetto_Esame/Assignment_3/DATASET/AIDRidotto/BaseballField/baseballfield_43.jpg'

img = cv.imread(IMAGE_PATH)
img_resized = cv.resize(img, (256, 256))
img_reshaped = img_resized.reshape(1, 256, 256, 3)  # Aggiunge una dimensione batch (1, 256, 256, 3)

y_pred_probs = model.predict(img_reshaped)[0]  # Ottiene le probabilità predette per la prima (e unica) immagine
y_pred_class = np.argmax(y_pred_probs, axis=1)

class_names = [f.name for f in Path("Progetto_Esame/Assignment_3/DATASET/UCMerced_LandUse/Images").glob('*') if f.is_dir()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#probs = y_pred_probs[0]
top5 = np.argsort(y_pred_probs)[-5:] #arg restituisce gli indici degli elementi ordinati in ordine crescente
class_names = np.array(class_names)

ax1.imshow(img_resized); ax1.axis('off')
ax1.set_title(f'Immagine\n Pred: {class_names[top5[0]]} ({y_pred_probs[top5[0]]:.2f})')

ax2.barh(class_names[top5], y_pred_probs[top5])
ax2.set(xlabel='Probabilità', title='Analisi Statistica del Classificatore (Top 5)')

fig.tight_layout()
plt.show()
plt.close()

