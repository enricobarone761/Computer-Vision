import cv2 as cv
import numpy as np
import keras
import matplotlib.pyplot as plt
from pathlib import Path

#carica il modello della Strategia 1 e l'immagine preprocessata (256x256)
model = keras.models.load_model("Progetto_Esame/Assignment_3/Modelli_e_CF/partial_finetuned_fase2.keras")
IMAGE_PATH = r'Progetto_Esame/Assignment_3/test_image.png'

img = cv.imread(IMAGE_PATH)
img_resized = cv.resize(img, (256, 256))
img_reshaped = img_resized.reshape(1, 256, 256, 3)  # Aggiunge la dimensione del batch (1, 256, 256, 3)

y_pred_probs = model.predict(img_reshaped)[0]  # Ottiene le probabilità predette per la prima (e unica) immagine
y_pred_class = np.argmax(y_pred_probs)

class_names = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
top5 = np.argsort(y_pred_probs)[-5:] #arg restituisce gli indici degli elementi ordinati in ordine crescente
class_names = np.array(class_names)

ax1.imshow(cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)) #matplot si aspetta immagini rgb, io le ho salvate in bgr
ax1.axis('off')
ax1.set_title(f'Immagine\n Pred: {class_names[y_pred_class]} ({y_pred_probs[y_pred_class]:.2f})')

ax2.barh(class_names[top5], y_pred_probs[top5])
ax2.set(xlabel='Probabilità', title='Analisi Statistica del Classificatore (Top 5)')

fig.tight_layout()
fig.savefig("Progetto_Esame/Assignment_3/predictions_plot.png", dpi=300, bbox_inches='tight')
print(f"Grafico salvato")
#plt.show() in wsl il plot non viene visualizzato

