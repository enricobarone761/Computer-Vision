import cv2 as cv
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import numpy as np

import tensorflow as tf
import keras
from keras import layers, Model

def load_dataset(cartella):
    X,y = [], []
    label_encoder = LabelEncoder()
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            classe = f.parent.name
            print(f"caricata immagine {f} con classe {classe}")
            im = cv.imread(str(f))
            im = cv.resize(im, (256, 256))
            #im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            X.append(im)
            y.append(classe)
    y = label_encoder.fit_transform(np.array(y))
    return np.array(X), y
    #return np.array(X), y


#PATH = r'Progetto_Esame/Assignment_3/DATASET/AIDRidotto'
PATH = "/home/enrib/progetto/dataset/DATASET/AIDRidotto" #per WSL

X, y = load_dataset(PATH)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)



model = tf.keras.applications.VGG16(
    include_top=True,
    weights=None,
    classes=len(np.unique(y)),
    input_shape=X.shape[1:]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Training del modello
print("\n=== Inizio Training ===")
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=8,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
)

# Valutazione sul test set
print("\n=== Valutazione Test Set ===")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Salvataggio modello e storia
model.save('Progetto_Esame/Assignment_3/modello_custom.keras')
