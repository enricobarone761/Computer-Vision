import os
from pathlib import Path
import numpy as np
import cv2 as cv
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ─────────────────────────────────────────────
# 1. CONFIGURAZIONE
# ─────────────────────────────────────────────
#AID_DIR      = "./AID"          # <-- modifica col tuo path
#IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
SEED         = 42
PATH = "/home/enrib/progetto/dataset/DATASET/AIDRidotto" #per WSL
#PATH = r'Progetto_Esame/Assignment_3/DATASET/AIDRidotto'


# ─────────────────────────────────────────────
# 2. CARICAMENTO DATASET CON OPENCV
# ─────────────────────────────────────────────
def load_dataset(cartella):
    X, y = [], []
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            classe = f.parent.name
            print(f"caricata immagine {f} con classe {classe}")
            im = cv.imread(str(f))
            im = cv.resize(im, (256, 256))
            #im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            X.append(im)
            y.append(classe)
    #y = label_encoder.fit_transform(np.array(y))
    return np.array(X), np.array(y)
    #return np.array(X), y

X, y = load_dataset(PATH)

# Normalize images to [0, 1]
#X = X.astype('float32') / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)
X_val, X_val_test, y_val, y_val_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test, shuffle=True)

# Convert integer labels to one-hot encoding using OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))
y_val = ohe.transform(y_val.reshape(-1, 1))
y_val_test = ohe.transform(y_val_test.reshape(-1, 1))

# ─────────────────────────────────────────────
# 4. DATA AUGMENTATION (solo sul training set)
# ─────────────────────────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
], name="augmentation")

# ─────────────────────────────────────────────
# 5. ARCHITETTURA: RETE RESIDUALE CUSTOM
# ─────────────────────────────────────────────
def residual_block(x, filters, kernel_size=3, stride=1, use_bn=True):
    """Blocco residuale custom (NO ResNet predefinita)."""
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      padding="same", use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1,
                      padding="same", use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)

    # Proiezione shortcut se le dimensioni non corrispondono
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                  padding="same", use_bias=False)(shortcut)
        if use_bn:
            shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_model(input_shape, num_classes, name="AID_ResNet"):
    inputs = keras.Input(shape=input_shape, name="input")

    # Augmentation solo in training
    x = data_augmentation(inputs)

    # Stem
    x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)     # 64x64

    # Blocco residuale 1
    x = residual_block(x, filters=64, stride=2)  # 32x32

    # Blocco residuale 2
    x = residual_block(x, filters=128, stride=2) # 16x16

    # Blocco residuale 3 (opzionale, migliora le feature)
    x = residual_block(x, filters=256, stride=2) # 8x8

    # Flattening (NO average pooling prima del classificatore)
    x = layers.Flatten(name="flatten")(x)

    # MLP classificatore finale
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_head")(x)

    model = keras.Model(inputs, outputs, name=name)
    return model


model = build_model(
    input_shape=X.shape[1:],
    num_classes=y_train.shape[1]
)
model.summary()

# ─────────────────────────────────────────────
# 6. COMPILAZIONE
# ─────────────────────────────────────────────
optimizer = keras.optimizers.Adam(learning_rate=LR)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────
# 7. CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="aid_pretrained_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir="./logs",
        histogram_freq=0,
    )
]

# ─────────────────────────────────────────────
# 8. TRAINING
# ─────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\nPre-training completato.")
print(f"Modello migliore salvato in: aid_pretrained_best.keras")

# ─────────────────────────────────────────────
# PER VISUALIZZARE TENSORBOARD:
# Nel terminale esegui: tensorboard --logdir=./logs
# Poi apri il browser a http://localhost:6006
# ─────────────────────────────────────────────