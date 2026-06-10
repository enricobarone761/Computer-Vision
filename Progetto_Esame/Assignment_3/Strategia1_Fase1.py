import os
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import utils

# Risoluzione standard 256x256 (coerente con UC Merced).
AID_SIZE = (256, 256)

# ─────────────────────────────────────────────
# 1. CONFIGURAZIONE
# ─────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
SEED         = 42
PATH = "/home/enrib/progetto/dataset/DATASET/AID" # Manteniamo il path per WSL o locale

# ─────────────────────────────────────────────
# 2. CARICAMENTO DATASET
# ─────────────────────────────────────────────
print("Caricamento dataset AID (256×256)...")
X, y = utils.load_dataset(PATH)

# Split 70/30 serve solo per il pre-training su AID
(X_train, y_train), (X_val, y_val), class_names = utils.divide_and_encode_data(X, y, only_val=True)

# ─────────────────────────────────────────────
# 3. CREAZIONE MODELLO
# ─────────────────────────────────────────────
model = utils.build_model(input_shape=X.shape[1:], num_classes=len(class_names))
model.summary()

# ─────────────────────────────────────────────
# 4. COMPILAZIONE E CALLBACKS
# ─────────────────────────────────────────────
optimizer = keras.optimizers.Adam(learning_rate=LR)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="Progetto_Esame/Assignment_3/Modelli_e_CF/aid_pretrained_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir="Progetto_Esame/Assignment_3/logs/phase1_aid",
        histogram_freq=0,
    )
]

# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
print("\nInzio addestramento Fase 1 (Pre-Training su AID)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\nPre-training completato.")
print("Modello migliore salvato in: Progetto_Esame/Assignment_3/Modelli_e_CF/aid_pretrained_best.keras")