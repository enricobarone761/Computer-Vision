import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import utils

# ─────────────────────────────────────────────
# 1. CONFIGURAZIONE
# ─────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-4
SEED         = 42
PATH = "/home/enrib/progetto/dataset/DATASET/UCMerced_LandUse/Images"

# ─────────────────────────────────────────────
# 2. CARICAMENTO E PREPARAZIONE DATI
# ─────────────────────────────────────────────
print("Caricamento dataset UCMerced...")
X, y = utils.load_dataset(PATH)
print(f"Caricate {len(X)} immagini. Shape: {X.shape}")

(X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = utils.divide_and_encode_data(X, y)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")

# ─────────────────────────────────────────────
# 3. CREAZIONE MODELLO DA ZERO (STRATEGIA 2)
# ─────────────────────────────────────────────
print("\nCostruzione modello da zero (Strategia 2)...")
model = utils.build_model(input_shape=X_train.shape[1:], num_classes=len(class_names), name="Strategia2")
model.summary()

# ─────────────────────────────────────────────
# 4. COMPILAZIONE E CALLBACKS
# ─────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath="Progetto_Esame/Assignment_3/Modelli_e_CF/strategia2_best.keras", monitor="val_loss", save_best_only=True, verbose=1),
    keras.callbacks.TensorBoard(log_dir="Progetto_Esame/Assignment_3/logs/strategia2", histogram_freq=0)
]

# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
print("\nInizio addestramento (Strategia 2)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# La valutazione è stata spostata in valuta_modelli.py
