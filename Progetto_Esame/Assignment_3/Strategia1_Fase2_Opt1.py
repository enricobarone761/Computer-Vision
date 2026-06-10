import os
import tensorflow as tf
import keras
from keras import layers
import utils

# ─────────────────────────────────────────────
# 1. CONFIGURAZIONE
# ─────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-4
SEED       = 42
PATH = "/home/enrib/progetto/dataset/DATASET/UCMerced_LandUse/Images"

# ─────────────────────────────────────────────
# 2. CARICAMENTO E PREPARAZIONE DATI
# ─────────────────────────────────────────────

X, y = utils.load_dataset(PATH)
print(f"Caricate {len(X)} immagini. Shape: {X.shape}")

(X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = utils.divide_and_encode_data(X, y)

print(f"Training set:   {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")
print(f"Test set:       {X_test.shape[0]} campioni")

# ─────────────────────────────────────────────
# 3. ADATTAMENTO MODELLO PRE-ADDESTRATO
# ─────────────────────────────────────────────
if not os.path.exists("Progetto_Esame/Assignment_3/Modelli_e_CF/aid_pretrained_best.keras"):
    print("ERRORE: Modello pre-addestrato non trovato. Esegui prima Strategia1_Fase1.py")
    exit(1)

pretrained_model = keras.models.load_model("Progetto_Esame/Assignment_3/Modelli_e_CF/aid_pretrained_best.keras")

# OPZIONE 1: Congelare i layer convoluzionali (Backbone)
# Usiamo il prefisso "mlp_" per identificare i layer Dense finali da addestrare.
# Il resto della rete (tutte le convoluzioni) rimarrà bloccato.
for layer in pretrained_model.layers:
    if layer.name.startswith("mlp_"):
        layer.trainable = True
    else:
        layer.trainable = False

print("\nLayer del backbone congelati. Solo i layer con prefisso 'mlp_' sono addestrabili.")

# Tagliamo l'ultimo layer (il Dense per AID) prendendo l'output del penultimo layer (il Dropout)
x = pretrained_model.layers[-2].output
# Aggiungiamo la nuova "testa" per UC Merced
outputs = layers.Dense(len(class_names), activation="softmax", name="ucmerced_classifier")(x)

model = keras.Model(inputs=pretrained_model.input, outputs=outputs, name="Finetune_Opt1")
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
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5,
        min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(
        filepath="Progetto_Esame/Assignment_3/Modelli_e_CF/ucmerced_finetuned_opt1_best.keras",
        monitor="val_loss", save_best_only=True, verbose=1),
    keras.callbacks.TensorBoard(
        log_dir="Progetto_Esame/Assignment_3/logs/phase2_opt1", histogram_freq=0),
]

# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
print("\nInizio fine-tuning Opzione 1 (solo MLP addestrabile)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)