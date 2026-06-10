import keras
import utils

BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-4     # lr iniziale intermedio per il fine-tuning parziale
PATH = "/home/enrib/progetto/dataset/DATASET/UCMerced_LandUse/Images"

X, y = utils.load_dataset(PATH)

(X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = utils.divide_and_encode_data(X, y)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")


pretrained_model = keras.models.load_model("Progetto_Esame/Assignment_3/Modelli_e_CF/preaddestrato_aid.keras")

#TODO da aggiustare
# OPZIONE 2: Congelare solo una parte della rete (i primi 3 stage),
# lasciando addestrabili gli ultimi layer convoluzionali (Stage 4) e il MLP finale.
# Per scegliere i layer in modo preciso, cerchiamo il primo layer Conv2D 
# che utilizza 512 filtri (che identifica l'inizio dello Stage 4).
# Siccome usiamo blocchi "pre-activation", il blocco inizia effettivamente 
# con i layer di BatchNormalization e Activation che lo precedono.

start_stage4_idx = None
for i, layer in enumerate(pretrained_model.layers):
    if isinstance(layer, keras.layers.Conv2D) and layer.filters == 512:
        # Trovato il primo Conv2D dello Stage 4. 
        # Indietreggiamo di 2 posizioni per includere la BN e l'Activation pre-convoluzione.
        start_stage4_idx = i - 2
        break

for i, layer in enumerate(pretrained_model.layers):
    if i < start_stage4_idx:
        layer.trainable = False
    else:
        layer.trainable = True

print(TODO)

# rimpiazzo la testa per adattarlo al nuovo dataset
x = pretrained_model.layers[-2].output #TODO da spiegare
outputs = keras.layers.Dense(len(class_names), activation="softmax", name="ucmerced_classifier")(x)

model = keras.Model(inputs=pretrained_model.input, outputs=outputs, name="Finetune_Opt2")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=10,
        restore_best_weights=True, 
        verbose=1),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=5,
        min_lr=1e-7, 
        verbose=1),

    keras.callbacks.ModelCheckpoint(
        filepath="Progetto_Esame/Assignment_3/Modelli_e_CF/finetuned_fase2.keras",
        monitor="val_loss", 
        save_best_only=True, 
        verbose=1),

    keras.callbacks.TensorBoard(
        log_dir="Progetto_Esame/Assignment_3/logs/strategia1_fase2", 
        histogram_freq=0),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
