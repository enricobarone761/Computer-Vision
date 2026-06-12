import keras
import CNN_e_Utility as utils

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

# OPZIONE 2: Fine-tuning parziale.
# congeliamo i primi 3 stage della rete per preservare i pesi delle feature di basso/medio livello.
# lasciamo invece addestrabili l'ultimo blocco convoluzionale (Stage 4) e il MLP finale.
#
# per trovare l'inizio dello Stage 4, cerchiamo il primo layer Conv2D con 512 filtri.
# poiché l'architettura adotta la struttura "pre-activation" (dove BN e Activation precedono la conv),
# indietreggiamo di 2 posizioni rispetto al layer Conv2D per includere anche questi due layer pre-convoluzione.

start_stage4_idx = None
for i, layer in enumerate(pretrained_model.layers):
    if isinstance(layer, keras.layers.Conv2D) and layer.filters == 512:
        # Trovato il primo Conv2D dello Stage 4.
        # Sottraiamo 2 all'indice per includere Batch Normalization e Activation pre-convoluzione.
        start_stage4_idx = i - 2
        break

# Iteriamo su tutti i layer per impostare la loro addestrabilità (trainable)
for i, layer in enumerate(pretrained_model.layers):
    if i < start_stage4_idx:
        layer.trainable = False  # congeliamo i layer dei primi 3 stage (pesi non aggiornabili)
    else:
        layer.trainable = True   # sblocca lo Stage 4 (i pesi verranno aggiornati nel training)


# rimpiazzo la testa per adattarlo al nuovo dataset
x = pretrained_model.layers[-2].output 
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
        patience=7,
        restore_best_weights=True, 
        verbose=1),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=3,
        min_lr=1e-7, 
        verbose=1),

    keras.callbacks.ModelCheckpoint(
        filepath="Progetto_Esame/Assignment_3/Modelli_e_CF/partial_finetuned_fase2.keras",
        monitor="val_loss", 
        save_best_only=True, 
        verbose=1),

    keras.callbacks.TensorBoard(
        log_dir="Progetto_Esame/Assignment_3/logs/strategia1_fase2_b")
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
